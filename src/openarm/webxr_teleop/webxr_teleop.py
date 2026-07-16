"""WebXR-driven bimanual teleop logic -- shared between sim and real
hardware demos.

Both backends read from the same WebXR pose bridge (webxr_pose_bridge),
and both backends run MuJoCo -- sim uses it to actually move the robot,
real hardware uses it in parallel for collision checking and viewing
while a separate real robot context drives the actual motors. So the
frame math, clutch, rig construction, viewer, and run loop are all
shared here. The only thing a demo file supplies is the `ctx` object
(from `robot.sim()` or `robot.real()`) that a TeleopController wraps.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from mj_manipulator.event_loop import PhysicsEventLoop
from mj_manipulator.teleop import TeleopController
from mj_viser import MujocoViewer, TeleopPanel

from .teleop_config import ClutchConfig, TeleopConfig
from . import bridge as xrt

logger = logging.getLogger(__name__)

SIDES = ("left", "right")

# Rotation mapping a WebXR controller's local-floor pose into world frame.
# World convention: X-forward, Y-left, Z-up. Rotates both world axes and
# the controller's own body axes. Fixed by the WebXR reference-space
# convention -- not something a user would ever want to tune.
R_CONTROLLER_TO_WORLD = np.array([
    [0.0, 0.0, -1.0],
    [-1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
])


# ---------------------------------------------------------------------------
# Frame math -- input-source agnostic. Any SDK that reports poses as a flat
# [x, y, z, qx, qy, qz, qw] array can use these.
# ---------------------------------------------------------------------------


def quat_to_rotmat(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Hamilton quaternion (x, y, z, w) -> 3x3 rotation matrix."""
    n = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if n < 1e-8:
        return np.eye(3)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz),     2 * (qx * qy - qz * qw),     2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw),     1 - 2 * (qx * qx + qz * qz),     2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw),         2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])


def pose7_to_matrix(pose7) -> np.ndarray:
    """Convert a flat [x, y, z, qx, qy, qz, qw] pose into a 4x4 matrix."""
    x, y, z, qx, qy, qz, qw = pose7
    T = np.eye(4)
    T[:3, :3] = quat_to_rotmat(qx, qy, qz, qw)
    T[:3, 3] = [x, y, z]
    return T


def controller_pose_to_world(raw_pose_4x4: np.ndarray) -> np.ndarray:
    """Similarity transform from controller space into world space.

    Rotates both world axes and the controller's own body axes by
    R_CONTROLLER_TO_WORLD.
    """
    R = raw_pose_4x4[:3, :3]
    t = raw_pose_4x4[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R_CONTROLLER_TO_WORLD @ R @ R_CONTROLLER_TO_WORLD.T
    out[:3, 3] = R_CONTROLLER_TO_WORLD @ t
    return out


def orthonormalize(R: np.ndarray) -> np.ndarray:
    """Project a near-rotation matrix back onto SO(3) via SVD."""
    u, _, vt = np.linalg.svd(R)
    R_clean = u @ vt
    if np.linalg.det(R_clean) < 0:
        u[:, -1] *= -1
        R_clean = u @ vt
    return R_clean


# ---------------------------------------------------------------------------
# Hysteresis toggle -- general debounce primitive, not teleop-specific.
# Converts a noisy continuous scalar into a clean rising-edge toggle event.
# ---------------------------------------------------------------------------


class HysteresisToggle:
    """Fires on the rising edge of a scalar crossing `press_threshold`,
    with a separate lower `release_threshold` so noise sitting between the
    two doesn't register as press-release-press for a single real press.
    """

    def __init__(self, press_threshold: float, release_threshold: float, label: str = "") -> None:
        self._press_threshold = press_threshold
        self._release_threshold = release_threshold
        self._label = label
        self._pressed = False

    def update(self, value: float) -> bool:
        """Feed a new scalar value. Returns True on a rising edge."""
        if value >= self._press_threshold:
            pressed = True
        elif value <= self._release_threshold:
            pressed = False
        else:
            pressed = self._pressed  # dead zone -- hold prior state

        rising_edge = pressed and not self._pressed
        self._pressed = pressed
        if rising_edge:
            logger.debug("TOGGLE %s: value=%.3f", self._label, value)
        return rising_edge


# ---------------------------------------------------------------------------
# Clutch -- pose-delta tracking, genuinely teleop-specific but agnostic to
# whether `arm`/`controller` are backed by MuJoCo or real hardware.
# ---------------------------------------------------------------------------


class HandClutch:
    """Tracks engage/disengage and produces relative-motion EE targets.

    Position and rotation deltas are applied separately (rotate the
    engage-time EE orientation by the controller's rotation delta, offset
    the engage-time EE position by the controller's position delta) rather
    than composing one transform, so scaling or remapping either channel
    independently doesn't couple into the other.
    """

    def __init__(self, arm, controller, label: str, config: ClutchConfig) -> None:
        self._arm = arm
        self._controller = controller
        self._label = label
        self._config = config
        self._gripper_toggle = HysteresisToggle(
            config.trigger_press_threshold, config.trigger_release_threshold, label
        )
        self._engaged = False
        self._R_ctrl_engage: np.ndarray | None = None
        self._p_ctrl_engage: np.ndarray | None = None
        self._T_ee_engage: np.ndarray | None = None
        self._last_target: np.ndarray | None = None

    @property
    def engaged(self) -> bool:
        return self._engaged

    def update(self, ctrl_pose_world: np.ndarray, grip: float, trigger: float) -> None:
        grip_held = grip >= self._config.grip_engage_threshold

        if grip_held and not self._engaged:
            self._engaged = True
            self._R_ctrl_engage = ctrl_pose_world[:3, :3].copy()
            self._p_ctrl_engage = ctrl_pose_world[:3, 3].copy()
            self._T_ee_engage = self._arm.get_ee_pose().copy()
            self._last_target = self._T_ee_engage.copy()
            logger.debug(
                "ENGAGE %s: det=%.4f ee_pos=%s",
                self._label,
                np.linalg.det(self._R_ctrl_engage),
                self._T_ee_engage[:3, 3],
            )

        elif not grip_held and self._engaged:
            self._engaged = False

        if self._engaged and self._T_ee_engage is not None:
            self._update_target(ctrl_pose_world)

        if self._gripper_toggle.update(trigger):
            self._controller.toggle_gripper()

    def _update_target(self, ctrl_pose_world: np.ndarray) -> None:
        R_delta = ctrl_pose_world[:3, :3] @ self._R_ctrl_engage.T
        p_delta = self._config.position_scale * (ctrl_pose_world[:3, 3] - self._p_ctrl_engage)

        target = np.eye(4)
        target[:3, :3] = orthonormalize(R_delta @ self._T_ee_engage[:3, :3])
        target[:3, 3] = self._T_ee_engage[:3, 3] + p_delta

        if np.any(~np.isfinite(target)):
            return

        if self._last_target is not None:
            step_vec = target[:3, 3] - self._last_target[:3, 3]
            step = np.linalg.norm(step_vec)
            if step > self._config.max_step_m:
                # Clamp the step rather than rejecting the tick outright,
                # so the arm keeps making forward progress every tick
                # instead of freezing until a disengage/re-engage resets
                # the anchor.
                logger.debug("%s: clamped %.3fm jump to %.3fm", self._label, step, self._config.max_step_m)
                target[:3, 3] = self._last_target[:3, 3] + step_vec * (self._config.max_step_m / step)

        self._last_target = target.copy()
        self._controller.set_target_pose(target)


# ---------------------------------------------------------------------------
# Shared per-tick bridge read -- both sim and real read from the same
# WebXR bridge, so this step is common to both demos.
# ---------------------------------------------------------------------------


def poll_and_update_clutches(clutches: dict[str, HandClutch]) -> None:
    """Read both sides' current state from the WebXR bridge and feed each
    side's HandClutch. Call once per control tick from either demo.
    """
    for side in SIDES:
        state = xrt.get_state(side)
        ctrl_pose_world = controller_pose_to_world(pose7_to_matrix(state["pose7"]))
        clutches[side].update(ctrl_pose_world, state["grip"], state["trigger"])


# ---------------------------------------------------------------------------
# Rig construction -- shared between sim and real. `ctx` is whatever
# `robot.sim()` or `robot.real()` hands to TeleopController; everything
# below (arms, clutches, viewer panels) is identical either way.
# ---------------------------------------------------------------------------


class TeleopRig:
    """Bundles the per-side controllers, clutches, and viewer panels for
    a bimanual teleop session, plus the MuJoCo viewer itself.
    """

    def __init__(self, robot, ctx, loop: PhysicsEventLoop, config: TeleopConfig) -> None:
        self.loop = loop
        self.arms = {side: robot.arms[side] for side in SIDES}
        self.controllers = {
            side: TeleopController(self.arms[side], ctx) for side in SIDES
        }
        for controller in self.controllers.values():
            controller.activate()

        clutch_configs = {"left": config.left_clutch, "right": config.right_clutch}
        self.clutches = {
            side: HandClutch(
                self.arms[side], self.controllers[side], side.upper(), clutch_configs[side]
            )
            for side in SIDES
        }
        self.panels = {
            side: TeleopPanel(
                arm=self.arms[side],
                controller=self.controllers[side],
                model=robot.model,
                data=robot.data,
                gripper_body_prefix=f"openarm_{side}_",
                arm_label=f"{side.title()} Arm",
            )
            for side in SIDES
        }

        for side in SIDES:
            self.loop.register_teleop(self.controllers[side], self.panels[side])

        self.viewer = MujocoViewer(robot.model, robot.data)
        for panel in self.panels.values():
            self.viewer.add_panel(panel)


def run_teleop(rig: TeleopRig, config: TeleopConfig) -> None:
    """Connect the WebXR bridge and run the control/render loop until the
    viewer is closed. Shared by sim and real -- rig.loop.tick() drives
    sim motion directly for the sim backend, and drives MuJoCo's
    collision-checked shadow state (while the real robot context handles
    actual motor commands separately) for the real backend.
    """
    rig.viewer.launch_passive()

    xrt.init(
        host=config.bridge.host,
        ws_port=config.bridge.ws_port,
        http_port=config.bridge.http_port,
        cert_file=str(config.bridge.cert_file),
        key_file=str(config.bridge.key_file),
    )

    if not xrt.wait_for_data(timeout=config.timing.wait_for_data_timeout):
        logger.warning("No pose data received within timeout -- check headset connection.")

    next_time = time.perf_counter()
    last_render = next_time
    period = config.timing.period

    try:
        while rig.viewer.is_running():
            poll_and_update_clutches(rig.clutches)
            rig.loop.tick()

            next_time += period
            remaining = next_time - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)
            else:
                next_time = time.perf_counter()

            now = time.perf_counter()
            if now - last_render > 1 / config.timing.rendered_fps:
                rig.viewer.sync()
                last_render = now

    except KeyboardInterrupt:
        pass
    finally:
        xrt.close()
        rig.viewer.close()