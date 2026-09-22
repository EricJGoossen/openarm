"""Robot configuration for Openarm."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from mj_manipulator.config import KinematicLimits, PhysicsConfig

# Try to import openarm_assets for model paths, fall back to None if not installed
try:
    from openarm_assets import get_generated_model_path
except ImportError:
    get_generated_model_path = None

# Fudge factor to scale down the URDF's velocity limits while validating.
#
# The unscaled numbers below (16.75/5.45/20.94 rad/s, confirmed against
# vendor/urdf/example/v1.urdf's own <limit velocity="..."> fields -- they
# are not a typo) are the motors' rated max speeds, not a safe pace for
# planning/retiming real trajectories during hardware bring-up: at
# SAFETY_SCALE=1, TOPP-RA retimes a plan_to_configuration() path as fast as
# these limits allow, producing trajectories on the order of 0.1s for a
# ~0.3 rad move (~2-3 rad/s effective) -- far faster than this arm's
# impedance controller can track, and much faster than the ~0.15 rad/s
# pace already validated as safe in Stage 3 hw_validation
# (stage3_open_loop_motion.py's ramp_duration_s=2.0 for delta up to 0.3
# rad).
#
# 0.02 (peak ~0.5 rad/s) was the first estimate; real Stage 4 tracking data
# showed joints 5-7 (kp=10, vs kp=70/60 for joints 1-4 -- see
# control_gains.yaml) converging too slowly to reach tolerance before the
# trajectory's nominal duration elapsed, even though 1-4 were fine. That
# turned out to be static friction/stiction, not raw bandwidth -- the real
# fix was stage4_trajectory_execution.py's --settle-time-s (extra wait
# after nominal duration before checking convergence, since the controller
# declares SUCCESS on elapsed time, not on actually converging), which
# resolved it independent of trajectory speed. With that in place, back to
# 0.02 (peak ~0.5 rad/s, still ~30-50x under rated max) for a faster test
# cycle. Raise further only once tracked execution at this pace is
# confirmed good and a deliberate decision is made to allow faster motion.
SAFETY_SCALE = 0.02

# ---------------------------------------------------------------------------
# Arm specification
# ---------------------------------------------------------------------------


def _default_arm_kinematic_limits() -> KinematicLimits:
    """Default per-joint velocity/acceleration limits for an OpenArm arm.

    Returns a KinematicLimits directly -- no intermediate wrapper class.
    A dataclass field's default_factory must be a single zero-arg callable
    that produces the final value; the previous OpenarmArmKinematicLimits
    wrapper (construct-then-read-a-property) couldn't be expressed that
    way in one step, which is what caused the AttributeError. This
    function is the same information with no indirection.
    """
    return KinematicLimits(
        velocity=np.array(
            [16.754666, 16.754666, 5.445426, 5.445426, 20.943946, 20.943946, 20.943946]
        )
        * SAFETY_SCALE,
        acceleration=np.array(
            [167.54666, 167.54666, 54.45426, 54.45426, 209.43946, 209.43946, 209.43946]
        )
        * SAFETY_SCALE,
    )


@dataclass
class OpenarmArmSpec:
    """Specification for an Openarm arm."""

    prefix: str  # e.g., "openarm_left" or "openarm_right"
    ee_site: str = ""  # MuJoCo site name for the end-effector
    gripper_prefix: str = ""  # e.g., "left/gripper/"
    hand_type: str = "openarm"  # e.g., "openarm", "robotiq", etc.
    ik_solver: Literal["auto", "eaik", "ssik", "mink", "none"] = "auto"
    ssik_module: str = "openarm.IK.openarm_ssik"  # Python module for ssik
    kinematic_limits: KinematicLimits = field(default_factory=_default_arm_kinematic_limits)


# ---------------------------------------------------------------------------
# Gripper specification
# ---------------------------------------------------------------------------


@dataclass
class OpenarmGripperSpec:
    """Gripper travel limits, shared by both left and right arms.

    Values sourced from the vendor URDF's finger_joint1/2 <limit> tags
    (lower="0.0" upper="0.044") -- both fingers use the same range, and
    both arms use the same gripper hardware, so one config instance
    covers both sides.
    """

    finger_open: float = 0.044
    finger_closed: float = 0.0
    body_prefix: str = ""
    actuator_prefix: str = ""

    @classmethod
    def for_arm(cls, arm_name: str, **overrides) -> "OpenarmGripperSpec":
        """Convenience constructor matching today's vendor XML naming.

        Args:
            arm_name: "openarm_left" or "openarm_right".
            **overrides: any other OpenarmGripperSpec field (e.g.
                finger_open) to override from its default.
        """
        side = arm_name.replace("openarm_", "", 1)
        defaults = {
            "body_prefix": f"{arm_name}_",
            "actuator_prefix": f"{side}_",
        }
        defaults.update(overrides)
        return cls(**defaults)


# ---------------------------------------------------------------------------
# Mast Base
# ---------------------------------------------------------------------------


def _default_mast_kinematic_limits() -> KinematicLimits:
    """Default velocity/acceleration limits for the Mast linear base.

    Same fix as _default_arm_kinematic_limits above -- returns
    KinematicLimits directly, no wrapper class, no chained-property bug.
    """
    return KinematicLimits(
        velocity=np.array(0.1),  # m/s -- TODO: Calibrate
        acceleration=np.array(0.2),  # m/s^2 -- TODO: Calibrate
    )


@dataclass
class OpenarmMastConfig:
    """Configuration for a Mast base."""

    name: str
    joint_name: str  # MuJoCo joint name
    actuator_name: str = ""  # MuJoCo actuator name
    height_range: tuple[float, float] = (0.0, 0.5)  # meters (min, max) TODO: Calibrate
    collision_check_resolution: float = 0.01  # meters between collision checks
    kinematic_limits: KinematicLimits = field(default_factory=_default_mast_kinematic_limits)


# ---------------------------------------------------------------------------
# Debug logging
# ---------------------------------------------------------------------------


@dataclass
class DebugConfig:
    """Debug logging configuration.

    Controls which subsystems emit debug-level log messages.
    Use OPENARM_DEBUG=subsystem1,subsystem2 or OPENARM_DEBUG=all.

    ``verbose`` enables behavior tree visualization after each
    primitive (pickup, place, go_home), showing which nodes
    succeeded/failed. Can also be overridden per-call::

        robot.pickup("can_0", verbose=True)  # override for one call
        robot.config.debug.verbose = True    # enable globally
    """

    verbose: bool = False  # show BT tree status after each primitive
    planning: bool = False
    primitives: bool = False

    show_timestamps: bool = True
    show_module: bool = True

    def enable_all(self) -> None:
        self.verbose = True
        self.planning = True
        self.primitives = True

    def get_enabled_subsystems(self) -> list[str]:
        return [s for s in ("planning", "primitives") if getattr(self, s)]

    @classmethod
    def from_env(cls) -> "DebugConfig":
        """Create config from OPENARM_DEBUG environment variable."""
        config = cls()
        debug_env = os.environ.get("OPENARM_DEBUG", "")
        if debug_env:
            if debug_env.lower() == "all":
                config.enable_all()
            else:
                for s in debug_env.split(","):
                    s = s.strip()
                    if s and hasattr(config, s):
                        setattr(config, s, True)
        return config


_SUBSYSTEM_LOGGERS = {
    # "planning" covers both this package's own robot.py AND
    # mj_manipulator.arm_group, which is where CBiRRT planning failures are
    # actually diagnosed (_plan_frame_sequence logs a specific reason --
    # "no collision-free combined goal", "no path", or the raised exception
    # -- on every failed attempt, including each of the up-to-10 seed
    # retries plan_to_configuration() does internally). Without this,
    # OPENARM_DEBUG=planning only unmutes openarm.robot, which never sees
    # those messages -- mj_manipulator is a separate top-level logger
    # namespace, not a child of "openarm", so it needs its own handler,
    # not just a level bump (a level bump alone would still fall through to
    # logging.lastResort, which is WARNING-level and drops INFO anyway).
    "planning": ("openarm.robot", "mj_manipulator"),
    "primitives": ("openarm.primitives",),
    "affordances": ("openarm.affordances",),
}


def setup_logging(config: DebugConfig | None = None) -> None:
    """Configure openarm loggers based on debug config."""
    if config is None:
        config = DebugConfig.from_env()

    fmt_parts = []
    if config.show_timestamps:
        fmt_parts.append("%(asctime)s")
    fmt_parts.append("%(levelname)s")
    if config.show_module:
        fmt_parts.append("[%(name)s]")
    fmt_parts.append("%(message)s")
    formatter = logging.Formatter(" - ".join(fmt_parts))

    root_logger = logging.getLogger("openarm")
    root_logger.propagate = False

    if not root_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)

    root_logger.setLevel(logging.WARNING)

    for subsystem in config.get_enabled_subsystems():
        for logger_name in _SUBSYSTEM_LOGGERS.get(subsystem, ()):
            subsystem_logger = logging.getLogger(logger_name)
            subsystem_logger.setLevel(logging.DEBUG)
            if not logger_name.startswith("openarm") and not subsystem_logger.handlers:
                # Separate top-level namespace (e.g. mj_manipulator) --
                # doesn't inherit "openarm"'s handler via propagation, and
                # without a handler of its own, a message here falls
                # through to logging.lastResort (WARNING-level) and is
                # silently dropped even though the level check above passed.
                subsystem_handler = logging.StreamHandler()
                subsystem_handler.setFormatter(formatter)
                subsystem_logger.addHandler(subsystem_handler)
                subsystem_logger.propagate = False


# ---------------------------------------------------------------------------
# Top-level Openarm configuration
# ---------------------------------------------------------------------------


# Openarm joint name suffixes (combined with arm prefix). NOTE: prefix
# already ends without a trailing underscore ("openarm_left"), and these
# suffixes already start with one ("_joint1") -- f"{prefix}{suffix}"
# below (NOT f"{prefix}_{suffix}") to avoid a double underscore.
_OPENARM_JOINT_SUFFIXES = [f"_joint{i}" for i in range(1, 8)]


@dataclass
class PlanningConfig:
    """Planning parameters -- single source of truth for timeouts etc."""

    timeout: float = 30.0  # seconds per planning attempt
    base_heights: list[float] = field(default_factory=lambda: [0.2, 0.0, 0.4])
    lift_height: float = 0.05  # meters to lift after grasping


@dataclass
class OpenarmConfig:
    """Full robot configuration."""

    model_path: Path
    left_arm: OpenarmArmSpec
    right_arm: OpenarmArmSpec
    left_gripper: OpenarmGripperSpec | None = None
    right_gripper: OpenarmGripperSpec | None = None
    base: OpenarmMastConfig | None = None
    physics_config: PhysicsConfig | None = None
    named_poses: dict[str, dict[str, list[float]]] = field(default_factory=dict)
    planning: PlanningConfig = field(default_factory=PlanningConfig)
    debug: DebugConfig = field(default_factory=DebugConfig.from_env)

    def joint_names(self, arm_spec: OpenarmArmSpec) -> list[str]:
        """Get prefixed OpenArm joint names for an arm spec.

        NOTE: fixed double-underscore bug -- arm_spec.prefix is
        "openarm_left" (no trailing underscore) and each suffix already
        starts with "_" (e.g. "_joint1"), so plain concatenation gives
        "openarm_left_joint1", not "openarm_left__joint1".
        """
        return [f"{arm_spec.prefix}{j}" for j in _OPENARM_JOINT_SUFFIXES]

    def gripper_spec_for_arm(self, arm_spec: OpenarmArmSpec) -> OpenarmGripperSpec | None:
        """Get gripper spec for an arm spec."""
        if arm_spec.prefix == "openarm_left":
            return self.left_gripper
        elif arm_spec.prefix == "openarm_right":
            return self.right_gripper
        else:
            return None

    @classmethod
    def default(cls) -> "OpenarmConfig":
        """Create default configuration for Openarm with its native gripper.

        UNVERIFIED: finger_open/finger_closed values below match
        OpenArmGripperConfig.for_arm()'s convention from earlier in this
        project (finger_open=0.044, finger_closed=0.0, per the vendor
        URDF's finger_joint1/2 <limit lower="0.0" upper="0.044"/>) --
        a previous version of this method had these two values swapped.
        Double-check against the real hardware/URDF before trusting this
        if gripper open/close ever look inverted in sim.
        """
        if get_generated_model_path is None:
            raise ImportError("openarm_assets package not found. Install it with:\n  uv add openarm_assets")
        return cls(
            model_path=get_generated_model_path(sides="bimanual"),
            left_arm=OpenarmArmSpec(
                prefix="openarm_left",
                ee_site="openarm_left_ee_site",
                gripper_prefix="openarm_left_gripper",
                ik_solver="auto",
                ssik_module="openarm.IK.openarm_left_ik",
            ),
            right_arm=OpenarmArmSpec(
                prefix="openarm_right",
                ee_site="openarm_right_ee_site",
                gripper_prefix="openarm_right_gripper",
                ik_solver="auto",
                ssik_module="openarm.IK.openarm_right_ik",
            ),
            left_gripper=OpenarmGripperSpec.for_arm("openarm_left"),
            right_gripper=OpenarmGripperSpec.for_arm("openarm_right"),
            # NOTE: no base= here -- unlike Geodude's Vention linear-actuator
            # base, OpenArm's current MJCF mounts the torso rigidly to world
            # via a fixed joint (openarm_body_world_joint); there's no linear
            # mast/base hardware to reference. base defaults to None. If a
            # real height-adjustable base is added to the model later,
            # construct an OpenarmMastConfig here with the actual joint/
            # actuator names from that MJCF -- not copied from Geodude's.
        )

    @classmethod
    def from_yaml(cls, path: Path) -> "OpenarmConfig":
        """Load configuration from YAML file."""
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)

        left_gripper = None
        if "left_gripper" in data:
            left_gripper = OpenarmGripperSpec(**data["left_gripper"])
        right_gripper = None
        if "right_gripper" in data:
            right_gripper = OpenarmGripperSpec(**data["right_gripper"])

        base = None
        if "base" in data:
            base = OpenarmMastConfig(**data["base"])

        return cls(
            model_path=Path(data["model_path"]),
            left_arm=OpenarmArmSpec(**data["left"]),
            right_arm=OpenarmArmSpec(**data["right"]),
            left_gripper=left_gripper,
            right_gripper=right_gripper,
            base=base,
            named_poses=data.get("named_poses", {}),
        )
    
    def to_hardware_config(self) -> "HardwareConfig":
        """Build a mj_manipulator_ros HardwareConfig from this robot config.

        Arm names ("left"/"right") must match your real ROS 2 nodes' topic/
        action namespacing (see mj_manipulator_ros.interfaces): e.g. "left"
        -> /left_joint_trajectory_controller/follow_joint_trajectory
        /left_gripper_controller/gripper_cmd,
        /left_controller/joint_commands.
        """
        from mj_manipulator_ros.config import ArmHardwareConfig, HardwareConfig

        def _arm_cfg(spec: OpenarmArmSpec, name: str) -> ArmHardwareConfig:
            gripper = self.gripper_spec_for_arm(spec)
            return ArmHardwareConfig(
                name=name,
                joint_names=self.joint_names(spec),
                has_gripper=gripper is not None,
                gripper_open=gripper.finger_open if gripper else 0.0,
                gripper_closed=gripper.finger_closed if gripper else 0.0,
                joint_trajectory_controller=f"{name}_joint_trajectory_controller",
                gripper_interface="follow_joint_trajectory" if gripper is not None else "gripper_command",
                gripper_joint_name=f"openarm_{name}_finger_joint1" if gripper is not None else None,
            )

        return HardwareConfig(
            arms=[
                _arm_cfg(self.left_arm, "left"),
                _arm_cfg(self.right_arm, "right"),
            ],
        )