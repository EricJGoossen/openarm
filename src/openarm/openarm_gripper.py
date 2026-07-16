# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Eric Goossen

"""OpenArm parallel-jaw gripper implementation.

Two prismatic finger joints (openarm_{left,right}_finger_joint1/2), each
with range [0.0, 0.044]. Confirmed from the source URDF: finger_joint2
carries a <mimic joint="...finger_joint1"/> tag, meaning it's a simple,
explicitly-declared linear follower of finger_joint1 -- not an
underactuated linkage requiring trajectory replay (contrast with
RobotiqGripper's 4-bar linkage, which genuinely needs one). Setting
both joints' qpos directly to the same target value is correct and
sufficient, matching FrankaGripper's approach rather than Robotiq's.

NOTE: left and right gripper actuators are different MuJoCo actuator
types in the vendor XML (left: <motor>, right: <position>) -- this
class only needs to resolve them by name (works regardless of type),
but this asymmetry will matter if you ever switch from kinematic mode
(physics=False) to physics=True, since motor vs position actuators
expect very different ctrl semantics. Not addressed here -- flagging
for whoever wires up physics mode later.

Also note: right_finger1_ctrl/right_finger2_ctrl in the vendor XML use
class="motor_finge" (typo, missing the "r"), so they're silently NOT
getting the motor_finger default class's gear/ctrlrange -- irrelevant
in kinematic mode (this class doesn't use ctrl values at all), but
will bite whoever enables physics mode for the right arm's gripper
until that vendor XML typo is fixed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
import numpy as np

from mj_manipulator.grippers._base import _BaseGripper

from openarm.config import OpenarmGripperSpec

if TYPE_CHECKING:
    from mj_manipulator.grasp_manager import GraspManager


class OpenarmGripper(_BaseGripper):
    """OpenArm parallel-jaw gripper, shared between left and right arms.

    Args:
        model: MuJoCo model.
        data: MuJoCo data.
        arm_name: Semantic identifier for GraspManager/protocol bookkeeping
            (e.g. "openarm_left"). Does NOT determine MuJoCo naming --
            see config's body_prefix/actuator_prefix for that.
        grasp_manager: Optional grasp state tracker.
        config: Gripper config (travel limits, naming prefixes). If None,
            derived from arm_name via OpenArmGripperConfig.for_arm(),
            matching today's single bimanual scene's vendor XML naming.
            Pass an explicit config for any other scene layout.
    """

    hand_type: str = "openarm"
    # UNVERIFIED: assumed True (fingers meet at fully-closed, same as
    # Franka) based on the 0.0-0.044 range being symmetric per-finger
    # rather than Robotiq-style large travel -- not confirmed against
    # the actual mesh geometry.
    empty_at_fully_closed: bool = True

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        arm_name: str,
        grasp_manager: GraspManager | None = None,
        config: OpenarmGripperSpec | None = None,
    ):
        # arm_name is the semantic identifier used for GraspManager/protocol
        # bookkeeping (matches Franka/Robotiq's use of arm_name), kept
        # separate from the MuJoCo naming prefixes below -- those come
        # entirely from config now, not derived from arm_name's text.
        # Falls back to today's known-correct vendor-XML naming if no
        # config is given, so existing call sites (create_openarm_left_arm
        # etc.) keep working unchanged.
        self._config = config or OpenarmGripperSpec.for_arm(arm_name)

        body_prefix = self._config.body_prefix
        actuator_prefix = self._config.actuator_prefix

        actuator_name = f"{actuator_prefix}finger1_ctrl"
        body_names = [f"{body_prefix}hand", f"{body_prefix}left_finger", f"{body_prefix}right_finger"]
        attachment_body = f"{body_prefix}hand"

        super().__init__(
            model=model,
            data=data,
            arm_name=arm_name,
            actuator_name=actuator_name,
            gripper_body_names=body_names,
            attachment_body=attachment_body,
            ctrl_open=self._config.finger_open,
            ctrl_closed=self._config.finger_closed,
            grasp_manager=grasp_manager,
        )

        self._finger_qpos_indices: list[int] = []
        for i in (1, 2):
            joint_name = f"{body_prefix}finger_joint{i}"
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id != -1:
                self._finger_qpos_indices.append(model.jnt_qposadr[joint_id])

    def _apply_kinematic_position(self, t: float) -> None:
        if not self._finger_qpos_indices:
            return
        open_pos = self._config.finger_open
        closed_pos = self._config.finger_closed
        pos = open_pos + t * (closed_pos - open_pos)
        for idx in self._finger_qpos_indices:
            self._data.qpos[idx] = pos
        mujoco.mj_forward(self._model, self._data)

    def get_actual_position(self) -> float:
        """Get actual gripper position (0=open, 1=closed).

        Reads finger_joint1's position and maps from [finger_open, finger_closed]
        to [0, 1], same convention as FrankaGripper.
        """
        if not self._finger_qpos_indices:
            return 0.0

        open_pos = self._config.finger_open
        closed_pos = self._config.finger_closed
        if abs(open_pos - closed_pos) < 1e-8:
            return 0.0

        finger_pos = self._data.qpos[self._finger_qpos_indices[0]]
        t = (open_pos - finger_pos) / (open_pos - closed_pos)
        return float(np.clip(t, 0.0, 1.0))