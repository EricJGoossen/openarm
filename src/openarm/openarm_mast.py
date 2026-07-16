"""Mast linear control."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import mujoco
import numpy as np
from mj_manipulator import Arm, GraspManager, Trajectory
from mj_manipulator.trajectory import create_linear_trajectory

from openarm.config import OpenarmMastConfig

class OpenarmMast:
    model: mujoco.MjModel
    data: mujoco.MjData
    config: OpenarmMastConfig

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        config: OpenarmMastConfig,
        arms: list[Arm],
    ):
        self.model: mujoco.MjModel = model
        self.data: mujoco.MjData = data
        self.config: OpenarmMastConfig = config
        self._arms = arms

        # Get joint ID and qpos index
        self._joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            config.joint_name,
        )
        if self._joint_id == -1:
            raise ValueError(f"Joint '{config.joint_name}' not found in model")
        self._qpos_idx = model.jnt_qposadr[self._joint_id]

        # Get actuator ID
        self._actuator_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            config.actuator_name,
        )
        if self._actuator_id == -1:
            raise ValueError(f"Actuator '{config.actuator_name}' not found")

        # Build set of body IDs for this arm (for collision checking)
        self._arm_body_ids: set[int] = set()
        self._build_arm_body_ids()

        # Bodies belonging to *other* robot-self chains (e.g. the OTHER arm
        # on a bimanual robot) that should be treated as self-contacts when
        # checking arm-environment collisions. Wired by the owning robot
        # after all bases are constructed; defaults to empty (single-arm
        # robots).
        self._other_robot_body_ids: set[int] = set()

    def _build_arm_body_ids(self) -> None:
        """Build set of body IDs that belong to this arm including gripper."""
        for arm in self._arms:
            for joint_name in arm.config.joint_names:
                joint_id = mujoco.mj_name2id(
                    self.model,
                    mujoco.mjtObj.mjOBJ_JOINT,
                    joint_name,
                )
                if joint_id != -1:
                    body_id = self.model.jnt_bodyid[joint_id]
                    self._arm_body_ids.add(body_id)
                    self._add_child_bodies(body_id)

    def _add_child_bodies(self, parent_id: int) -> None:
        """Recursively add child bodies to arm body set."""
        for i in range(self.model.nbody):
            if self.model.body_parentid[i] == parent_id and i not in self._arm_body_ids:
                self._arm_body_ids.add(i)
                self._add_child_bodies(i)

    # TODO: Finish class

    # Height access

    def get_height(self) -> float:
        """Current height in meters."""
        return float(self.data.qpos[self._qpos_idx])