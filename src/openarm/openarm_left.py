# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Eric Goossen

"""OpenArm (left) 7-DOF arm constants and factory.

OpenArm is a 7R arm with real geometric asymmetries (joint4's one-sided
range, joint6's x-axis rotation) that EAIK's exact-equality decomposition
likely won't resolve cleanly. The factory passes a custom-built ssik
artifact (jointlock.seven_r, one joint locked + discretized 6R sub-solves)
to ``resolve_ik_solver`` so the ``"auto"`` chain skips EAIK and lands on
ssik (analytical). mink remains the final fallback if ssik fails to
construct for any reason.

All joint specs from ``openarm_v10.urdf.xacro``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
   from mj_environment import Environment
   from mj_manipulator.arm import Arm


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OPENARM_LEFT_JOINT_NAMES = [f"openarm_left_joint{i}" for i in range(1, 8)]

# Joint origins relative to parent (from openarm_v10.urdf.xacro).
# The RPY rotations are handled by MuJoCo's joint axis definitions at
# build time — these XYZ offsets are what positions each link relative
# to its parent.
OPENARM_LEFT_JOINT_ORIGINS = [
   [0.0, 0.0, 0.0625],      # J1: link0 → link1
   [-0.0301, 0.0, 0.06],    # J2: link1 → link2
   [0.0301, 0.0, 0.06625],  # J3: link2 → link3
   [-0.0, 0.0315, 0.15375], # J4: link3 → link4
   [0.0, -0.0315, 0.0955],  # J5: link4 → link5
   [0.0375, 0.0, 0.1205],   # J6: link5 → link6
   [-0.0375, 0.0, 0.0],     # J7: link6 → link7
]

# Joint RPY rotations from the xacro (applied as frame rotations).
OPENARM_LEFT_JOINT_RPYS = [
   [0.0, 0.0, 0.0],              # J1: link0 → link1
   [-1.57079632679, 0.0, 0.0],   # J2: link1 → link2
   [0.0, 0.0, 0.0],              # J3: link2 → link3
   [0.0, 0.0, 0.0],              # J4: link3 → link4
   [0.0, 0.0, 0.0],              # J5: link4 → link5
   [0.0, 0.0, 0.0],              # J6: link5 → link6
   [0.0, 0.0, 0.0],              # J7: link6 → link7
]

# Fudge factor to scale down the URDF's velocity limits while validating
SAFETY_SCALE = 0.1 # TODO: Calibrate

# Velocity limits (rad/s) from URDF.
OPENARM_LEFT_VELOCITY_LIMITS = np.array([16.754666, 16.754666, 5.445426, 5.445426, 20.943946, 20.943946, 20.943946]) * SAFETY_SCALE

# Acceleration limits — not published; derived from v_max / 0.1s
OPENARM_LEFT_ACCELERATION_LIMITS = np.array([167.54666, 167.54666, 54.45426, 54.45426, 209.43946, 209.43946, 209.43946]) * SAFETY_SCALE

# Position limits (rad). All seven joints are revolute with explicit
# limits from openarm_v10.urdf.xacro — no continuous joints here.
OPENARM_LEFT_LOWER = np.array([-3.490659, -3.3161253267948965, -1.570796, 0.0, -1.570796, -0.785398, -1.570796])
OPENARM_LEFT_UPPER = np.array([1.3962629999999998, 0.17453267320510335, 1.570796, 2.443461, 1.570796, 0.785398, 1.570796])

# Effort limits (N·m) from URDF.
OPENARM_LEFT_EFFORT_LIMITS = np.array([40.0, 40.0, 27.0, 27.0, 7.0, 7.0, 7.0])


# ---------------------------------------------------------------------------
# Named configurations (radians)
# ---------------------------------------------------------------------------

OPENARM_LEFT_HOME = np.zeros(7)

# Gripper travel limits (meters, per URDF).
OPENARM_LEFT_FINGER_CLOSED = 0.0   # meters — fully closed per URDF limit
OPENARM_LEFT_FINGER_OPEN = 0.044   # meters — fully open per URDF limit


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_openarm_left_arm(
   env: "Environment",
   *,
   ee_site: str = "openarm_left_ee_site",
   with_ik="auto",
   extra_arm_body_names: list[str] | None = None,
   grasp_manager=None,
) -> "Arm":
   """Create an OpenArm (left) arm with analytical IK via ssik.

   OpenArm's 7R chain has real geometric asymmetries (joint4's
   one-sided range, joint6's x-axis rotation) that EAIK's exact-equality
   decomposition likely won't resolve cleanly. The factory's "auto"
   chain falls through to a custom-built ssik artifact (jointlock.seven_r,
   one joint locked + discretized 6R sub-solves, ~50ms expected median).

   Args:
       env: MuJoCo environment containing the OpenArm model.
       ee_site: Name of the end-effector site on the arm chain (not the
           gripper — ssik solves for the arm only, link7/flange).
       with_ik: IK solver mode. "auto" tries EAIK first, then ssik,
           then mink.
       extra_arm_body_names: Bodies to treat as part of the arm for
           collision checking (e.g. gripper finger bodies).

   Returns:
       Arm ready for planning and execution.
   """
   from mj_manipulator.arm import Arm
   from mj_manipulator.arms._ik_factory import resolve_ik_solver
   from mj_manipulator.config import ArmConfig, KinematicLimits, PlanningDefaults

   from openarm import openarm_left_ik


   config = ArmConfig(
       name="openarm_left",
       entity_type="arm",
       joint_names=list(OPENARM_LEFT_JOINT_NAMES),
       kinematic_limits=KinematicLimits(
           velocity=OPENARM_LEFT_VELOCITY_LIMITS.copy(),
           acceleration=OPENARM_LEFT_ACCELERATION_LIMITS.copy(),
       ),
       ee_site=ee_site,
       extra_arm_body_names=extra_arm_body_names,
       planning_defaults=PlanningDefaults(smoothing_iterations=25),
       max_cartesian_speed=0.12,   # TODO: Calibrate
       max_cartesian_angular=0.6,  # TODO: Calibrate
   )

   arm = Arm(env, config)
   ik_solver = resolve_ik_solver(arm, with_ik=with_ik, ssik_module=openarm_left_ik)
   return Arm(env, config, ik_solver=ik_solver, grasp_manager=grasp_manager)
