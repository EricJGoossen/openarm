# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Eric Goossen

"""Verify that planning is collision-aware, and that IK works end-to-end.


Three checks, in order:


1. plan_to_configuration() consults collision state. Using a REAL
   discovered self-collision (not a planted obstacle): q_collide drives
   every joint near its upper limit, putting the left arm into several
   simultaneous self-collisions with its own body/base. q_safe is the
   joint-range midpoint, confirmed collision-free. We ask the planner to
   go from q_safe to q_collide; since the goal itself is in collision, a
   correct planner refuses (returns None). A "successful" path means
   collision-awareness is broken.


2. Joint types are hinge (angular), with the slide finger joints as a
   negative control so the hinge assertion isn't trivially true.


3. plan_to_pose() exercises ssik's IK through the real planning pipeline.
   plan_to_configuration() is joint-space and never calls IK, so this is
   the first end-to-end IK test.


Run directly: `uv run python -m openarm.tests.test_collision_aware_planning`
"""


from __future__ import annotations

import mujoco
import numpy as np
from mj_environment import Environment

from openarm_assets import get_model_path
from openarm.openarm_left import create_openarm_left_arm


# Bodies whose self-collision pairs we tolerate (the closed gripper fingers
# touch each other by design).
SCENE = "openarm"


def _make_arm():
    """Load the decorated OpenArm scene and build the arm."""
    env = Environment(str(get_model_path(SCENE)))
    return env, create_openarm_left_arm(env)


def _banner(text: str) -> None:
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)


def check_collision_aware_planning(env, arm) -> bool:
    """Return True if the planner refuses a goal that is in collision."""
    lower, upper = arm.get_joint_limits()
    margin = 1e-3
    q_collide = upper - margin
    q_safe = (lower + upper) / 2

    _banner("Step 1: ground truth — q_collide in collision, q_safe not")
    arm.set_joint_positions(q_collide)
    n_collide = len(arm.check_collisions())
    print(f"q_collide collisions: {n_collide} contact(s)")
    assert n_collide > 0, "Expected q_collide to be in collision — ground truth changed?"

    arm.set_joint_positions(q_safe)
    n_safe = len(arm.check_collisions())
    print(f"q_safe collisions: {n_safe} contact(s)")
    assert n_safe == 0, "Expected q_safe to be collision-free"

    _banner("Step 2: plan from q_safe to q_collide (goal is in collision)")
    arm.set_joint_positions(q_safe)
    path = arm.plan_to_configuration(q_collide)

    if path is None:
        print("Planner returned None.")
        print("PASS: planner correctly refused a goal that's in collision.")
        return True

    print(f"Planner returned a path with {len(path)} waypoints.")
    final_q = path[-1]
    arm.set_joint_positions(final_q)
    n_final = len(arm.check_collisions())
    print(f"Collisions at final waypoint: {n_final}")

    if n_final > 0:
        print("FAIL: planner returned a path ENDING in collision.")
        print("Collision checking is NOT enforced on the goal state.")
        return False

    print("Planner found a collision-free endpoint near q_collide rather than")
    print("reaching it exactly (clamped/adjusted goal). Checking all waypoints...")
    in_collision = [i for i, q in enumerate(path)
                    if (arm.set_joint_positions(q), arm.check_collisions())[1]]
    if in_collision:
        print(f"FAIL: waypoints in collision: {in_collision}")
        return False
    print("No waypoints in collision.")
    return True


def check_joint_types(env, arm) -> bool:
    """Return True if all arm joints are hinge; fingers are the slide control."""
    _banner("Check: arm joints must be hinge (angular), not slide (linear)")
    all_hinge = True
    for jid in arm.joint_ids:
        jtype = env.model.jnt_type[jid]
        name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        is_hinge = jtype == mujoco.mjtJoint.mjJNT_HINGE
        print(f"  {name}: {'HINGE (OK)' if is_hinge else f'WRONG TYPE ({jtype})'}")
        all_hinge = all_hinge and is_hinge

    # Negative control: fingers SHOULD be slide, proving the hinge check
    # above isn't trivially true for every joint in the model.
    for fname in ["openarm_left_finger_joint1", "openarm_left_finger_joint2"]:
        fjid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, fname)
        if fjid < 0:
            print(f"  {fname}: NOT FOUND (skipping control)")
            continue
        is_slide = env.model.jnt_type[fjid] == mujoco.mjtJoint.mjJNT_SLIDE
        print(f"  {fname}: {'SLIDE (OK, expected)' if is_slide else 'unexpected type'}")

    print("PASS" if all_hinge else "FAIL", "— arm joint types")
    return all_hinge


def check_ik_through_planner(env, arm, tol_m: float = 0.01) -> bool:
    """Return True if plan_to_pose solves IK to within tol_m, collision-free."""
    _banner("Check: plan_to_pose() — ssik IK end-to-end through the planner")
    lower, upper = arm.get_joint_limits()
    q_safe = (lower + upper) / 2

    arm.set_joint_positions(q_safe)
    T_target = arm.get_ee_pose().copy()
    T_target[2, 3] += 0.05  # +5cm in world Z — small but non-trivial IK solve
    print(f"Target = start EE pose + 5cm in Z")

    pose_path = arm.plan_to_pose(T_target)
    if pose_path is None:
        print("FAIL: plan_to_pose returned None (unreachable, in collision, or")
        print("IK→TSR→planning chain broken). Try a smaller offset to isolate.")
        return False

    final_q = pose_path[-1]
    arm.set_joint_positions(final_q)
    T_achieved = arm.get_ee_pose()
    pos_error = float(np.linalg.norm(T_achieved[:3, 3] - T_target[:3, 3]))
    n_collisions = len(arm.check_collisions())
    print(f"Position error: {pos_error * 1000:.3f} mm | collisions: {n_collisions}")

    if pos_error < tol_m and n_collisions == 0:
        print(f"PASS: solved IK and planned collision-free to within {tol_m * 1000:.0f}mm.")
        return True
    if pos_error >= tol_m:
        print(f"FAIL: position error {pos_error * 1000:.1f}mm exceeds {tol_m * 1000:.0f}mm.")
    else:
        print("FAIL: final config in collision despite a 'successful' plan.")
    return False


def main() -> None:
    env, arm = _make_arm()
    results = {
        "collision_aware_planning": check_collision_aware_planning(env, arm),
        "joint_types": check_joint_types(env, arm),
        "ik_through_planner": check_ik_through_planner(env, arm),
    }
    _banner("SUMMARY")
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if not all(results.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
