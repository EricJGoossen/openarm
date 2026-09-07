#!/usr/bin/env python3
"""Stage 6 -- Bimanual shared-stack verification (independent motion).

What this proves, per the Definition of Done:
  - each arm independently repeats Stage 4/5-equivalent behavior while the
    other arm's software/hardware presence is active (shared resources
    don't degrade single-arm behavior);
  - commanding one arm has no observable effect on the other;
  - a baseline check of how the abort mechanism's scope (per-arm vs
    whole-robot) actually behaves, recorded as an observation rather than
    assumed.

This stage deliberately does NOT use joint bimanual planning
(`plan_to_configuration({"left":..., "right":...})`) -- that is Stage 7's
job. Everything here is two independent single-arm plans, to isolate
"sharing a stack works" from "moving together on purpose works".

Usage:
    python3 stage6_bimanual_independent.py --repeats 5
"""

from __future__ import annotations

import argparse
import threading
import time

import numpy as np

from common.checks import position_reached, stayed_near
from common.operator_io import banner, confirm, confirm_phrase, confirm_step, safety_banner, StepResult
from common.telemetry import JointStateRecorder, TelemetrySession
from common.trial_runner import TrialOutcome, run_repeated_trials


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repeats", type=int, default=5, help="spec minimum: 5 consecutive trials")
    p.add_argument("--goal-delta-rad", type=float, default=0.3)
    p.add_argument("--position-tolerance-rad", type=float, default=0.01)
    p.add_argument("--idle-hold-tolerance-rad", type=float, default=0.01)
    p.add_argument("--log-dir", default="bringup_logs")
    return p.parse_args()


def build_robot():
    from openarm.config import OpenarmConfig
    from openarm.robot import Openarm

    return Openarm(config=OpenarmConfig.default())


def joint_names_for(robot, arm: str) -> list[str]:
    return list(getattr(robot, arm).arm.config.joint_names)


def make_goal(robot, arm: str, rng: np.random.Generator, delta: float) -> np.ndarray:
    current = getattr(robot, arm).arm.get_joint_positions().copy()
    return current + rng.uniform(-delta, delta, size=current.shape)


def phase_one_arm_other_idle(robot, active_arm: str, idle_arm: str, args, rng, trial_index: int) -> StepResult:
    active_joints = joint_names_for(robot, active_arm)
    idle_joints = joint_names_for(robot, idle_arm)
    idle_reference = getattr(robot, idle_arm).arm.get_joint_positions().copy()
    goal = make_goal(robot, active_arm, rng, args.goal_delta_rad)

    confirm_phrase(
        f"About to move ONLY the '{active_arm}' arm while '{idle_arm}' stays idle. Confirm clear/ready "
        f"on both arms (the idle one should not move at all)."
    )

    robot.reset()
    robot.clear_abort()
    result = robot.plan_to_configuration({active_arm: goal}, seed=trial_index)
    if result is None or not getattr(result, "success", False):
        return StepResult(f"{active_arm}-alone / {idle_arm}-idle", False, "planning failed")

    recorder = JointStateRecorder(_shared_node, joint_filter=active_joints + idle_joints)
    label = f"{active_arm}_alone_{idle_arm}_idle_{trial_index}"
    with TelemetrySession(args.log_dir, "stage6_bimanual_independent", label, recorder) as tel:
        with robot.real() as ctx:  # noqa: F841
            ok = robot.execute(result)
        samples = recorder.samples()
        tel.note("goal", goal.tolist())

    if not ok:
        return StepResult(f"{active_arm}-alone / {idle_arm}-idle", False, "execute() returned False")

    active_ok = all(
        position_reached(samples, n, float(t), args.position_tolerance_rad).ok
        for n, t in zip(active_joints, goal)
    )
    idle_ok = all(
        stayed_near(samples, n, float(r), args.idle_hold_tolerance_rad).ok
        for n, r in zip(idle_joints, idle_reference)
    )
    ok_all = active_ok and idle_ok
    msg = f"active-reached-goal={active_ok}, idle-stayed-still={idle_ok}"
    return StepResult(f"{active_arm}-alone / {idle_arm}-idle", ok_all, msg)


def phase_concurrent_independent(robot, args, rng, trial_index: int) -> StepResult:
    left_joints = joint_names_for(robot, "left")
    right_joints = joint_names_for(robot, "right")
    goal_l = make_goal(robot, "left", rng, args.goal_delta_rad)
    goal_r = make_goal(robot, "right", rng, args.goal_delta_rad)

    confirm_phrase(
        "About to command BOTH arms at once via two SEPARATE, uncoordinated single-arm calls "
        "(not a joint bimanual plan). This deliberately exercises shared-resource contention -- "
        "if something breaks here, that is a real finding, not a scripting error. Confirm clear/ready "
        "on both arms."
    )

    robot.reset()
    robot.clear_abort()
    result_l = robot.plan_to_configuration({"left": goal_l}, seed=trial_index)
    result_r = robot.plan_to_configuration({"right": goal_r}, seed=trial_index + 1000)
    if result_l is None or not result_l.success or result_r is None or not result_r.success:
        return StepResult("concurrent independent dispatch", False, "planning failed for one or both arms")

    outcome = {}

    def run_left():
        try:
            with robot.real() as ctx:  # noqa: F841
                outcome["left_ok"] = robot.execute(result_l)
        except Exception as e:  # noqa: BLE001
            outcome["left_exception"] = str(e)

    def run_right():
        try:
            with robot.real() as ctx:  # noqa: F841
                outcome["right_ok"] = robot.execute(result_r)
        except Exception as e:  # noqa: BLE001
            outcome["right_exception"] = str(e)

    recorder = JointStateRecorder(_shared_node, joint_filter=left_joints + right_joints)
    label = f"concurrent_independent_{trial_index}"
    with TelemetrySession(args.log_dir, "stage6_bimanual_independent", label, recorder) as tel:
        t_left = threading.Thread(target=run_left, daemon=True)
        t_right = threading.Thread(target=run_right, daemon=True)
        t_left.start()
        t_right.start()
        t_left.join(timeout=60.0)
        t_right.join(timeout=60.0)
        samples = recorder.samples()
        tel.note("goal_left", goal_l.tolist())
        tel.note("goal_right", goal_r.tolist())
        tel.note("outcome", {k: v for k, v in outcome.items() if not isinstance(v, bool) or True})

    no_exceptions = "left_exception" not in outcome and "right_exception" not in outcome
    both_returned_ok = outcome.get("left_ok") is True and outcome.get("right_ok") is True
    left_reached = all(
        position_reached(samples, n, float(t), args.position_tolerance_rad).ok
        for n, t in zip(left_joints, goal_l)
    )
    right_reached = all(
        position_reached(samples, n, float(t), args.position_tolerance_rad).ok
        for n, t in zip(right_joints, goal_r)
    )
    ok = no_exceptions and both_returned_ok and left_reached and right_reached
    msg = (
        f"no_exceptions={no_exceptions}, both_execute_ok={both_returned_ok}, "
        f"left_reached={left_reached}, right_reached={right_reached}, raw_outcome={outcome}"
    )
    return StepResult("concurrent independent dispatch", ok, msg)


def phase_abort_scope_observation(robot, args) -> StepResult:
    """Exploratory, not pass/fail in the usual sense: records whether this
    system's abort mechanism is global (affects both arms) or per-arm, since
    the spec's cross-arm fault-isolation requirement can only be phrased
    meaningfully once this is known."""
    banner("Observation: abort mechanism scope")
    robot.reset()
    robot.clear_abort()
    idle_arm_pos_before = robot.right.arm.get_joint_positions().copy()
    robot.request_abort()
    time.sleep(0.5)
    idle_arm_pos_after = robot.right.arm.get_joint_positions().copy()
    moved = float(np.max(np.abs(idle_arm_pos_after - idle_arm_pos_before)))
    robot.clear_abort()
    is_global = confirm(
        "Based on this codebase's abort implementation and what you just observed, is the abort "
        "mechanism whole-robot (affects all arms) rather than per-arm? Answer based on the actual "
        "behavior/code, not a guess.",
        default=True,
    )
    print(f"Recorded: abort scope = {'GLOBAL' if is_global else 'PER-ARM'} (idle-arm drift during test: {moved:.4f} rad)")
    return StepResult("abort scope observation", True, f"scope={'global' if is_global else 'per-arm'}")


def one_trial(robot, args, rng, trial_index: int) -> TrialOutcome:
    label = f"trial-{trial_index}"
    steps = [
        confirm_step(phase_one_arm_other_idle(robot, "left", "right", args, rng, trial_index)),
        confirm_step(phase_one_arm_other_idle(robot, "right", "left", args, rng, trial_index)),
        confirm_step(phase_concurrent_independent(robot, args, rng, trial_index)),
    ]
    passed = all(s.passed for s in steps)
    notes = "; ".join(f"{s.name}={'ok' if s.passed else 'FAIL'}" for s in steps)
    return TrialOutcome(passed, label, notes)


_shared_node = None


def main():
    args = parse_args()
    safety_banner("Stage 6", "Bimanual shared-stack verification -- independent (uncoordinated) motion.")
    if not confirm(
        "Confirm Stage 4 and Stage 5 have already passed independently for BOTH arms.",
        default=False,
    ):
        raise SystemExit("Stopping: prerequisites not confirmed.")

    import rclpy
    from rclpy.executors import MultiThreadedExecutor
    import threading as _threading

    global _shared_node
    if not rclpy.ok():
        rclpy.init()
    _shared_node = rclpy.create_node("stage6_telemetry_listener")
    executor = MultiThreadedExecutor()
    executor.add_node(_shared_node)
    spin_thread = _threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    robot = build_robot()
    rng = np.random.default_rng(0)

    confirm_step(phase_abort_scope_observation(robot, args))

    def trial(i: int) -> TrialOutcome:
        return one_trial(robot, args, rng, i)

    summary = run_repeated_trials(stage="Stage 6", trial_fn=trial, required_consecutive=args.repeats)

    executor.shutdown()
    _shared_node.destroy_node()
    raise SystemExit(0 if summary.reached_target else 1)


if __name__ == "__main__":
    main()