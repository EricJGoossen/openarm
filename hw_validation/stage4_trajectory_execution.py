#!/usr/bin/env python3
"""Stage 4 -- Single-arm trajectory execution via the full software stack.

What this proves, per the Definition of Done:
  - a trajectory produced by the planning layer, retimed/packaged exactly as
    for a real operation, executes on real hardware and reaches the goal
    within tolerance;
  - the executed motion's *timing*, not just its endpoint, matches the plan
    within tolerance;
  - this is repeatable across varied goals and repeated cycles without
    state leaking between operations;
  - every safety behavior already proven at the raw-interface level in
    Stage 2/3 still holds when commands originate from the full stack --
    i.e. the extra layer never bypasses a check a lower layer already had.

This script uses this codebase's actual `openarm.robot.Openarm` public API
(`plan_to_configuration`, `real()`, `execute()`) rather than talking to ROS
directly -- Stage 4 is specifically about validating that API surface, so it
has to go through it, not around it. If that API isn't wired correctly yet
for this robot, this script will fail loudly and immediately, which is
itself the correct Stage 4 result to report.

Usage:
    python3 stage4_trajectory_execution.py --arm left \\
        [--goals-file goals.json] [--repeats 10]

`goals.json` (optional): a JSON list of per-joint goal arrays for the target
arm. If omitted, goals are generated as small, varied offsets from the
arm's current position at the start of each trial.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from common.checks import position_reached
from common.operator_io import banner, confirm, confirm_phrase, confirm_step, safety_banner, StepResult
from common.telemetry import JointStateRecorder, TelemetrySession
from common.trial_runner import TrialOutcome, run_repeated_trials


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", required=True, choices=["left", "right"])
    p.add_argument("--goals-file", default=None)
    p.add_argument("--repeats", type=int, default=10, help="consecutive clean round trips required (spec minimum: 10)")
    p.add_argument("--position-tolerance-rad", type=float, default=0.01)
    p.add_argument("--timing-tolerance-fraction", type=float, default=0.2)
    p.add_argument("--goal-delta-rad", type=float, default=0.3, help="scale of auto-generated goal offsets")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-dir", default="bringup_logs")
    return p.parse_args()


def load_goals(path: str | None) -> list[list[float]] | None:
    if path is None:
        return None
    with open(path) as f:
        return json.load(f)


def build_robot():
    from openarm.config import OpenarmConfig
    from openarm.robot import Openarm

    return Openarm(config=OpenarmConfig.default())


def regression_check_rejects_absurd_goal(robot, arm: str) -> StepResult:
    """Re-run the out-of-bound rejection property (proven at the raw
    interface in Stage 3) through the *full stack* this time -- confirms the
    planning/execution layer doesn't provide a path around a check a lower
    layer already enforces."""
    current = getattr(robot, arm).arm.get_joint_positions().copy()
    absurd = current.copy()
    absurd[0] += 100.0
    try:
        result = robot.plan_to_configuration({arm: absurd}, seed=0)
    except Exception as e:  # noqa: BLE001 -- any clean, non-crashing rejection is acceptable here
        return StepResult(
            "full-stack rejects an absurd goal", True, f"raised {type(e).__name__} as expected: {e}"
        )
    ok = result is None or not getattr(result, "success", False)
    return StepResult(
        "full-stack rejects an absurd goal",
        ok,
        "planning correctly reported failure/None for an infeasible goal"
        if ok
        else "planning returned a success result for an absurd goal -- investigate before trusting this stack",
    )


def make_goal(robot, arm: str, rng: np.random.Generator, delta_scale: float) -> np.ndarray:
    current = getattr(robot, arm).arm.get_joint_positions().copy()
    offset = rng.uniform(-delta_scale, delta_scale, size=current.shape)
    return current + offset


def run_one_round_trip(robot, arm: str, goal: np.ndarray, args, trial_index: int) -> TrialOutcome:
    label = f"{arm}-goal-{trial_index}"
    joint_names = list(getattr(robot, arm).arm.config.joint_names)

    confirm_phrase(
        f"About to plan and execute a real trajectory on the '{arm}' arm to goal={list(np.round(goal, 3))}. "
        f"Confirm workspace clear and e-stop hand-ready.",
    )

    robot.reset()
    robot.clear_abort()

    t_plan0 = time.time()
    result = robot.plan_to_configuration({arm: goal}, seed=trial_index)
    t_plan1 = time.time()
    if result is None or not getattr(result, "success", False):
        return TrialOutcome(False, label, f"planning failed (took {t_plan1 - t_plan0:.2f}s)")

    traj = getattr(result, arm)
    planned_duration = float(traj.timestamps[-1] - traj.timestamps[0]) if hasattr(traj, "timestamps") else None

    node_for_recorder = _shared_recorder_node
    recorder = JointStateRecorder(node_for_recorder, joint_filter=joint_names)
    with TelemetrySession(args.log_dir, "stage4_trajectory_execution", label, recorder) as tel:
        t_exec0 = time.time()
        with robot.real() as ctx:  # noqa: F841 -- context is entered for its side effect (connects hardware)
            ok = robot.execute(result)
        t_exec1 = time.time()
        samples = recorder.samples()
        tel.note("planned_duration_s", planned_duration)
        tel.note("executed_duration_s", t_exec1 - t_exec0)
        tel.note("goal", goal.tolist())

    if not ok:
        return TrialOutcome(False, label, "robot.execute() returned False")

    endpoint_ok = True
    endpoint_msgs = []
    for name, target in zip(joint_names, goal):
        res = position_reached(samples, name, float(target), args.position_tolerance_rad)
        endpoint_ok = endpoint_ok and res.ok
        endpoint_msgs.append(f"{name}: {res.message}")

    timing_ok = True
    timing_msg = "no planned-duration reference available -- skipped"
    if planned_duration is not None and planned_duration > 0:
        executed_duration = t_exec1 - t_exec0
        frac_err = abs(executed_duration - planned_duration) / planned_duration
        timing_ok = frac_err <= args.timing_tolerance_fraction
        timing_msg = (
            f"planned={planned_duration:.2f}s executed={executed_duration:.2f}s "
            f"(fractional error {frac_err:.2f}, tolerance {args.timing_tolerance_fraction:.2f})"
        )

    print("\nEndpoint check:\n  " + "\n  ".join(endpoint_msgs))
    print(f"\nTiming check: {timing_msg}")

    endpoint_step = confirm_step(
        StepResult("endpoint reached", endpoint_ok, "; ".join(endpoint_msgs)),
        question="Did the arm visibly reach the intended goal pose and stop there cleanly?",
    )
    timing_step = confirm_step(
        StepResult("execution timing matched plan", timing_ok, timing_msg),
        question="Did the motion's speed/duration look consistent with a smooth planned trajectory "
        "(not suspiciously instant, stalled, or jerky)?",
    )

    passed = endpoint_step.passed and timing_step.passed
    return TrialOutcome(passed, label, f"endpoint={endpoint_ok} timing={timing_ok}")


_shared_recorder_node = None  # set in main(); a single lightweight rclpy node reused across trials


def main():
    args = parse_args()
    banner_text = (
        "Stage 4: single-arm trajectory execution via the full software stack. "
        f"Target arm: {args.arm}."
    )
    safety_banner("Stage 4", banner_text)
    if not confirm(
        "Confirm reduced-authority limits are configured, telemetry/rosbag logging is available, "
        "and Stage 2 and Stage 3 have both already passed for this arm.",
        default=False,
    ):
        raise SystemExit("Stopping: prerequisites not confirmed.")

    import rclpy
    from rclpy.executors import MultiThreadedExecutor
    import threading

    global _shared_recorder_node
    if not rclpy.ok():
        rclpy.init()
    _shared_recorder_node = rclpy.create_node("stage4_telemetry_listener")
    executor = MultiThreadedExecutor()
    executor.add_node(_shared_recorder_node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    robot = build_robot()

    print("\nRunning a preliminary regression check: does the full stack still reject an absurd goal?")
    reg = confirm_step(regression_check_rejects_absurd_goal(robot, args.arm))
    if not reg.passed:
        raise SystemExit(
            "Stopping: the full-stack rejection regression check did not pass. Fix this before "
            "running real trajectories through this stack."
        )

    goals_raw = load_goals(args.goals_file)
    rng = np.random.default_rng(args.seed)

    def trial(i: int) -> TrialOutcome:
        if goals_raw is not None:
            goal = np.asarray(goals_raw[(i - 1) % len(goals_raw)], dtype=float)
        else:
            goal = make_goal(robot, args.arm, rng, args.goal_delta_rad)
        return run_one_round_trip(robot, args.arm, goal, args, i)

    summary = run_repeated_trials(
        stage=f"Stage 4 ({args.arm})",
        trial_fn=trial,
        required_consecutive=args.repeats,
    )

    executor.shutdown()
    _shared_recorder_node.destroy_node()

    raise SystemExit(0 if summary.reached_target else 1)


if __name__ == "__main__":
    main()