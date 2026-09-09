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

import mujoco
import numpy as np

from common.checks import position_reached
from common.hardware_sync import sync_shadow_from_hardware
from common.operator_io import banner, confirm, confirm_phrase, confirm_step, safety_banner, StepResult
from common.telemetry import JointStateRecorder, TelemetrySession
from common.trial_runner import TrialOutcome, run_repeated_trials

# HardwareContext._execute_trajectories (mj_manipulator_ros/hardware_context.py)
# always adds this fixed pre-roll delay before a multi-trajectory (bimanual)
# dispatch actually starts moving, via `getattr(self._config,
# "sync_start_buffer_sec", 0.15)` -- HardwareConfig has no such field, so this
# always falls through to the 0.15 default. Every Stage 4 call is bimanual
# (the untested arm always gets a "hold" trajectory alongside the tested
# one -- see PlanGroupResult.arm_results), so every execute() call pays this
# once. It's dispatch/sync overhead, not motion time, but the wall-clock
# executed_duration below can't tell the difference -- subtract it before
# judging whether the *motion itself* took about as long as planned.
DISPATCH_SYNC_BUFFER_S = 0.15


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", required=True, choices=["left", "right"])
    p.add_argument("--goals-file", default=None)
    p.add_argument("--repeats", type=int, default=10, help="consecutive clean round trips required (spec minimum: 10)")
    p.add_argument("--position-tolerance-rad", type=float, default=0.05)
    p.add_argument("--timing-tolerance-fraction", type=float, default=0.2)
    p.add_argument("--goal-delta-rad", type=float, default=0.3, help="scale of auto-generated goal offsets")
    p.add_argument(
        "--settle-time-s", type=float, default=1.0,
        help="extra wait after the trajectory's nominal duration before checking convergence -- "
        "the controller declares SUCCESS purely on elapsed time, not on actually having converged, "
        "and real tracking lags behind the reference (see stage4 investigation notes on stiction).",
    )
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


def _signed_delta_for_joint(arm: str, joint_name: str, magnitude: float) -> float:
    """Same directional convention as stage3_open_loop_motion.py's
    get_signed_delta: left arm joints move positive, right arm joints move
    negative, except joint6 (always opposite the arm's default sign) and
    joint4 (always positive regardless of side). `magnitude` is taken as
    absolute -- only its size is used, this function decides the sign.
    """
    if arm == "left":
        sign = 1.0
    elif arm == "right":
        sign = -1.0
    else:
        raise ValueError(f"unknown arm '{arm}'")

    if "joint6" in joint_name:
        sign = -sign
    if "joint4" in joint_name:
        sign = 1.0

    return sign * abs(magnitude)


def make_goal(robot, arm: str, rng: np.random.Generator, delta_scale: float, max_attempts: int = 20) -> np.ndarray:
    arm_scope = getattr(robot, arm).arm
    current = arm_scope.get_joint_positions().copy()
    joint_names = list(arm_scope.config.joint_names)
    lower, upper = arm_scope.get_joint_limits()

    # Draws are independent per joint, with no idea about arm geometry --
    # joint2's narrow range ([-3.3, 0.17]) in particular means a random draw
    # can land close enough to its limit, combined with joint3/4 flexion, to
    # self-collide the wrist/hand into the torso (confirmed via direct
    # contact check: openarm_body_link0 <-> left_link7/hand/right_finger).
    # With --seed fixed (default 0) and return_to_zero() putting every trial
    # back at ~the same starting pose, that isn't occasional bad luck -- the
    # same self-colliding draw recurs deterministically at the same trial
    # index every run. Resample (checked via the same collision checker the
    # planner itself relies on) instead of handing out a goal we already
    # know is dead-on-arrival.
    saved_qpos = robot.data.qpos.copy()
    goal = current.copy()
    try:
        for attempt in range(max_attempts):
            magnitudes = rng.uniform(0.0, delta_scale, size=current.shape)
            offset = np.array(
                [_signed_delta_for_joint(arm, name, mag) for name, mag in zip(joint_names, magnitudes)]
            )
            # A non-negative offset can still land out of range if `current`
            # itself is already sitting right at (or a hair past) a joint's
            # limit -- e.g. joint4's real resting position drifts slightly
            # negative past its own 0.0 lower bound.
            candidate = np.clip(current + offset, lower, upper)
            for val, idx in zip(candidate, arm_scope.joint_qpos_indices):
                robot.data.qpos[idx] = val
            mujoco.mj_forward(robot.model, robot.data)
            goal = candidate
            if not robot.arms.check_collisions(arm, verbose=False):
                break
        else:
            print(
                f"  (make_goal: every draw self-collided for '{arm}' after {max_attempts} attempts -- "
                f"using the last one; planning will likely reject it too)"
            )
        # TODO(review): one summary line for the final candidate, not every
        # retry -- see the same fix in stage6/stage7's make_goal().
        robot.arms.check_collisions(arm)
    finally:
        robot.data.qpos[:] = saved_qpos
        mujoco.mj_forward(robot.model, robot.data)
    return goal




def return_to_zero(robot, arm: str, args) -> None:
    """Plan and execute a return to the zero configuration.

    Purely an operator convenience between trials (consistent, known
    starting pose to watch from) -- not a checked test step, so no
    endpoint/timing confirmation prompts, just the usual real-motion
    confirm_phrase gate.
    """
    joint_names = list(getattr(robot, arm).arm.config.joint_names)
    zero = np.zeros(len(joint_names))
    confirm_phrase(
        f"About to return the '{arm}' arm to zero before the next trial. Confirm workspace clear.",
    )
    result = robot.plan_to_configuration({arm: zero}, seed=0)
    if result is None or not getattr(result, "success", False):
        print(f"  (couldn't plan a return-to-zero for '{arm}' -- skipping; next trial syncs from wherever it is)")
        return
    with robot.real() as ctx:  # noqa: F841
        robot.execute(result)


def run_one_round_trip(robot, arm: str, goal: np.ndarray, args, trial_index: int) -> TrialOutcome:
    label = f"{arm}-goal-{trial_index}"
    joint_names = list(getattr(robot, arm).arm.config.joint_names)

    confirm_phrase(
        f"About to plan and execute a real trajectory on the '{arm}' arm to goal={list(np.round(goal, 3))}. "
        f"Confirm workspace clear and e-stop hand-ready.",
    )

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
            # The controller declares SUCCESS purely on elapsed time reaching
            # the trajectory's nominal duration, not on actually having
            # converged -- real tracking lags behind the reference (see
            # stiction investigation notes). q_ref_ isn't reset when the goal
            # ends, so the arm keeps being driven toward the final point on
            # its own; wait here (still connected, still recording) to give
            # it a chance to actually get there before we check.
            if ok and args.settle_time_s > 0:
                time.sleep(args.settle_time_s)
        samples = recorder.samples()
        tel.note("planned_duration_s", planned_duration)
        tel.note("executed_duration_s", t_exec1 - t_exec0)
        tel.note("settle_time_s", args.settle_time_s)
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
        motion_duration = max(0.0, executed_duration - DISPATCH_SYNC_BUFFER_S)
        frac_err = abs(motion_duration - planned_duration) / planned_duration
        timing_ok = frac_err <= args.timing_tolerance_fraction
        timing_msg = (
            f"planned={planned_duration:.2f}s executed={executed_duration:.2f}s "
            f"(motion={motion_duration:.2f}s after subtracting {DISPATCH_SYNC_BUFFER_S:.2f}s dispatch "
            f"buffer; fractional error {frac_err:.2f}, tolerance {args.timing_tolerance_fraction:.2f})"
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

    # TODO(review): everything that can raise SystemExit (the regression
    # check failing, a declined confirm_phrase further down) is now inside
    # this try/finally -- it used to be able to skip executor.shutdown()/
    # _shared_recorder_node.destroy_node() entirely, leaving the daemon
    # spin_thread (running the blocking executor.spin()) orphaned and still
    # touching rclpy's C bindings while the interpreter tore them down
    # during shutdown. Ported from the same fix in Stage 6, where this
    # CONFIRMED TO SEGFAULT ON REAL HARDWARE (operator declined a
    # confirm_phrase mid-run).
    exit_code = 1
    try:
        print("\nRunning a preliminary regression check: does the full stack still reject an absurd goal?")
        reg = confirm_step(regression_check_rejects_absurd_goal(robot, args.arm))
        if not reg.passed:
            raise SystemExit(
                "Stopping: the full-stack rejection regression check did not pass. Fix this before "
                "running real trajectories through this stack."
            )

        robot.reset()
        robot.clear_abort()

        # Session-level listener used to sync the planning shadow to the real
        # arm's actual measured position before every trial -- see
        # sync_shadow_from_hardware(). Covers BOTH arms, not just the one under
        # test: plan_to_configuration() has the other (unnamed) arm "hold at
        # its current [shadow] config" every trial, and that shadow is never
        # otherwise refreshed -- left stale, the other arm gets a real
        # trajectory commanding it back toward wherever the shadow last
        # happened to be (originally the reset()-set "ready" pose), not
        # "stay where you actually are". Left running for the whole session.
        other_arm = "right" if args.arm == "left" else "left"
        all_joint_names = list(getattr(robot, args.arm).arm.config.joint_names) + list(
            getattr(robot, other_arm).arm.config.joint_names
        )
        sync_recorder = JointStateRecorder(_shared_recorder_node, joint_filter=all_joint_names)
        sync_recorder.start()

        confirm_phrase(
            f"About to read both arms' actual current joint state from /joint_states and sync the "
            f"internal planning model to it before the first trial (repeated before every trial from "
            f"here on -- the planning model is never otherwise read from hardware feedback). The "
            f"'{other_arm}' arm isn't being tested this run but will be commanded to hold its synced "
            f"position. Confirm workspace clear and telemetry is live.",
        )

        goals_raw = load_goals(args.goals_file)
        rng = np.random.default_rng(args.seed)

        def trial(i: int) -> TrialOutcome:
            sync_shadow_from_hardware(robot, args.arm, sync_recorder)
            sync_shadow_from_hardware(robot, other_arm, sync_recorder)
            if goals_raw is not None:
                goal = np.asarray(goals_raw[(i - 1) % len(goals_raw)], dtype=float)
            else:
                goal = make_goal(robot, args.arm, rng, args.goal_delta_rad)
            outcome = run_one_round_trip(robot, args.arm, goal, args, i)
            return_to_zero(robot, args.arm, args)
            return outcome

        summary = run_repeated_trials(
            stage=f"Stage 4 ({args.arm})",
            trial_fn=trial,
            required_consecutive=args.repeats,
        )
        exit_code = 0 if summary.reached_target else 1
    finally:
        executor.shutdown()
        spin_thread.join(timeout=2.0)
        _shared_recorder_node.destroy_node()

    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()