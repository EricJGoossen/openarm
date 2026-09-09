#!/usr/bin/env python3
"""Stage 5 -- Abort and fault recovery on real hardware.

What this proves, per the Definition of Done:
  - a software-initiated abort issued while a real trajectory is executing
    measurably halts commanded motion, with latency recorded (not assumed);
  - a physical e-stop engaged during real motion halts the arm, with
    latency recorded;
  - both paths are tested at multiple points during a trajectory (early,
    middle, late), not just one easy moment;
  - after a stop, the arm's final state matches the intended fail-safe
    behavior, and recovery via the documented procedure actually works.

Two modes, sharing all the same trial/telemetry/confirmation machinery:

    --mode software   the script itself calls robot.request_abort() at a
                       timed point during execution and measures the result.
    --mode physical    the script starts execution and asks a human to
                       press the physical e-stop at a moment of their
                       choosing; it passively records and measures.

This script deliberately does not assume a particular abort latency is
"acceptable" -- it measures and reports the number, and compares it against
a simulated baseline if one is supplied, flagging a large discrepancy rather
than pass/failing against an arbitrary constant. Whether a given latency is
good enough is a judgment call for whoever's running this, informed by the
number this script gives them.

Usage:
    python3 stage5_abort_and_estop.py --mode software --arm left --repeats-per-point 5
    python3 stage5_abort_and_estop.py --mode physical --arm left --repeats-per-point 5
"""

from __future__ import annotations

import argparse
import threading
import time

import mujoco
import numpy as np

from common.checks import position_reached, settle_time_after
from common.hardware_sync import sync_shadow_from_hardware
from common.operator_io import banner, confirm, confirm_phrase, confirm_step, safety_banner, StepResult
from common.telemetry import JointStateRecorder, TelemetrySession
from common.trial_runner import TrialOutcome, run_repeated_trials

TRIGGER_POINTS = ("early", "middle", "late")
FRACTIONS = {"early": 0.3, "middle": 0.5, "late": 0.85}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", required=True, choices=["software", "physical"])
    p.add_argument("--arm", required=True, choices=["left", "right"])
    p.add_argument("--repeats-per-point", type=int, default=5, help="spec minimum: 5 per trigger point per arm")
    p.add_argument("--goal-delta-rad", type=float, default=1.0, help="a long-enough motion to observe mid-flight")
    p.add_argument(
        "--velocity-settle-threshold", type=float, default=0.08,
        help="reported /joint_states velocity is quantized/noisy even at true zero motion -- inspected "
        "recorded telemetry from a real abort where position was bit-exact frozen for ~1s straight, "
        "and reported |velocity| still oscillated up to ~0.055 rad/s the whole time. 0.08 clears that "
        "noise floor with margin; 0.02 (the old default) is tighter than the noise itself, so "
        "settle_time_after() could never find a qualifying window regardless of real motion.",
    )
    p.add_argument(
        "--post-abort-settle-s", type=float, default=1.5,
        help="extra recording time after execute() returns (i.e. after the goal is cancelled/stopped) "
        "before checking settle -- execute() returning only means the goal ended, not that the arm has "
        "physically finished decelerating, and settle_time_after() needs sustain_s of post-stop data "
        "to confirm a settle at all.",
    )
    p.add_argument("--sim-baseline-latency-s", type=float, default=None,
                   help="optional Stage-0 simulated abort latency for this mechanism, for comparison")
    p.add_argument(
        "--recovery-check-delta-rad", type=float, default=0.2,
        help="size of the real test motion confirm_recovery() commands after clear_abort() to verify "
        "the arm is actually holdable/commandable again, instead of just asking the operator.",
    )
    p.add_argument("--recovery-check-tolerance-rad", type=float, default=0.03)
    p.add_argument("--log-dir", default="bringup_logs")
    return p.parse_args()


def build_robot():
    from openarm.config import OpenarmConfig
    from openarm.robot import Openarm

    return Openarm(config=OpenarmConfig.default())


def long_goal(robot, arm: str, delta: float) -> np.ndarray:
    arm_scope = getattr(robot, arm).arm
    current = arm_scope.get_joint_positions().copy()
    offset = np.zeros_like(current)
    offset[0] = delta  # a single large-ish joint move is enough to give a long, observable trajectory
    goal = current + offset
    # Defensive clamp (see stage4_trajectory_execution.py's make_goal()) --
    # with sync_shadow_from_hardware()/return_to_zero() below keeping
    # `current` accurate and near a known safe pose, this shouldn't bind in
    # practice, but costs nothing to guarantee.
    lower, upper = arm_scope.get_joint_limits()
    return np.clip(goal, lower, upper)


def return_to_zero(robot, arm: str, other_arm: str, sync_recorder: JointStateRecorder) -> None:
    """Plan and execute a return to the zero configuration between trials.

    Not just an operator convenience here (contrast
    stage4_trajectory_execution.py's version of this): long_goal() computes
    each trial's target as current + delta, and an abort can leave the real
    arm at any unpredictable point along the trajectory (early/middle/late
    triggers, plus whatever the operator's physical-mode press timing
    actually lands on). Without resetting to a known position between
    trials, repeated trials could walk the joint cumulatively toward its
    limit rather than each starting fresh. robot.reset() alone does NOT do
    this -- see common/hardware_sync.py.

    Re-syncs `other_arm` immediately before planning even though trial()
    already synced both arms at the top -- software_trial()/physical_trial()
    run for several real seconds (execute, wait-for-trigger, settle,
    operator prompts) in between, long enough for that initial sync to go
    stale and get the untested arm nudged by a "hold at stale shadow"
    trajectory (observed: right arm moving slightly during left-arm trials).
    """
    sync_shadow_from_hardware(robot, other_arm, sync_recorder)
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


def confirm_recovery(
    robot, arm: str, joint_names: list[str], args, other_arm: str, sync_recorder: JointStateRecorder
) -> StepResult:
    """Clear the abort, then actually command a small real motion and
    verify it via telemetry -- rather than just asking the operator whether
    the arm 'looks' recovered, prove it's genuinely holdable/commandable by
    driving it a known small distance and checking it got there.

    Re-syncs `other_arm` first -- see return_to_zero()'s docstring for why
    the sync at the top of trial() isn't enough by the time this runs.
    """
    sync_shadow_from_hardware(robot, other_arm, sync_recorder)
    robot.clear_abort()
    still_aborted = robot.is_abort_requested()
    print(f"\nis_abort_requested() after clear_abort(): {still_aborted} (expected False)")

    recorder = JointStateRecorder(_shared_node, joint_filter=joint_names)
    recorder.start()
    real = recorder.wait_for_joints(joint_names, timeout_s=5.0)
    motion_ok = False
    if real is None:
        motion_msg = "recovery-check motion not attempted -- couldn't read real joint state"
    else:
        arm_scope = getattr(robot, arm).arm
        for name, idx in zip(joint_names, arm_scope.joint_qpos_indices):
            robot.data.qpos[idx] = real[name]
        mujoco.mj_forward(robot.model, robot.data)

        lower, upper = arm_scope.get_joint_limits()
        goal = np.array([real[n] for n in joint_names])
        goal[0] = goal[0] + args.recovery_check_delta_rad
        # Every joint's target here defaults to wherever the arm actually
        # is right now -- including joints we're not deliberately moving.
        # joint4 in particular keeps drifting a hair past its own 0.0 lower
        # limit at rest (seen -0.0002 to -0.0105 across this session); left
        # unclamped, reproducing that as its own "target" gets this whole
        # plan rejected by _config_candidates depending on which side of
        # zero the noise happens to land on that trial.
        goal = np.clip(goal, lower, upper)

        result = robot.plan_to_configuration({arm: goal}, seed=0)
        if result is None or not getattr(result, "success", False):
            motion_msg = "recovery-check motion failed to plan -- treat recovery as NOT verified"
        else:
            with robot.real() as ctx:  # noqa: F841
                exec_ok = robot.execute(result)
            time.sleep(1.0)  # let it actually get there and settle before checking
            samples = recorder.samples()
            res = position_reached(samples, joint_names[0], float(goal[0]), args.recovery_check_tolerance_rad)
            motion_ok = exec_ok and res.ok
            motion_msg = f"recovery-check motion: execute()={exec_ok}, {res.message}"

    print(f"  {motion_msg}")
    operator_recovered = confirm(
        f"Automated check: {motion_msg}\nDid that recovery-check motion also look normal to you "
        "(moved smoothly, correct direction, no fault)?",
        default=False,
    )
    passed = (not still_aborted) and motion_ok and operator_recovered
    return StepResult(
        "recovery after stop", passed,
        f"abort flag cleared={not still_aborted}; {motion_msg}",
    )


def software_trial(
    robot, arm: str, joint_names: list[str], args, trigger_point: str, trial_index: int,
    other_arm: str, sync_recorder: JointStateRecorder,
) -> TrialOutcome:
    label = f"software-{trigger_point}-{trial_index}"
    goal = long_goal(robot, arm, args.goal_delta_rad)

    confirm_phrase(
        f"About to execute a real trajectory on '{arm}' and issue a SOFTWARE abort at the "
        f"'{trigger_point}' point ({FRACTIONS[trigger_point]*100:.0f}% through). Confirm clear/ready."
    )

    robot.clear_abort()
    result = robot.plan_to_configuration({arm: goal}, seed=trial_index)
    if result is None or not getattr(result, "success", False):
        return TrialOutcome(False, label, "planning failed")

    traj = getattr(result, arm)
    planned_duration = float(traj.timestamps[-1] - traj.timestamps[0]) if hasattr(traj, "timestamps") else 3.0

    recorder = JointStateRecorder(_shared_node, joint_filter=joint_names)
    exec_result = {}

    def run_exec():
        with robot.real() as ctx:  # noqa: F841
            exec_result["ok"] = robot.execute(result)

    with TelemetrySession(args.log_dir, "stage5_abort_and_estop", label, recorder) as tel:
        t_start = time.time()
        thread = threading.Thread(target=run_exec, daemon=True)
        thread.start()

        wait_s = planned_duration * FRACTIONS[trigger_point]
        time.sleep(max(0.05, wait_s))
        t_abort = time.time()
        robot.request_abort()
        print(f"  request_abort() issued at t+{t_abort - t_start:.2f}s (planned duration {planned_duration:.2f}s)")

        thread.join(timeout=planned_duration + 15.0)
        t_thread_done = time.time()
        # execute() returning just means the goal was cancelled -- it says
        # nothing about whether the arm has physically stopped decelerating
        # yet. settle_time_after() needs sustain_s (0.15s default) of
        # continuous low-velocity data *after* the settle point to confirm
        # it, so without waiting here the recording can end right as the
        # arm is still coasting to a stop, and a genuinely-fine abort gets
        # reported as "never settled" for lack of post-abort data, not
        # because anything was actually wrong.
        if args.post_abort_settle_s > 0:
            time.sleep(args.post_abort_settle_s)
        samples = recorder.samples()
        tel.note("t_abort_request", t_abort)
        tel.note("planned_duration_s", planned_duration)

    t_settled = settle_time_after(samples, joint_names, after_t=t_abort, velocity_threshold=args.velocity_settle_threshold)
    if t_settled is None:
        latency_msg = "motion never settled within the recorded window -- treat as a FAIL"
        latency_ok = False
    else:
        latency = t_settled - t_abort
        cmp_msg = ""
        if args.sim_baseline_latency_s is not None:
            ratio = latency / max(args.sim_baseline_latency_s, 1e-6)
            cmp_msg = f" (sim baseline was {args.sim_baseline_latency_s:.2f}s, ratio {ratio:.1f}x)"
            if ratio > 3:
                print(
                    "  WARNING: real-hardware abort latency is more than 3x the simulated baseline for "
                    "this mechanism -- the abort path likely isn't reaching hardware the way it does in "
                    "simulation. Investigate before relying on this."
                )
        latency_msg = f"measured settle latency = {latency:.3f}s after request_abort(){cmp_msg}"
        latency_ok = True  # no hardcoded pass/fail threshold; operator judgment below decides

    print(f"\n{latency_msg}")
    exec_returned_before_settle = exec_result.get("ok") is False
    print(f"robot.execute() returned: {exec_result.get('ok')} (expect False -- aborted, not completed)")

    step = confirm_step(
        StepResult(
            f"software abort at {trigger_point}",
            latency_ok and exec_result.get("ok") is False,
            latency_msg,
        ),
        question="Did the arm actually stop promptly and safely when the abort was issued -- not "
        "continue to the original goal, and not move violently?",
    )
    recovery = confirm_step(confirm_recovery(robot, arm, joint_names, args, other_arm, sync_recorder))

    passed = step.passed and recovery.passed
    return TrialOutcome(passed, label, latency_msg)


def physical_trial(
    robot, arm: str, joint_names: list[str], args, trigger_point: str, trial_index: int,
    other_arm: str, sync_recorder: JointStateRecorder,
) -> TrialOutcome:
    label = f"physical-{trigger_point}-{trial_index}"
    goal = long_goal(robot, arm, args.goal_delta_rad)

    confirm_phrase(
        f"About to execute a real trajectory on '{arm}'. This trial, please press the PHYSICAL "
        f"e-stop yourself at roughly the '{trigger_point}' point of the motion "
        f"({FRACTIONS[trigger_point]*100:.0f}% through). Confirm ready."
    )

    robot.clear_abort()
    result = robot.plan_to_configuration({arm: goal}, seed=trial_index)
    if result is None or not getattr(result, "success", False):
        return TrialOutcome(False, label, "planning failed")

    traj = getattr(result, arm)
    planned_duration = float(traj.timestamps[-1] - traj.timestamps[0]) if hasattr(traj, "timestamps") else 3.0
    print(f"  (planned duration ~{planned_duration:.2f}s -- aim your press accordingly)")

    recorder = JointStateRecorder(_shared_node, joint_filter=joint_names)
    exec_result = {}

    def run_exec():
        try:
            with robot.real() as ctx:  # noqa: F841
                exec_result["ok"] = robot.execute(result)
        except Exception as e:  # noqa: BLE001 -- an e-stop may surface as a hardware/comm exception; capture it
            exec_result["exception"] = str(e)

    with TelemetrySession(args.log_dir, "stage5_abort_and_estop", label, recorder) as tel:
        t_start = time.time()
        thread = threading.Thread(target=run_exec, daemon=True)
        thread.start()
        thread.join(timeout=planned_duration + 30.0)
        # See software_trial()'s identical wait -- without it, the recording
        # can end before there's enough post-stop data for
        # settle_time_after() to confirm a genuinely-fine stop.
        if args.post_abort_settle_s > 0:
            time.sleep(args.post_abort_settle_s)
        samples = recorder.samples()
        tel.note("planned_duration_s", planned_duration)

    approx_press_frac = input(
        "\nApproximately what fraction of the motion had elapsed when you pressed the e-stop? "
        "(0.0-1.0, best guess): "
    ).strip()
    try:
        approx_press_t = t_start + float(approx_press_frac) * planned_duration
    except ValueError:
        approx_press_t = t_start

    t_settled = settle_time_after(
        samples, joint_names, after_t=approx_press_t - 0.5, velocity_threshold=args.velocity_settle_threshold
    )
    if t_settled is None:
        latency_msg = "could not detect a clean settle after your reported press time -- treat as inconclusive/FAIL"
        latency_ok = False
    else:
        latency = t_settled - approx_press_t
        latency_msg = (
            f"approximate settle latency = {latency:.3f}s after your self-reported press time "
            f"(this is approximate, not a precise measurement -- wire a real e-stop status signal "
            f"into telemetry for a precise number)"
        )
        latency_ok = True

    print(f"\n{latency_msg}")

    step = confirm_step(
        StepResult(f"physical e-stop at {trigger_point}", latency_ok, latency_msg),
        question="Did the arm actually stop promptly and safely when you pressed the e-stop?",
    )

    print("\nRelease the e-stop per your documented recovery procedure now.")
    recovery = confirm_step(confirm_recovery(robot, arm, joint_names, args, other_arm, sync_recorder))

    passed = step.passed and recovery.passed
    return TrialOutcome(passed, label, latency_msg)


_shared_node = None


def main():
    args = parse_args()
    safety_banner(
        "Stage 5",
        f"Abort ({args.mode}) and fault recovery on real hardware -- arm: {args.arm}.",
    )
    if not confirm(
        "Confirm Stage 4 has already passed for this arm, reduced-authority limits are still in "
        "effect, and telemetry logging is available.",
        default=False,
    ):
        raise SystemExit("Stopping: prerequisites not confirmed.")

    if args.mode == "physical":
        print(
            "\nPHYSICAL MODE: you will need to press the real e-stop yourself during each trial. "
            "Make sure you can do so safely and comfortably before continuing."
        )
        if not confirm("Ready to proceed in physical mode?", default=False):
            raise SystemExit(0)

    import rclpy
    from rclpy.executors import MultiThreadedExecutor
    import threading as _threading

    global _shared_node
    if not rclpy.ok():
        rclpy.init()
    _shared_node = rclpy.create_node("stage5_telemetry_listener")
    executor = MultiThreadedExecutor()
    executor.add_node(_shared_node)
    spin_thread = _threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    robot = build_robot()
    joint_names = list(getattr(robot, args.arm).arm.config.joint_names)

    # Sync the planning shadow to real hardware feedback before every trial
    # (both arms -- the untested arm always gets a "hold" trajectory
    # alongside the tested one) and return to a known safe pose after every
    # trial -- see common/hardware_sync.py and return_to_zero() above for
    # why robot.reset() alone doesn't cover this here.
    other_arm = "right" if args.arm == "left" else "left"
    all_joint_names = joint_names + list(getattr(robot, other_arm).arm.config.joint_names)
    sync_recorder = JointStateRecorder(_shared_node, joint_filter=all_joint_names)
    sync_recorder.start()

    # TODO(review): the trigger-point loop is now inside try/finally --
    # ported from the identical fix in Stage 6/4. Anything raising
    # SystemExit in here (software_trial/physical_trial's confirm_phrase
    # calls, most likely) used to skip executor.shutdown()/
    # _shared_node.destroy_node() entirely, leaving the daemon spin_thread
    # (running the blocking executor.spin()) orphaned and still touching
    # rclpy's C bindings while the interpreter tore them down during
    # shutdown -- CONFIRMED TO SEGFAULT ON REAL HARDWARE in Stage 6's
    # version of this exact pattern.
    overall_pass = True
    try:
        for trigger_point in TRIGGER_POINTS:
            def trial(i: int, tp=trigger_point) -> TrialOutcome:
                sync_shadow_from_hardware(robot, args.arm, sync_recorder)
                sync_shadow_from_hardware(robot, other_arm, sync_recorder)
                if args.mode == "software":
                    outcome = software_trial(robot, args.arm, joint_names, args, tp, i, other_arm, sync_recorder)
                else:
                    outcome = physical_trial(robot, args.arm, joint_names, args, tp, i, other_arm, sync_recorder)
                return_to_zero(robot, args.arm, other_arm, sync_recorder)
                return outcome

            summary = run_repeated_trials(
                stage=f"Stage 5 ({args.mode}/{args.arm}/{trigger_point})",
                trial_fn=trial,
                required_consecutive=args.repeats_per_point,
            )
            overall_pass = overall_pass and summary.reached_target
            if not summary.reached_target:
                if not confirm(f"Trigger point '{trigger_point}' did not reach its target. Continue to the next trigger point anyway?", default=False):
                    break
    finally:
        executor.shutdown()
        spin_thread.join(timeout=2.0)
        _shared_node.destroy_node()

    raise SystemExit(0 if overall_pass else 1)


if __name__ == "__main__":
    main()