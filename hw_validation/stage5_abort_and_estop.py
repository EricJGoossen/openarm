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

import numpy as np

from common.checks import settle_time_after
from common.operator_io import banner, confirm, confirm_phrase, confirm_step, safety_banner, StepResult
from common.telemetry import JointStateRecorder, TelemetrySession
from common.trial_runner import TrialOutcome, run_repeated_trials

TRIGGER_POINTS = ("early", "middle", "late")
FRACTIONS = {"early": 0.15, "middle": 0.5, "late": 0.85}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", required=True, choices=["software", "physical"])
    p.add_argument("--arm", required=True, choices=["left", "right"])
    p.add_argument("--repeats-per-point", type=int, default=5, help="spec minimum: 5 per trigger point per arm")
    p.add_argument("--goal-delta-rad", type=float, default=1.0, help="a long-enough motion to observe mid-flight")
    p.add_argument("--velocity-settle-threshold", type=float, default=0.02)
    p.add_argument("--sim-baseline-latency-s", type=float, default=None,
                   help="optional Stage-0 simulated abort latency for this mechanism, for comparison")
    p.add_argument("--log-dir", default="bringup_logs")
    return p.parse_args()


def build_robot():
    from openarm.config import OpenarmConfig
    from openarm.robot import Openarm

    return Openarm(config=OpenarmConfig.default())


def long_goal(robot, arm: str, delta: float) -> np.ndarray:
    current = getattr(robot, arm).arm.get_joint_positions().copy()
    offset = np.zeros_like(current)
    offset[0] = delta  # a single large-ish joint move is enough to give a long, observable trajectory
    return current + offset


def confirm_recovery(robot) -> StepResult:
    robot.clear_abort()
    still_aborted = robot.is_abort_requested()
    print(f"\nis_abort_requested() after clear_abort(): {still_aborted} (expected False)")
    operator_recovered = confirm(
        "Per your documented recovery procedure: has the arm returned to normal, holdable, "
        "commandable operation (no stale fault, no unexpected residual motion)?",
        default=False,
    )
    return StepResult(
        "recovery after stop", (not still_aborted) and operator_recovered,
        f"abort flag cleared={not still_aborted}",
    )


def software_trial(robot, arm: str, joint_names: list[str], args, trigger_point: str, trial_index: int) -> TrialOutcome:
    label = f"software-{trigger_point}-{trial_index}"
    goal = long_goal(robot, arm, args.goal_delta_rad)

    confirm_phrase(
        f"About to execute a real trajectory on '{arm}' and issue a SOFTWARE abort at the "
        f"'{trigger_point}' point ({FRACTIONS[trigger_point]*100:.0f}% through). Confirm clear/ready."
    )

    robot.reset()
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
    recovery = confirm_step(confirm_recovery(robot))

    passed = step.passed and recovery.passed
    return TrialOutcome(passed, label, latency_msg)


def physical_trial(robot, arm: str, joint_names: list[str], args, trigger_point: str, trial_index: int) -> TrialOutcome:
    label = f"physical-{trigger_point}-{trial_index}"
    goal = long_goal(robot, arm, args.goal_delta_rad)

    confirm_phrase(
        f"About to execute a real trajectory on '{arm}'. This trial, please press the PHYSICAL "
        f"e-stop yourself at roughly the '{trigger_point}' point of the motion "
        f"({FRACTIONS[trigger_point]*100:.0f}% through). Confirm ready."
    )

    robot.reset()
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
    recovery = confirm_step(confirm_recovery(robot))

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

    overall_pass = True
    for trigger_point in TRIGGER_POINTS:
        def trial(i: int, tp=trigger_point) -> TrialOutcome:
            if args.mode == "software":
                return software_trial(robot, args.arm, joint_names, args, tp, i)
            return physical_trial(robot, args.arm, joint_names, args, tp, i)

        summary = run_repeated_trials(
            stage=f"Stage 5 ({args.mode}/{args.arm}/{trigger_point})",
            trial_fn=trial,
            required_consecutive=args.repeats_per_point,
        )
        overall_pass = overall_pass and summary.reached_target
        if not summary.reached_target:
            if not confirm(f"Trigger point '{trigger_point}' did not reach its target. Continue to the next trigger point anyway?", default=False):
                break

    executor.shutdown()
    _shared_node.destroy_node()
    raise SystemExit(0 if overall_pass else 1)


if __name__ == "__main__":
    main()