#!/usr/bin/env python3
"""Stage 7 -- Bimanual simultaneous, collision-safe trajectory execution.

The target capability. What this proves, per the Definition of Done:
  - a jointly-planned bimanual trajectory actually executes with both arms
    moving over genuinely overlapping time, not sequentially -- checked
    numerically from telemetry, not eyeballed;
  - the recorded, *executed* joint states of both arms are independently
    verified collision-free by a fresh collision checker at every sampled
    instant -- the planner's guarantee is re-proven against reality, not
    assumed to transfer;
  - a goal pair specifically constructed to collide under naive/independent
    execution is executed jointly and confirmed NOT to collide for real;
  - if one arm faults mid-trajectory, the other's response matches a
    behavior you declare in advance -- not whatever happens to occur.

Three phases, selectable via --phase (default: all, run in order):

    basic        a general jointly-planned bimanual round trip, checked for
                 simultaneity, endpoint accuracy, and collision-freedom.
    adversarial  searches for (or loads) a goal pair that would collide
                 under naive independent straight-line motion, executes it
                 jointly on real hardware, and confirms no collision.
    fault        aborts a joint execution mid-flight and checks the other
                 arm's response against a behavior you declare with
                 --expected-behavior.

A note on "confirmed to have actually been at risk of collision, not just
in simulation": there is no safe way to literally run the dangerous naive
version on real hardware -- doing so would defeat the point. What this
script does instead is find the adversarial pair using the *same* geometric
collision model the planner itself uses, built from the real robot's
as-built geometry -- the best available proxy for "physically" without
actually colliding the arms. That caveat is worth keeping in mind when
reading a pass here.

Usage:
    python3 stage7_bimanual_simultaneous.py --phase all --repeats 10 \\
        --expected-behavior both-stop
"""

from __future__ import annotations

import argparse
import json
import threading
import time

import numpy as np

from common.checks import first_motion_time, position_reached, settle_time_after, simultaneity_overlap_fraction
from common.collision_check import check_pair_collision_free, default_openarm_factory
from common.operator_io import banner, confirm, confirm_phrase, confirm_step, safety_banner, StepResult
from common.telemetry import JointStateRecorder, TelemetrySession
from common.trial_runner import TrialOutcome, run_repeated_trials


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--phase", choices=["basic", "adversarial", "fault", "all"], default="all")
    p.add_argument("--repeats", type=int, default=10, help="spec minimum for 'basic': 10")
    p.add_argument("--goal-delta-rad", type=float, default=0.3)
    p.add_argument("--position-tolerance-rad", type=float, default=0.01)
    p.add_argument("--simultaneity-start-tolerance-s", type=float, default=0.5)
    p.add_argument("--simultaneity-min-overlap-fraction", type=float, default=0.6)
    p.add_argument("--velocity-threshold", type=float, default=0.02)
    p.add_argument("--collision-check-dt", type=float, default=0.02)
    p.add_argument("--adversarial-goals-file", default=None, help="pre-found {\"left\": [...], \"right\": [...]}")
    p.add_argument("--adversarial-max-search-attempts", type=int, default=300)
    p.add_argument("--abort-is-global", action="store_true", default=True,
                   help="whether this system's abort mechanism affects both arms at once (see Stage 6's observation)")
    p.add_argument("--expected-behavior", choices=["both-stop", "left-stops-right-continues", "right-stops-left-continues"],
                   default=None, help="required for --phase fault/all: the documented, declared design behavior")
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


# ---------------------------------------------------------------------------
# Shared: run one joint bimanual plan+execute, record, and analyze.
# ---------------------------------------------------------------------------


def run_joint_execution(robot, goal_l, goal_r, args, label: str, seed: int):
    """Returns (ok, samples, result) for one plan+execute round trip, or
    (False, [], None) if planning itself failed."""
    robot.reset()
    robot.clear_abort()
    result = robot.plan_to_configuration({"left": goal_l, "right": goal_r}, seed=seed)
    if result is None or not getattr(result, "success", False):
        return False, [], None

    left_joints = joint_names_for(robot, "left")
    right_joints = joint_names_for(robot, "right")
    recorder = JointStateRecorder(_shared_node, joint_filter=left_joints + right_joints)
    with TelemetrySession(args.log_dir, "stage7_bimanual_simultaneous", label, recorder) as tel:
        with robot.real() as ctx:  # noqa: F841
            ok = robot.execute(result)
        samples = recorder.samples()
        tel.note("goal_left", np.asarray(goal_l).tolist())
        tel.note("goal_right", np.asarray(goal_r).tolist())
    return ok, samples, result


def analyze_simultaneity(samples, left_joints, right_joints, args) -> StepResult:
    t_left = first_motion_time(samples, left_joints, args.velocity_threshold)
    t_right = first_motion_time(samples, right_joints, args.velocity_threshold)
    if t_left is None or t_right is None:
        return StepResult("simultaneity", False, "one or both arms were never observed moving")
    start_gap = abs(t_left - t_right)
    overlap = simultaneity_overlap_fraction(
        samples, left_joints, samples, right_joints, args.velocity_threshold
    )
    ok = (start_gap <= args.simultaneity_start_tolerance_s) and overlap.ok and (
        overlap.value >= args.simultaneity_min_overlap_fraction
    )
    msg = (
        f"start-time gap = {start_gap:.3f}s (tolerance {args.simultaneity_start_tolerance_s:.2f}s); "
        f"{overlap.message} (required >= {args.simultaneity_min_overlap_fraction:.2f})"
    )
    return StepResult("simultaneity (numeric definition)", ok, msg)


def analyze_collision_free(samples, left_joints, right_joints, args) -> StepResult:
    result = check_pair_collision_free(
        default_openarm_factory(),
        left_joints,
        samples,
        right_joints,
        samples,
        dt=args.collision_check_dt,
    )
    return StepResult("independent collision check on executed telemetry", result.ok, result.summary)


def analyze_endpoints(samples, left_joints, goal_l, right_joints, goal_r, args) -> StepResult:
    left_ok = all(position_reached(samples, n, float(t), args.position_tolerance_rad).ok for n, t in zip(left_joints, goal_l))
    right_ok = all(position_reached(samples, n, float(t), args.position_tolerance_rad).ok for n, t in zip(right_joints, goal_r))
    ok = left_ok and right_ok
    return StepResult("both arms reached their goals", ok, f"left={left_ok} right={right_ok}")


# ---------------------------------------------------------------------------
# Phase: basic
# ---------------------------------------------------------------------------


def basic_trial(robot, args, rng, trial_index: int) -> TrialOutcome:
    label = f"basic-{trial_index}"
    left_joints = joint_names_for(robot, "left")
    right_joints = joint_names_for(robot, "right")
    goal_l = make_goal(robot, "left", rng, args.goal_delta_rad)
    goal_r = make_goal(robot, "right", rng, args.goal_delta_rad)

    confirm_phrase(
        "About to plan and execute a REAL JOINT bimanual trajectory -- both arms moving at once. "
        "Confirm workspace clear around BOTH arms and e-stop hand-ready."
    )

    ok, samples, result = run_joint_execution(robot, goal_l, goal_r, args, label, seed=trial_index)
    if not ok or not samples:
        return TrialOutcome(False, label, "planning or execute() failed")

    steps = [
        confirm_step(
            analyze_simultaneity(samples, left_joints, right_joints, args),
            question="Did both arms visibly move at the same time, not one after the other?",
        ),
        confirm_step(
            analyze_collision_free(samples, left_joints, right_joints, args),
            question="Did the arms stay clearly clear of each other throughout (no near-miss)?",
        ),
        confirm_step(analyze_endpoints(samples, left_joints, goal_l, right_joints, goal_r, args)),
    ]
    passed = all(s.passed for s in steps)
    notes = "; ".join(f"{s.name}={'ok' if s.passed else 'FAIL'}" for s in steps)
    return TrialOutcome(passed, label, notes)


def run_basic_phase(robot, args, rng):
    return run_repeated_trials(
        stage="Stage 7 / basic",
        trial_fn=lambda i: basic_trial(robot, args, rng, i),
        required_consecutive=args.repeats,
    )


# ---------------------------------------------------------------------------
# Phase: adversarial
# ---------------------------------------------------------------------------


def search_adversarial_pair(rng, max_attempts: int, n_interp: int = 15):
    """Search, using a fresh offline ArmGroup, for a goal pair whose
    straight-line, independently-interpolated motion collides -- i.e. would
    be at risk under naive (non-jointly-planned) execution. See the module
    docstring's caveat about what "physically" means here."""
    ag = default_openarm_factory()()
    left, right = ag["left"], ag["right"]
    q_l0, q_r0 = left.get_joint_positions().copy(), right.get_joint_positions().copy()
    lo_l, hi_l = left.get_joint_limits()
    lo_r, hi_r = right.get_joint_limits()

    for attempt in range(1, max_attempts + 1):
        q_l1 = rng.uniform(lo_l, hi_l)
        q_r1 = rng.uniform(lo_r, hi_r)
        for alpha in np.linspace(0.0, 1.0, n_interp):
            q_l = q_l0 + alpha * (q_l1 - q_l0)
            q_r = q_r0 + alpha * (q_r1 - q_r0)
            left.set_joint_positions(q_l)
            right.set_joint_positions(q_r)
            if ag.check_collisions():
                return q_l1, q_r1, attempt, float(alpha)
    raise RuntimeError(f"no adversarial pair found after {max_attempts} attempts")


def run_adversarial_phase(robot, args, rng):
    banner("Stage 7 / adversarial: finding or loading a goal pair that collides under naive execution")

    if args.adversarial_goals_file:
        with open(args.adversarial_goals_file) as f:
            data = json.load(f)
        goal_l, goal_r = np.asarray(data["left"]), np.asarray(data["right"])
        print(f"Loaded adversarial goal pair from {args.adversarial_goals_file}.")
    else:
        print(f"Searching for an adversarial pair (up to {args.adversarial_max_search_attempts} attempts)...")
        goal_l, goal_r, attempts, alpha = search_adversarial_pair(rng, args.adversarial_max_search_attempts)
        if goal_l is None:
            print(
                "\nNo adversarial pair found within the search budget. This could mean the arms' "
                "workspaces genuinely can't be made to collide in this model (contype/conaffinity or "
                "an <exclude> may be preventing cross-arm contact detection entirely) -- that itself "
                "is worth checking before concluding this phase can't be run."
            )
            return None
        print(f"Found an adversarial pair after {attempts} attempts (collision at interpolation alpha={alpha:.2f}).")
        save_path = f"{args.log_dir}/stage7_bimanual_simultaneous/adversarial_pair_found.json"
        import os

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump({"left": goal_l.tolist(), "right": goal_r.tolist()}, f, indent=1)
        print(f"Saved for reuse: {save_path}")

    left_joints = joint_names_for(robot, "left")
    right_joints = joint_names_for(robot, "right")

    def trial(i: int) -> TrialOutcome:
        label = f"adversarial-{i}"
        confirm_phrase(
            "About to execute the ADVERSARIAL bimanual goal pair jointly on real hardware. This pair "
            "was specifically chosen because moving independently would collide -- the joint planner "
            "is expected to avoid that. Confirm clear/ready and watch closely."
        )
        ok, samples, result = run_joint_execution(robot, goal_l, goal_r, args, label, seed=i)
        if not ok or not samples:
            return TrialOutcome(False, label, "planning or execute() failed")

        steps = [
            confirm_step(analyze_simultaneity(samples, left_joints, right_joints, args)),
            confirm_step(
                analyze_collision_free(samples, left_joints, right_joints, args),
                question="Did the arms clearly avoid each other despite this being an adversarial goal pair?",
            ),
            confirm_step(analyze_endpoints(samples, left_joints, goal_l, right_joints, goal_r, args)),
        ]
        passed = all(s.passed for s in steps)
        return TrialOutcome(passed, label, "; ".join(f"{s.name}={'ok' if s.passed else 'FAIL'}" for s in steps))

    return run_repeated_trials(stage="Stage 7 / adversarial", trial_fn=trial, required_consecutive=1, max_attempts=5)


# ---------------------------------------------------------------------------
# Phase: fault
# ---------------------------------------------------------------------------


def run_fault_phase(robot, args, rng):
    if args.expected_behavior is None:
        print(
            "\nStage 7 / fault SKIPPED -- pass --expected-behavior "
            "{both-stop,left-stops-right-continues,right-stops-left-continues} to declare the "
            "documented design behavior this phase should verify against."
        )
        return None

    left_joints = joint_names_for(robot, "left")
    right_joints = joint_names_for(robot, "right")

    def trial(i: int) -> TrialOutcome:
        label = f"fault-{i}"
        goal_l = make_goal(robot, "left", rng, args.goal_delta_rad)
        goal_r = make_goal(robot, "right", rng, args.goal_delta_rad)

        confirm_phrase(
            f"About to execute a joint bimanual trajectory and abort mid-flight. Declared expected "
            f"behavior: '{args.expected_behavior}'. Confirm clear/ready on both arms."
        )

        robot.reset()
        robot.clear_abort()
        result = robot.plan_to_configuration({"left": goal_l, "right": goal_r}, seed=i)
        if result is None or not result.success:
            return TrialOutcome(False, label, "planning failed")

        recorder = JointStateRecorder(_shared_node, joint_filter=left_joints + right_joints)
        exec_result = {}

        def run_exec():
            with robot.real() as ctx:  # noqa: F841
                exec_result["ok"] = robot.execute(result)

        with TelemetrySession(args.log_dir, "stage7_bimanual_simultaneous", label, recorder) as tel:
            thread = threading.Thread(target=run_exec, daemon=True)
            thread.start()
            time.sleep(1.0)  # crude fixed mid-flight point; good enough to exercise the fault path
            t_abort = time.time()
            robot.request_abort()
            thread.join(timeout=30.0)
            samples = recorder.samples()
            tel.note("t_abort", t_abort)
            tel.note("expected_behavior", args.expected_behavior)

        t_left_settled = settle_time_after(samples, left_joints, t_abort, args.velocity_threshold)
        t_right_settled = settle_time_after(samples, right_joints, t_abort, args.velocity_threshold)
        left_reached_goal = all(position_reached(samples, n, float(t), args.position_tolerance_rad).ok for n, t in zip(left_joints, goal_l))
        right_reached_goal = all(position_reached(samples, n, float(t), args.position_tolerance_rad).ok for n, t in zip(right_joints, goal_r))

        left_stopped = t_left_settled is not None
        right_stopped = t_right_settled is not None

        expectation_map = {
            "both-stop": left_stopped and right_stopped and not left_reached_goal and not right_reached_goal,
            "left-stops-right-continues": left_stopped and right_reached_goal,
            "right-stops-left-continues": right_stopped and left_reached_goal,
        }
        matched = expectation_map[args.expected_behavior]
        msg = (
            f"declared='{args.expected_behavior}'; observed: left_stopped={left_stopped} "
            f"left_reached_goal={left_reached_goal} right_stopped={right_stopped} "
            f"right_reached_goal={right_reached_goal}"
        )
        if not matched:
            print(f"\n  MISMATCH: observed behavior does not match the declared expectation.\n  {msg}")

        step = confirm_step(
            StepResult("observed behavior matches declared expectation", matched, msg),
            question="Did what you saw on the robot match the declared expected behavior?",
        )
        return TrialOutcome(step.passed, label, msg)

    return run_repeated_trials(stage="Stage 7 / fault", trial_fn=trial, required_consecutive=3)


_shared_node = None


def main():
    args = parse_args()
    safety_banner("Stage 7", "Bimanual simultaneous, collision-safe joint trajectory execution.")
    if not confirm(
        "Confirm Stage 6 has already passed, and everyone in the room understands that BOTH arms "
        "will move together, at once, in this stage.",
        default=False,
    ):
        raise SystemExit("Stopping: prerequisites not confirmed.")

    import rclpy
    from rclpy.executors import MultiThreadedExecutor
    import threading as _threading

    global _shared_node
    if not rclpy.ok():
        rclpy.init()
    _shared_node = rclpy.create_node("stage7_telemetry_listener")
    executor = MultiThreadedExecutor()
    executor.add_node(_shared_node)
    spin_thread = _threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    robot = build_robot()
    rng = np.random.default_rng(0)

    results = {}
    phases = ["basic", "adversarial", "fault"] if args.phase == "all" else [args.phase]
    for phase in phases:
        if phase == "basic":
            results["basic"] = run_basic_phase(robot, args, rng)
        elif phase == "adversarial":
            results["adversarial"] = run_adversarial_phase(robot, args, rng)
        elif phase == "fault":
            results["fault"] = run_fault_phase(robot, args, rng)

    executor.shutdown()
    _shared_node.destroy_node()

    banner("STAGE 7 OVERALL RESULT", char="#")
    all_ok = True
    for name, summary in results.items():
        ok = summary is not None and summary.reached_target
        all_ok = all_ok and ok
        print(f"  {name}: {'PASS' if ok else 'NOT MET / SKIPPED'}")

    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()