#!/usr/bin/env python3
"""Stage 8 -- Extended validation and sign-off.

What this proves, per the Definition of Done:
  - a realistic full session, run back to back at real duration, completes
    without a safety-invariant violation, drift, or unexplained anomaly;
  - all Part 1 invariants still hold at the end of an extended run, not
    only at the start;
  - the whole telemetry record is archived and indexed for future
    comparison.

This script is a deliberate departure from Stages 2-7's "confirm before
every real action" pattern -- that pattern exists to build trust
incrementally, one small verified step at a time. Stage 8 exists to prove
the *opposite* property: that the system holds up under realistic,
continuous, low-friction operation. Prompting before every operation here
would defeat the point. Instead: one confirmation at the start, one
continuous telemetry recording for the whole session, automated checks
after every operation, and an immediate pause ONLY when something looks
wrong -- exactly mirroring how a real operating session would actually be
run and supervised.

Usage:
    python3 stage8_extended_soak.py --session-file session.json --cycles 5

`session.json` schema:
    {"operations": [
        {"type": "single", "arm": "left", "goal": [q1..q7]},
        {"type": "bimanual", "goal_left": [...], "goal_right": [...]},
        ...
    ]}

If no session file is given, a procedurally-generated sequence is used --
fine for a first pass, but a real sign-off session should use an actual
representative sequence of the system's intended operations.
"""

from __future__ import annotations

import argparse
import json
import time

import mujoco
import numpy as np

from common.checks import position_reached
from common.collision_check import check_pair_collision_free, default_openarm_factory
from common.hardware_sync import sync_shadow_from_hardware
from common.operator_io import banner, confirm, confirm_phrase, safety_banner, wait_for_enter
from common.telemetry import JointStateRecorder, TelemetrySession


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--session-file", default=None)
    p.add_argument("--cycles", type=int, default=3)
    p.add_argument("--goal-delta-rad", type=float, default=0.3)
    p.add_argument(
        "--position-tolerance-rad", type=float, default=0.1,
        help="Stage 4 bring-up found the real onboard motor control (P+D, no integral term) has a "
        "genuine steady-state error floor from imperfect gravity compensation -- 0.02 (this arg's "
        "original default) is tighter than what pure P control can actually achieve at some poses.",
    )
    p.add_argument("--velocity-threshold", type=float, default=0.02)
    p.add_argument(
        "--settle-time-s", type=float, default=1.0,
        help="extra wait after execute() returns, before this op's OWN endpoint/collision check -- "
        "execute() returning only means the goal's nominal duration elapsed, not that the arm(s) have "
        "actually converged. Separate from --pause-between-ops-s, which paces the gap AFTER an op's "
        "check has already run, before the next op starts.",
    )
    p.add_argument("--pause-between-ops-s", type=float, default=1.0)
    p.add_argument("--log-dir", default="bringup_logs")
    p.add_argument(
        "--reverse-direction", action="store_true",
        help="Flip the session's fixed draw direction (default: left arm joints move positive, "
        "right arm joints move negative -- chosen deliberately to keep goals moving away from "
        "obstacles in the arms' normal starting configuration, not at random). Use this to "
        "deliberately run a whole session in the opposite direction; it does not randomize "
        "direction per-op, since drifting either way unpredictably is exactly what the fixed "
        "convention exists to avoid.",
    )
    return p.parse_args()


def build_robot():
    from openarm.config import OpenarmConfig
    from openarm.robot import Openarm

    return Openarm(config=OpenarmConfig.default())


def joint_names_for(robot, arm: str) -> list[str]:
    return list(getattr(robot, arm).arm.config.joint_names)


def _signed_delta_for_joint(arm: str, joint_name: str, magnitude: float, reverse: bool = False) -> float:
    """Same directional convention as stage4_trajectory_execution.py's
    get_signed_delta: left arm joints move positive, right arm joints move
    negative (or the opposite, for a whole session, if `reverse` is set --
    this is a deliberate per-session choice, not randomized per-op: the
    fixed convention exists specifically to keep goals moving away from
    obstacles near the arms' normal starting configuration, so flipping it
    per-draw would defeat the point), except joint6 (always opposite the
    arm's default sign) and joint4 (always positive -- its real resting
    position sits right at its own 0.0 lower limit, so a negative draw
    there is dead-on-arrival)."""
    sign = 1.0 if arm == "left" else -1.0
    if reverse:
        sign = -sign
    if "joint6" in joint_name:
        sign = -sign
    if "joint4" in joint_name:
        sign = 1.0
    return sign * abs(magnitude)


def make_goal(
    robot, arm: str, rng: np.random.Generator, delta: float, max_attempts: int = 20, reverse: bool = False
) -> np.ndarray:
    """Draw a random per-joint goal for `arm`, resampling away from
    self-collision or collision with the OTHER arm's current pose -- see
    stage4_trajectory_execution.py's make_goal() for why independent
    per-joint draws need this."""
    arm_scope = getattr(robot, arm).arm
    current = arm_scope.get_joint_positions().copy()
    joint_names = list(arm_scope.config.joint_names)
    lower, upper = arm_scope.get_joint_limits()

    saved_qpos = robot.data.qpos.copy()
    goal = current.copy()
    try:
        for attempt in range(max_attempts):
            magnitudes = rng.uniform(0.0, delta, size=current.shape)
            offset = np.array(
                [_signed_delta_for_joint(arm, name, mag, reverse) for name, mag in zip(joint_names, magnitudes)]
            )
            candidate = np.clip(current + offset, lower, upper)
            for val, idx in zip(candidate, arm_scope.joint_qpos_indices):
                robot.data.qpos[idx] = val
            mujoco.mj_forward(robot.model, robot.data)
            goal = candidate
            if not robot.arms.check_collisions(arm, verbose=False):
                break
        else:
            print(f"  (make_goal: every draw self-collided for '{arm}' after {max_attempts} attempts -- using the last one)")
        # TODO(review): one summary line for the final candidate, not every
        # retry -- a 20-attempt resample was flooding the console.
        robot.arms.check_collisions(arm)
    finally:
        robot.data.qpos[:] = saved_qpos
        mujoco.mj_forward(robot.model, robot.data)
    return goal


def make_concurrent_goals(
    robot, rng: np.random.Generator, delta: float, max_attempts: int = 20, reverse: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Draw independent left/right goals for a bimanual op, verified
    jointly collision-free with each other (not just individually) -- see
    stage6_bimanual_independent.py's version of this."""
    left_scope = robot.left.arm
    right_scope = robot.right.arm
    saved_qpos = robot.data.qpos.copy()
    goal_l = goal_r = None
    try:
        for attempt in range(max_attempts):
            goal_l = make_goal(robot, "left", rng, delta, reverse=reverse)
            goal_r = make_goal(robot, "right", rng, delta, reverse=reverse)
            for val, idx in zip(goal_l, left_scope.joint_qpos_indices):
                robot.data.qpos[idx] = val
            for val, idx in zip(goal_r, right_scope.joint_qpos_indices):
                robot.data.qpos[idx] = val
            mujoco.mj_forward(robot.model, robot.data)
            if not robot.arms.check_collisions(verbose=False):
                robot.arms.check_collisions()  # one summary line, not every retry
                return goal_l, goal_r
        print(f"  (make_concurrent_goals: no jointly collision-free pair found after {max_attempts} attempts -- using the last one)")
        robot.arms.check_collisions()
        return goal_l, goal_r
    finally:
        robot.data.qpos[:] = saved_qpos
        mujoco.mj_forward(robot.model, robot.data)


def load_or_generate_session(robot, args, rng, sync_recorder: JointStateRecorder) -> list[dict]:
    if args.session_file:
        with open(args.session_file) as f:
            return json.load(f)["operations"]

    print(
        "\nNo --session-file given -- generating a procedural sequence "
        "(single-arm and bimanual operations alternating). For a real sign-off, "
        "replace this with an actual representative operation sequence."
    )
    # Sync from real hardware before generating -- get_joint_positions()
    # reads the shadow model, which is never otherwise synced (see
    # common/hardware_sync.py); without this, goals would be computed
    # relative to wherever the shadow happened to default to, not the
    # arms' real current position.
    sync_shadow_from_hardware(robot, "left", sync_recorder)
    sync_shadow_from_hardware(robot, "right", sync_recorder)

    ops = []
    for i in range(6):
        if i % 3 == 2:
            goal_l, goal_r = make_concurrent_goals(robot, rng, args.goal_delta_rad, reverse=args.reverse_direction)
            ops.append({"type": "bimanual", "goal_left": goal_l.tolist(), "goal_right": goal_r.tolist()})
        else:
            arm = "left" if i % 2 == 0 else "right"
            goal = make_goal(robot, arm, rng, args.goal_delta_rad, reverse=args.reverse_direction)
            ops.append({"type": "single", "arm": arm, "goal": goal.tolist()})
    return ops


def run_single(robot, op: dict, args, recorder: JointStateRecorder) -> tuple[bool, str]:
    arm = op["arm"]
    other_arm = "right" if arm == "left" else "left"
    goal = np.asarray(op["goal"])
    # robot.reset() alone does NOT sync from real hardware -- see
    # common/hardware_sync.py. Reuses the session-long `recorder` (already
    # subscribed to both arms' joints) rather than a separate one.
    sync_shadow_from_hardware(robot, arm, recorder)
    sync_shadow_from_hardware(robot, other_arm, recorder)
    robot.clear_abort()
    result = robot.plan_to_configuration({arm: goal}, seed=0)
    if result is None or not result.success:
        return False, "planning failed"
    with robot.real() as ctx:  # noqa: F841
        ok = robot.execute(result)
        # execute() returning only means the goal's nominal duration
        # elapsed, not that the arm has actually converged -- see
        # stage4_trajectory_execution.py's identical wait.
        if ok and args.settle_time_s > 0:
            time.sleep(args.settle_time_s)
    if not ok:
        return False, "execute() returned False"
    joints = joint_names_for(robot, arm)
    endpoint_ok = all(
        position_reached(recorder.samples(), n, float(t), args.position_tolerance_rad).ok
        for n, t in zip(joints, goal)
    )
    return endpoint_ok, "endpoint reached" if endpoint_ok else "endpoint NOT reached"


def run_bimanual(robot, op: dict, args, recorder: JointStateRecorder) -> tuple[bool, str]:
    goal_l = np.asarray(op["goal_left"])
    goal_r = np.asarray(op["goal_right"])
    # robot.reset() alone does NOT sync from real hardware -- see
    # common/hardware_sync.py.
    sync_shadow_from_hardware(robot, "left", recorder)
    sync_shadow_from_hardware(robot, "right", recorder)
    robot.clear_abort()
    result = robot.plan_to_configuration({"left": goal_l, "right": goal_r}, seed=0)
    if result is None or not result.success:
        return False, "planning failed"

    t_start = time.time()
    with robot.real() as ctx:  # noqa: F841
        ok = robot.execute(result)
        # execute() returning only means the goal's nominal duration
        # elapsed, not that either arm has actually converged -- see
        # stage4_trajectory_execution.py's identical wait.
        if ok and args.settle_time_s > 0:
            time.sleep(args.settle_time_s)
    if not ok:
        return False, "execute() returned False"

    samples = [s for s in recorder.samples() if s.t >= t_start]
    left_joints = joint_names_for(robot, "left")
    right_joints = joint_names_for(robot, "right")
    coll = check_pair_collision_free(default_openarm_factory(), left_joints, samples, right_joints, samples)
    endpoint_l = all(position_reached(samples, n, float(t), args.position_tolerance_rad).ok for n, t in zip(left_joints, goal_l))
    endpoint_r = all(position_reached(samples, n, float(t), args.position_tolerance_rad).ok for n, t in zip(right_joints, goal_r))
    ok_all = coll.ok and endpoint_l and endpoint_r
    return ok_all, f"collision_free={coll.ok} ({coll.summary}), endpoints: left={endpoint_l} right={endpoint_r}"


def main():
    args = parse_args()
    safety_banner(
        "Stage 8",
        "Extended validation / soak test -- a realistic session run continuously, with "
        "automated monitoring rather than per-step confirmation.",
    )
    if not confirm(
        "Confirm Stage 7 has passed, and this session is intended to run for an extended, "
        "realistic duration without stopping between every operation.",
        default=False,
    ):
        raise SystemExit("Stopping: prerequisites not confirmed.")
    confirm_phrase(
        "This is the final gate before letting the system run continuously. Confirm the workspace "
        "is clear, e-stop is accessible throughout, and someone is supervising for the entire session."
    )

    import rclpy
    from rclpy.executors import MultiThreadedExecutor
    import threading

    if not rclpy.ok():
        rclpy.init()
    node = rclpy.create_node("stage8_telemetry_listener")
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    robot = build_robot()
    rng = np.random.default_rng(0)
    all_joints = joint_names_for(robot, "left") + joint_names_for(robot, "right")

    recorder = JointStateRecorder(node, joint_filter=all_joints)
    # Start now (not just when `with session:` begins below) so
    # load_or_generate_session()'s hardware sync has live data to read;
    # TelemetrySession.__enter__() calling start() again right after is
    # harmless (just clears the buffer and keeps recording).
    recorder.start()

    log = []
    anomalies = 0
    cycle = 0
    # TODO(review): everything that can raise (load_or_generate_session,
    # KeyboardInterrupt during a long soak run -- this stage is explicitly
    # meant to run "for an extended, realistic duration", so an operator
    # Ctrl+C is a real path, not just a hypothetical) is now inside
    # try/finally. Ported from the identical fix in Stage 6/4/5/7: any of
    # these used to skip executor.shutdown()/node.destroy_node() entirely,
    # leaving the daemon spin_thread (running the blocking executor.spin())
    # orphaned and still touching rclpy's C bindings while the interpreter
    # tore them down during shutdown -- CONFIRMED TO SEGFAULT ON REAL
    # HARDWARE in Stage 6's version of this exact pattern.
    try:
        operations = load_or_generate_session(robot, args, rng, recorder)

        session = TelemetrySession(args.log_dir, "stage8_extended_soak", "full_session", recorder, use_rosbag=True)
        with session:
            for cycle in range(1, args.cycles + 1):
                banner(f"Stage 8: cycle {cycle}/{args.cycles}", char="-")
                for i, op in enumerate(operations, 1):
                    print(f"  [{cycle}.{i}] {op['type']} operation...")
                    try:
                        if op["type"] == "single":
                            ok, msg = run_single(robot, op, args, recorder)
                        elif op["type"] == "bimanual":
                            ok, msg = run_bimanual(robot, op, args, recorder)
                        else:
                            ok, msg = False, f"unknown operation type '{op['type']}'"
                    except Exception as e:  # noqa: BLE001 -- any exception during a soak run is itself the finding
                        ok, msg = False, f"EXCEPTION: {type(e).__name__}: {e}"

                    log.append({"cycle": cycle, "op_index": i, "op": op, "ok": ok, "message": msg})
                    print(f"        -> {'ok' if ok else 'ANOMALY'}: {msg}")

                    if not ok:
                        anomalies += 1
                        banner(f"ANOMALY DETECTED at cycle {cycle}, operation {i}: {msg}", char="!")
                        if not confirm("An anomaly was detected. Continue the extended session anyway?", default=False):
                            print("Stopping the session early due to an anomaly.")
                            break
                    time.sleep(args.pause_between_ops_s)
                else:
                    continue
                break

            session.note("operations_log", log)
            session.note("anomaly_count", anomalies)
            session.note("cycles_completed", cycle)
    finally:
        executor.shutdown()
        spin_thread.join(timeout=2.0)
        node.destroy_node()

    banner("STAGE 8 FINAL SIGN-OFF CHECKLIST", char="#")
    print(f"Operations attempted: {len(log)}   Anomalies: {anomalies}")
    print(f"Telemetry archived at: {session.dir}\n")
    print(
        "The automated checks above cover endpoint accuracy and (for bimanual operations) "
        "post-hoc collision-freedom. The invariants below are procedural and need your own "
        "attestation before calling this stage -- and this bring-up -- done:\n"
    )
    invariant_prompts = [
        "No motion occurred at any point without an explicit, logged operator or session action behind it.",
        "Every command sent during this session was subject to the same validation as in earlier stages "
        "(no path was observed to bypass position/velocity/workspace checks).",
        "Stop mechanisms (software abort, physical e-stop) remained available and were not needed, "
        "or if needed, behaved as measured in Stage 5.",
        "Telemetry for this entire session is complete and has been archived.",
        "No naming/identity inconsistency was observed across any layer during this session.",
    ]
    all_attested = True
    for statement in invariant_prompts:
        all_attested = confirm(f"  Attest: {statement}", default=False) and all_attested

    ok_overall = (anomalies == 0) and all_attested
    banner(
        "STAGE 8 RESULT: " + ("PASS -- system meets this Definition of Done." if ok_overall else "NOT MET"),
        char="#",
    )
    raise SystemExit(0 if ok_overall else 1)


if __name__ == "__main__":
    main()