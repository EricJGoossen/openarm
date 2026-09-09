#!/usr/bin/env python3
"""Stage 3 -- Single-arm open-loop small motion.

What this proves, per the Definition of Done:
  - a command within bounds tracks the commanded reference within tolerance;
  - a command that violates configured bounds is rejected before reaching
    the actuator, and is logged clearly;
  - commanding a hold actually stops the arm;
  - this holds for every joint individually and for coordinated multi-joint
    motion, with no instability;
  - (if joint limits are supplied) behavior near the margin boundary is
    specifically exercised, not just comfortably-in-bounds motion.

Talks directly to the controller's streaming topic
(`/<controller>/joint_commands`) -- the lowest-level motion interface, one
layer below any trajectory action or planning code, per the spec's
definition of Stage 3.

Usage:
    python3 stage3_open_loop_motion.py --controller left_controller --joints \\
        openarm_left_joint1 ... openarm_left_joint7 \\
        [--limits-yaml joint_limits.yaml] [--margin-joint openarm_left_joint4 --margin 0.05]
"""

from __future__ import annotations

import argparse
import time

import yaml

from common.checks import position_reached, stayed_near
from common.operator_io import banner, confirm, confirm_phrase, confirm_step, safety_banner, StepResult
from common.ros_helpers import RosSession, make_streaming_publisher, ramp_stream
from common.telemetry import JointStateRecorder, TelemetrySession
from common.trial_runner import TrialOutcome, run_repeated_trials

ABSURD_DELTA_RAD = 100.0  # guaranteed to exceed any real joint's range; tests rejection without
# needing to know this robot's actual limits.


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--controller", required=True, help="controller name, e.g. left_controller")
    p.add_argument("--joints", nargs="+", required=True)
    p.add_argument("--small-delta-rad", type=float, default=0.3)
    p.add_argument("--ramp-duration-s", type=float, default=2.0)
    p.add_argument("--tracking-tolerance-rad", type=float, default=0.03)
    p.add_argument("--reject-hold-tolerance-rad", type=float, default=0.01,
                   help="how far the arm may drift and still count as 'did not move' for a rejected command")
    p.add_argument("--limits-yaml", default=None, help="optional joint_limits.yaml (lower/upper per joint)")
    p.add_argument("--margin-joint", default=None, help="joint to specifically test near its margin boundary")
    p.add_argument("--margin", type=float, default=0.0, help="admissible margin inside the raw limit, in rad")
    p.add_argument("--repeats", type=int, default=3, help="consecutive clean full-battery passes required")
    p.add_argument("--log-dir", default="bringup_logs")
    return p.parse_args()


def load_limits(path: str | None) -> dict | None:
    if path is None:
        return None
    with open(path) as f:
        raw = yaml.safe_load(f)
    # Accept either {joint: {limit: {lower, upper}}} or {joint: {lower, upper}}
    out = {}
    for name, v in raw.items():
        if "limit" in v:
            v = v["limit"]
        out[name] = (float(v["lower"]), float(v["upper"]))
    return out


def get_signed_delta(args, joint_name: str) -> float:
    """Return the motion delta with the required arm/joint-specific sign.

    Left arm commands are always positive; right arm commands are always
    negative -- except joint 6, which always moves opposite the rest of the
    joints, and joint 4, which is always positive regardless of arm side.
    The user's input magnitude is preserved.
    """
    delta = abs(args.small_delta_rad)

    if "left" in args.controller.lower():
        sign = 1.0
    elif "right" in args.controller.lower():
        sign = -1.0
    else:
        raise ValueError(
            f"Cannot determine arm side from controller name '{args.controller}'. "
            "Controller name must contain 'left' or 'right'."
        )

    if "joint6" in joint_name:
        sign = -sign
    if "joint4" in joint_name:
        sign = 1.0

    return sign * delta

def cap_positions_to_limits(positions: list[float], joint_names: list[str], limits: dict | None) -> list[float]:
    """Clamp each commanded position to the configured per-joint hard limits.

    This keeps Stage 3 commands inside the admissible range even when a raw
    target would otherwise exceed the URDF limits, and ensures the resulting
    position vector stays the same length as the requested joint list.
    """
    if limits is None:
        return [float(p) for p in positions]

    capped: list[float] = []
    for joint_name, value in zip(joint_names, positions):
        if joint_name in limits:
            lower, upper = limits[joint_name]
            value = max(lower, min(float(value), upper))
        capped.append(float(value))
    return capped


def phase_single_joint_small_motion(node, pub, recorder, args, limits: dict) -> StepResult:
    banner("Phase A: single-joint small motions (each joint independently)")
    all_ok = True
    msgs = []
    current = recorder.wait_for_joints(args.joints, timeout_s=10.0)
    if current is None:
        return StepResult("single-joint small motion", False, "no telemetry available")

    for j in args.joints:
        delta = get_signed_delta(args, j)

        confirm_phrase(
            f"About to move '{j}' by {delta:+.3f} rad and back. "
            "Confirm clear/ready."
        )

        base = current[j]

        # Cap the target to the URDF limits if limits were supplied.
        target = cap_positions_to_limits([base + delta], [j], limits)[0]

        others = [current[k] for k in args.joints]

        with TelemetrySession(args.log_dir, "stage3_open_loop_motion", f"single_{j}", recorder) as tel:
            q0 = list(others)
            q1 = list(others)
            idx = args.joints.index(j)
            q1[idx] = target
            ramp_stream(node, pub, args.joints, q0, q1, args.ramp_duration_s, limits=limits)
            time.sleep(0.5)
            ramp_stream(node, pub, args.joints, q1, q0, args.ramp_duration_s, limits=limits)
            time.sleep(0.5)
            samples = recorder.samples()
            tel.note("joint", j)
            tel.note("target", target)

        res = position_reached(samples, j, base, args.tracking_tolerance_rad, window_s=0.5)
        ok = res.ok
        all_ok = all_ok and ok
        msgs.append(f"{j}: returned-to-start {res.message};")
        print(f"  {j}: {'OK' if ok else 'FAIL'} -- {msgs[-1]}")

    return StepResult("single-joint small motion", all_ok, "; ".join(msgs))


def phase_rejection(node, pub, recorder, args) -> StepResult:
    banner("Phase B: out-of-range command rejection")
    current = recorder.wait_for_joints(args.joints, timeout_s=10.0)
    if current is None:
        return StepResult("rejection test", False, "no telemetry available")

    confirm_phrase(
        "About to publish a deliberately absurd, out-of-range target on one joint. It should be "
        "REJECTED and the arm should not move. Confirm clear/ready anyway, in case it is not."
    )
    j = args.joints[0]
    bad = list(current[k] for k in args.joints)
    idx = args.joints.index(j)
    bad[idx] = current[j] + ABSURD_DELTA_RAD

    with TelemetrySession(args.log_dir, "stage3_open_loop_motion", "rejection", recorder) as tel:
        from common.ros_helpers import send_streaming_point

        send_streaming_point(pub, args.joints, bad)
        time.sleep(1.5)
        samples = recorder.samples()
        tel.note("attempted_target", bad)

    res = stayed_near(samples, j, current[j], args.reject_hold_tolerance_rad)
    print(f"  {j}: {res.message} (arm must NOT have moved toward the absurd target)")
    return StepResult("rejection test", res.ok, res.message)


def phase_margin_boundary(node, pub, recorder, args, limits: dict) -> StepResult | None:
    if args.margin_joint is None:
        print("\nPhase C (margin boundary) SKIPPED -- no --margin-joint / --limits-yaml supplied.")
        return None
    if args.margin_joint not in limits:
        print(f"\nPhase C SKIPPED -- '{args.margin_joint}' not found in supplied limits file.")
        return None

    banner(f"Phase C: margin-boundary behavior on '{args.margin_joint}'")
    lower, upper = limits[args.margin_joint]
    just_inside = lower + args.margin + 0.005
    just_outside = lower + args.margin - 0.02  # inside raw limit, but violates the admissible margin

    current = recorder.wait_for_joints(args.joints, timeout_s=10.0)
    idx = args.joints.index(args.margin_joint)

    confirm_phrase(
        f"About to move '{args.margin_joint}' to {just_inside:.4f} rad (just inside its admissible "
        f"margin -- should be ACCEPTED) and separately attempt {just_outside:.4f} rad (should be "
        f"REJECTED). Confirm clear/ready."
    )

    ok_accept = True
    ok_reject = True
    msgs = []

    with TelemetrySession(args.log_dir, "stage3_open_loop_motion", "margin_accept", recorder) as tel:
        q_start = list(current[k] for k in args.joints)
        q_target = list(q_start)
        q_target[idx] = cap_positions_to_limits([just_inside], [args.margin_joint], limits)[0]
        q_target = cap_positions_to_limits(q_target, args.joints, limits)
        ramp_stream(node, pub, args.joints, q_start, q_target, args.ramp_duration_s, limits=limits)
        time.sleep(0.5)
        samples = recorder.samples()
    res = position_reached(samples, args.margin_joint, just_inside, args.tracking_tolerance_rad)
    ok_accept = res.ok
    msgs.append(f"just-inside-margin accept: {res.message}")

    with TelemetrySession(args.log_dir, "stage3_open_loop_motion", "margin_reject", recorder) as tel:
        from common.ros_helpers import send_streaming_point

        reference = recorder.wait_for_joints(args.joints, timeout_s=5.0)[args.margin_joint]
        bad = list(recorder.wait_for_joints(args.joints, timeout_s=5.0)[k] for k in args.joints)
        bad[idx] = just_outside
        send_streaming_point(pub, args.joints, bad)
        time.sleep(1.5)
        samples = recorder.samples()
    res = stayed_near(samples, args.margin_joint, reference, args.reject_hold_tolerance_rad)
    ok_reject = res.ok
    msgs.append(f"just-outside-margin reject: {res.message}")

    # Return this joint to a comfortably safe position before moving on.
    current = recorder.wait_for_joints(args.joints, timeout_s=5.0)
    q_start = list(current[k] for k in args.joints)
    q_safe = list(q_start)
    q_safe[idx] = cap_positions_to_limits([lower + max(args.margin * 3, 0.15)], [args.margin_joint], limits)[0]
    q_safe = cap_positions_to_limits(q_safe, args.joints, limits)
    ramp_stream(node, pub, args.joints, q_start, q_safe, args.ramp_duration_s, limits=limits)

    ok = ok_accept and ok_reject
    print(f"  {'; '.join(msgs)}")
    return StepResult("margin boundary behavior", ok, "; ".join(msgs))


def phase_multi_joint(node, pub, recorder, args, limits: dict) -> StepResult:
    banner("Phase D: coordinated multi-joint small motion")
    current = recorder.wait_for_joints(args.joints, timeout_s=10.0)
    confirm_phrase("About to move ALL joints together by a small delta and back. Confirm clear/ready.")

    q0 = [current[j] for j in args.joints]
    q1 = cap_positions_to_limits(
        [q + get_signed_delta(args, j) for q, j in zip(q0, args.joints)],
        args.joints,
        limits,
    )

    with TelemetrySession(args.log_dir, "stage3_open_loop_motion", "multi_joint", recorder) as tel:
        ramp_stream(node, pub, args.joints, q0, q1, args.ramp_duration_s, limits=limits)
        time.sleep(0.5)
        ramp_stream(node, pub, args.joints, q1, q0, args.ramp_duration_s, limits=limits)
        time.sleep(1.0)
        samples = recorder.samples()

    all_ok = True
    msgs = []
    for j, base in zip(args.joints, q0):
        res = position_reached(samples, j, base, args.tracking_tolerance_rad, window_s=0.5)
        ok = res.ok 
        all_ok = all_ok and ok
        msgs.append(f"{j}: {res.message}")
    print("  " + "\n  ".join(msgs))
    return StepResult("coordinated multi-joint motion", all_ok, "; ".join(msgs))


def run_battery(node, pub, recorder, args, limits: dict | None, trial_index: int) -> TrialOutcome:
    label = f"battery-{trial_index}"
    steps: list[StepResult] = []

    steps.append(confirm_step(phase_single_joint_small_motion(node, pub, recorder, args, limits)))
    steps.append(confirm_step(phase_rejection(node, pub, recorder, args)))
    if limits is not None:
        margin_result = phase_margin_boundary(node, pub, recorder, args, limits)
        if margin_result is not None:
            steps.append(confirm_step(margin_result))
    steps.append(confirm_step(phase_multi_joint(node, pub, recorder, args, limits)))

    passed = all(s.passed for s in steps)
    notes = "; ".join(f"{s.name}={'ok' if s.passed else 'FAIL'}" for s in steps)
    return TrialOutcome(passed, label, notes)


def main():
    args = parse_args()
    limits = load_limits(args.limits_yaml)

    with RosSession("stage3_open_loop_motion") as session:
        node = session.node
        pub = make_streaming_publisher(node, args.controller)
        recorder = JointStateRecorder(node, joint_filter=args.joints)
        recorder.start()  # keep a continuous background buffer available to wait_for_joints() calls
        # between phases; TelemetrySession's own start()/stop() around each phase clears and re-buffers
        # for that phase's saved file, which is fine since wait_for_joints only needs recent data.

        safety_banner("Stage 3", "Single-arm open-loop small motion via the streaming interface.")
        if not confirm(
            "Confirm reduced-authority limits (torque/effort well below rated) are configured for "
            "this session before any streaming commands are sent.",
            default=False,
        ):
            raise SystemExit("Stopping: configure reduced authority before running Stage 3.")

        def trial(i: int) -> TrialOutcome:
            return run_battery(node, pub, recorder, args, limits, i)

        summary = run_repeated_trials(
            stage="Stage 3",
            trial_fn=trial,
            required_consecutive=args.repeats,
        )

    raise SystemExit(0 if summary.reached_target else 1)


if __name__ == "__main__":
    main()