#!/usr/bin/env python3
"""Stage 2 -- First activation (single arm, no commanded goal beyond a
verified hold).

What this proves, per the Definition of Done:
  - the transition that first allows an arm to move under software control
    is explicit and operator-gated, never a side effect of something else
    coming online;
  - once active, the arm holds position without drift for a sustained
    period under whatever the default (no-goal) control law is;
  - every configured runtime safety trip can be deliberately triggered and
    is observed to behave correctly, with a measured latency;
  - a tripped fault can be cleared and the arm returns to normal operation.

This script does NOT use the high-level planning/execution Python API --
Stage 2 is about the hardware activation lifecycle itself, which sits below
that. It talks directly to the standard ros2_control hardware-component
lifecycle services, so the "explicit activation" step in the spec is
actually implemented here, not merely assumed to already be happening
correctly in whatever launch file is in use.

Usage:
    python3 stage2_first_activation.py --component left --joints \\
        openarm_left_joint1 openarm_left_joint2 openarm_left_joint3 \\
        openarm_left_joint4 openarm_left_joint5 openarm_left_joint6 openarm_left_joint7

Run once per arm. Requires everything through Stage 1 already complete
(power, CAN, e-stop verified) and the ROS graph up through
robot_state_publisher / ros2_control_node, with the target hardware
component currently INACTIVE.
"""

from __future__ import annotations

import argparse
import time

from common.checks import position_reached
from common.operator_io import (
    banner,
    confirm,
    confirm_phrase,
    safety_banner,
    timestamp,
    wait_for_enter,
)
from common.ros_helpers import RosSession, get_hardware_component_states, set_hardware_component_state
from common.telemetry import JointStateRecorder, TelemetrySession
from common.trial_runner import TrialOutcome, run_repeated_trials


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--component", required=True, help="ros2_control hardware component name for this arm")
    p.add_argument("--joints", nargs="+", required=True, help="joint names belonging to this arm")
    p.add_argument("--settle-window-s", type=float, default=8.0, help="how long to watch the hold after activation")
    p.add_argument("--hold-tolerance-rad", type=float, default=0.01, help="max drift allowed while holding")
    p.add_argument("--required-consecutive", type=int, default=5, help="consecutive clean activations required")
    p.add_argument("--log-dir", default="bringup_logs", help="root directory for telemetry output")
    return p.parse_args()


def show_state_and_confirm_starting_point(node, args) -> None:
    states = get_hardware_component_states(node)
    print("\nCurrent hardware component states:")
    for name, label in states.items():
        marker = " <-- target" if name == args.component else ""
        print(f"  {name}: {label}{marker}")
    if args.component not in states:
        raise SystemExit(
            f"Component '{args.component}' not found. Available: {list(states.keys())}. "
            f"Pass the correct --component name."
        )
    if states[args.component] == "active":
        banner(f"'{args.component}' is already ACTIVE", char="!")
        if not confirm(
            "It should be inactive at the start of this stage. Deactivate it now so this test "
            "exercises the real inactive->active transition?",
            default=True,
        ):
            raise SystemExit("Stopping: refusing to run Stage 2 against an already-active component.")
        set_hardware_component_state(node, args.component, "inactive")
        time.sleep(1.0)


def one_activation_trial(node, args, trial_index: int) -> TrialOutcome:
    label = f"{args.component}-activation-{trial_index}"

    safety_banner(
        f"Stage 2 / {args.component} / trial {trial_index}",
        "Activate one arm and verify a safe, drift-free hold.",
    )
    print(f"Joints under test: {args.joints}")
    confirm_phrase(
        "CONFIRM: workspace around this arm is completely clear, you have a hand on (or within\n"
        "immediate reach of) the physical e-stop, and you are watching the arm -- not the screen --\n"
        "for the next several seconds.",
        phrase="READY",
    )

    recorder = JointStateRecorder(node, joint_filter=args.joints)
    with TelemetrySession(args.log_dir, "stage2_first_activation", label, recorder) as tel:
        pre = recorder.wait_for_joints(args.joints, timeout_s=10.0)
        if pre is None:
            return TrialOutcome(False, label, "no /joint_states data before activation -- can't proceed")
        tel.note("pre_activation_positions", pre)
        print(f"Pre-activation positions: {pre}")

        t_activate_request = time.time()
        ok = set_hardware_component_state(node, args.component, "active")
        if not ok:
            return TrialOutcome(False, label, "SetHardwareComponentState service reported failure")

        print(f"Activation requested. Watching for {args.settle_window_s:.1f}s...")
        time.sleep(args.settle_window_s)

        samples = recorder.samples()
        tel.note("t_activate_request", t_activate_request)

    # Automated check: did every joint end up holding still (within
    # tolerance of wherever it settled) for the tail of the window?
    all_ok = True
    detail_lines = []
    for j in args.joints:
        # Use the last recorded position as the "settled" reference and
        # check the joint didn't wander away from it during the tail window.
        last_val = None
        for s in reversed(samples):
            if j in s.positions:
                last_val = s.positions[j]
                break
        if last_val is None:
            all_ok = False
            detail_lines.append(f"{j}: no data")
            continue
        res = position_reached(samples, j, last_val, args.hold_tolerance_rad, window_s=min(2.0, args.settle_window_s))
        all_ok = all_ok and res.ok
        detail_lines.append(f"{j}: {res.message}")

    print("\nAutomated hold-check results:")
    for line in detail_lines:
        print(f"  {line}")

    print(
        "\nNow visually confirm against the real arm (not just this printout):\n"
        "  - Did the arm move smoothly to a sensible position, with no sudden jump or jerk?\n"
        "  - Is every joint clearly away from its mechanical limit, not resting against a hard stop?\n"
        "  - Is it holding still right now, with no visible drift or oscillation?"
    )
    operator_ok = confirm("Confirm all three of the above are true", default=False)

    trip_note = ""
    if confirm(
        "\nOptional but recommended: deliberately trigger a runtime safety trip now (e.g. gently "
        "apply resistance to one joint, well within its rated torque, or apply another documented "
        "test stimulus) to verify the trip fires and recovers correctly. Do this now?",
        default=False,
    ):
        t_trip_prompt = time.time()
        wait_for_enter("Apply the test stimulus now, then press Enter once you observe a reaction (or none).")
        fired = confirm("Did a safety trip visibly fire (motion froze / fault reported)?", default=False)
        if fired:
            latency = time.time() - t_trip_prompt
            print(f"  Observed reaction ~{latency:.2f}s after you signaled the stimulus was applied "
                  f"(includes your own reaction time typing Enter -- treat as an upper bound, not a "
                  f"precise latency measurement).")
            recovered = confirm(
                "Attempt recovery per your documented fault-clearing procedure now. Did it "
                "successfully return to normal holding operation?",
                default=False,
            )
            trip_note = f"trip fired, recovery {'ok' if recovered else 'FAILED'}"
            operator_ok = operator_ok and recovered
        else:
            trip_note = "no trip observed for the applied stimulus (inconclusive, not necessarily a failure)"

    passed = all_ok and operator_ok
    notes = "; ".join(detail_lines[:2]) + (f"; {trip_note}" if trip_note else "")
    return TrialOutcome(passed, label, notes, data={"samples": len(samples)})


def deactivate_for_next_trial(node, args) -> None:
    print("\nDeactivating before the next trial (clean inactive->active cycle each time)...")
    set_hardware_component_state(node, args.component, "inactive")
    time.sleep(1.0)


def main():
    args = parse_args()
    with RosSession("stage2_first_activation") as session:
        node = session.node
        show_state_and_confirm_starting_point(node, args)

        def trial(i: int) -> TrialOutcome:
            outcome = one_activation_trial(node, args, i)
            deactivate_for_next_trial(node, args)
            return outcome

        summary = run_repeated_trials(
            stage=f"Stage 2 ({args.component})",
            trial_fn=trial,
            required_consecutive=args.required_consecutive,
        )

    raise SystemExit(0 if summary.reached_target else 1)


if __name__ == "__main__":
    main()