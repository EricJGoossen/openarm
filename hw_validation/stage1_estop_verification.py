#!/usr/bin/env python3
"""Stage 1d -- E-stop circuit verification via an observable software signal.

What this proves, per the Definition of Done: the e-stop is tested in both
directions (engage and release), and its effect matches the DOCUMENTED
design -- not assumed from a wiring diagram.

The honest limitation up front: this script cannot know, on its own, what
signal your e-stop is actually supposed to affect. That depends on your
wiring (does it cut power to the CAN transceivers? trip a GPIO? assert a
fault topic? something with zero software-visible effect at all?). Rather
than guess, this script makes you declare it explicitly via --signal-source,
and then mechanically checks the real behavior against that declaration --
the same "declare the expected behavior, then verify against it" pattern
used for the bimanual fault-response check in Stage 7. A script that
silently assumed a signal source would be worse than useless here; it would
create false confidence in exactly the system meant to protect people.

Supported --signal-source values:
    can-dropout     e-stop is expected to interrupt power to the CAN bus
                    /transceivers, so frame traffic should stop when
                    engaged and resume when released. This is the default,
                    and the most common wiring for a bus-level cutoff.
    status-topic    e-stop is expected to be reflected on a ROS topic
                    (e.g. a Bool on ROBOT_STATUS_TOPIC or similar). Pass
                    --status-topic and --status-type.
    none-visible    the e-stop is a purely electromechanical cutoff with
                    no software-visible signal at all (e.g. it just kills
                    motor driver power downstream of anything ROS can see).
                    In this case the ONLY thing this script can verify is
                    that CAN traffic (if the bus itself stays powered)
                    behaves as declared, or it will tell you plainly that
                    it cannot verify anything here and this bullet must be
                    closed by physical inspection instead.

Usage:
    python3 stage1_estop_verification.py --signal-source can-dropout \\
        --interface can5 --expect-during-engage no-traffic --repeats 3
"""

from __future__ import annotations

import argparse
import time

from common.can_helpers import count_can_frames
from common.operator_io import banner, confirm, confirm_phrase, safety_banner, wait_for_enter
from common.trial_runner import TrialOutcome, run_repeated_trials


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--signal-source", required=True, choices=["can-dropout", "status-topic", "none-visible"])
    p.add_argument("--interface", default=None, help="required for --signal-source can-dropout")
    p.add_argument("--expect-ids", nargs="+", default=None, help="arbitration IDs to monitor (hex), can-dropout mode")
    p.add_argument("--expect-during-engage", choices=["no-traffic", "traffic-continues"], default="no-traffic",
                   help="the DOCUMENTED expected effect while the e-stop is engaged -- declare this, don't assume it")
    p.add_argument("--status-topic", default=None, help="required for --signal-source status-topic")
    p.add_argument("--status-type", default="std_msgs/msg/Bool")
    p.add_argument("--expect-status-during-engage", default=None,
                   help="the DOCUMENTED expected value on --status-topic while engaged, e.g. 'False'")
    p.add_argument("--baseline-window-s", type=float, default=3.0)
    p.add_argument("--engaged-window-s", type=float, default=3.0)
    p.add_argument("--recovery-timeout-s", type=float, default=10.0)
    p.add_argument("--repeats", type=int, default=3)
    return p.parse_args()


def can_dropout_trial(args, trial_index: int) -> TrialOutcome:
    expect_ids = (
        [int(x, 16) for x in args.expect_ids] if args.expect_ids else [0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17]
    )
    label = f"can-dropout-{trial_index}"

    confirm_phrase("About to record a CAN traffic baseline, then ask you to engage the e-stop. Confirm ready.")
    print(f"Recording baseline traffic for {args.baseline_window_s:.1f}s...")
    baseline = count_can_frames(args.interface, args.baseline_window_s, filter_ids=expect_ids)
    baseline_present = baseline.total > 0
    print(f"  baseline total frames on watched IDs: {baseline.total} ({'OK' if baseline_present else 'FAIL -- no baseline traffic to compare against'})")
    if not baseline_present:
        return TrialOutcome(False, label, "no baseline CAN traffic -- can't test a dropout against nothing")

    wait_for_enter("Engage the e-stop now, then press Enter.")
    t_engage = time.time()
    print(f"Recording during-engage traffic for {args.engaged_window_s:.1f}s...")
    engaged = count_can_frames(args.interface, args.engaged_window_s, filter_ids=expect_ids)
    print(f"  during-engage total frames: {engaged.total}")

    if args.expect_during_engage == "no-traffic":
        matched = engaged.total == 0
    else:
        matched = engaged.total > 0
    print(f"  declared expectation ('{args.expect_during_engage}'): {'MATCHED' if matched else 'DID NOT MATCH'}")

    wait_for_enter("Release the e-stop now, then press Enter.")
    t_release = time.time()
    print(f"Watching for recovery (up to {args.recovery_timeout_s:.1f}s)...")
    deadline = time.time() + args.recovery_timeout_s
    recovered_at = None
    while time.time() < deadline:
        probe = count_can_frames(args.interface, 0.5, filter_ids=expect_ids)
        if probe.total > 0:
            recovered_at = time.time()
            break
    recovery_latency = (recovered_at - t_release) if recovered_at else None
    if recovery_latency is not None:
        print(f"  traffic resumed {recovery_latency:.2f}s after release")
    else:
        print(f"  traffic did NOT resume within {args.recovery_timeout_s:.1f}s of release")

    operator_ok = confirm(
        "Did the physical/electrical behavior you observed (whatever indicator your e-stop has -- "
        "lights, relay click, etc.) match what the CAN traffic data above suggests?",
        default=False,
    )

    passed = matched and (recovered_at is not None) and operator_ok
    notes = f"engage_matched={matched}, recovery_latency={recovery_latency}"
    return TrialOutcome(passed, label, notes)


def status_topic_trial(args, trial_index: int) -> TrialOutcome:
    from common.ros_helpers import RosSession
    import importlib

    label = f"status-topic-{trial_index}"
    module_name, _, class_name = args.status_type.replace("/msg/", ".msg.").rpartition(".")
    msg_cls = getattr(importlib.import_module(module_name), class_name)

    with RosSession("stage1_estop_status_probe") as session:
        node = session.node
        assert node is not None, "RosSession did not provide a node"
        latest = {}

        def cb(msg):
            latest["value"] = getattr(msg, "data", msg)

        node.create_subscription(msg_cls, args.status_topic, cb, 10)

        confirm_phrase("About to record a status-topic baseline, then ask you to engage the e-stop. Confirm ready.")
        time.sleep(args.baseline_window_s)
        baseline_value = latest.get("value")
        print(f"  baseline value on {args.status_topic}: {baseline_value}")

        wait_for_enter("Engage the e-stop now, then press Enter.")
        time.sleep(args.engaged_window_s)
        engaged_value = latest.get("value")
        print(f"  during-engage value: {engaged_value}")

        expected = args.expect_status_during_engage
        matched = (expected is not None) and (str(engaged_value) == expected)
        print(f"  declared expectation ('{expected}'): {'MATCHED' if matched else 'DID NOT MATCH'}")

        wait_for_enter("Release the e-stop now, then press Enter.")
        t_release = time.time()
        deadline = time.time() + args.recovery_timeout_s
        recovered_at = None
        while time.time() < deadline:
            if str(latest.get("value")) == str(baseline_value):
                recovered_at = time.time()
                break
            time.sleep(0.1)
        recovery_latency = (recovered_at - t_release) if recovered_at else None
        print(f"  recovery latency: {recovery_latency}")

    operator_ok = confirm("Did the physical behavior you observed match the topic data above?", default=False)
    passed = matched and (recovered_at is not None) and operator_ok
    return TrialOutcome(passed, label, f"engage_matched={matched}, recovery_latency={recovery_latency}")


def main():
    args = parse_args()
    safety_banner("Stage 1d", "E-stop verification via a declared, observable software signal.")

    if args.signal_source == "none-visible":
        banner("No software-visible signal declared for this e-stop", char="!")
        print(
            "You've declared this e-stop has no software-observable effect. This script cannot verify "
            "anything about it, and this bullet of Stage 1 must be closed by physical/electrical "
            "inspection instead (measure power at the motor driver terminals with the e-stop engaged, "
            "etc.) -- that's a legitimate way to close it, just not one this script can help with. "
            "Consider whether ANY software-visible signal could be added (even a simple GPIO-to-topic "
            "bridge) so future sessions don't rely purely on manual inspection every time."
        )
        raise SystemExit(1)

    if args.signal_source == "can-dropout" and not args.interface:
        raise SystemExit("--interface is required for --signal-source can-dropout")
    if args.signal_source == "status-topic" and not (args.status_topic and args.expect_status_during_engage):
        raise SystemExit("--status-topic and --expect-status-during-engage are required for --signal-source status-topic")

    trial_fn = can_dropout_trial if args.signal_source == "can-dropout" else status_topic_trial

    summary = run_repeated_trials(
        stage=f"Stage 1d ({args.signal_source})",
        trial_fn=lambda i: trial_fn(args, i),
        required_consecutive=args.repeats,
    )
    raise SystemExit(0 if summary.reached_target else 1)


if __name__ == "__main__":
    main()