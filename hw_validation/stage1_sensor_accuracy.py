#!/usr/bin/env python3
"""Stage 1b -- Encoder/position feedback liveness and accuracy, motors never enabled.

What this proves, per the Definition of Done:
  - joints move-by-hand -> reported values change plausibly (liveness);
  - moving one joint doesn't move another joint's *reported* value
    (no CAN-ID / joint-mapping crosstalk);
  - reported values match an independently-known physical reference at TWO
    separated points per joint, not one -- a single-point check can hide a
    scale error (encoder counts-per-radian misconfigured) that a two-point
    check catches by also verifying the reported DELTA between the two
    references matches the actual physical delta.

How this stays motor-safe: `joint_state_broadcaster` is a *controller* with
its own lifecycle, independent of the hardware component's. This script
spawns only that controller and leaves the hardware component itself
'inactive' -- the hardware interface's read() still populates joint states
(see openarm_hardware's on_configure(), which reads state without ever
calling enable_all()), but activation/enable_all()/return_to_zero() -- the
things that actually energize the motors -- never happen. If your hardware
interface's actual lifecycle behaves differently, this script will simply
fail to see live data change, which is itself useful information -- it
will NOT silently proceed to move anything.

Usage:
    python3 stage1_sensor_liveness_and_accuracy.py --arm left --joints \\
        openarm_left_joint1 ... openarm_left_joint7 \\
        [--reference-file references.json]

`references.json` (optional, needed for the accuracy phase):
    {"openarm_left_joint1": [{"description": "hard stop, negative direction",
                               "value_rad": 0.0},
                              {"description": "hard stop, positive direction",
                               "value_rad": 2.44}],
     ...}
"""

from __future__ import annotations

import argparse
import json
import time

from common.operator_io import banner, confirm, confirm_phrase, safety_banner, wait_for_enter
from common.ros_helpers import (
    RosSession,
    get_hardware_component_states,
    list_controllers,
    spawn_controller,
)
from common.telemetry import JointStateRecorder, TelemetrySession


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--component", required=True, help="hardware component name for this arm (must be 'inactive')")
    p.add_argument("--joints", nargs="+", required=True)
    p.add_argument("--reference-file", default=None)
    p.add_argument("--liveness-window-s", type=float, default=6.0)
    p.add_argument("--liveness-threshold-rad", type=float, default=0.03,
                   help="minimum change to count as 'moved', comfortably above encoder/read noise")
    p.add_argument("--crosstalk-tolerance-rad", type=float, default=0.01,
                   help="max change allowed on an untouched joint while another is being moved")
    p.add_argument("--accuracy-tolerance-rad", type=float, default=0.05)
    p.add_argument("--log-dir", default="bringup_logs")
    return p.parse_args()


def load_references(path: str | None) -> dict | None:
    if path is None:
        return None
    with open(path) as f:
        return json.load(f)


def ensure_safe_precondition(node, args) -> None:
    states = get_hardware_component_states(node)
    if args.component not in states:
        raise SystemExit(f"Component '{args.component}' not found. Available: {list(states.keys())}")
    if states[args.component] == "active":
        banner(f"'{args.component}' is ACTIVE", char="!")
        raise SystemExit(
            "Refusing to proceed: this stage is specifically for verifying sensors with the hardware "
            "component INACTIVE (motors never enabled). Deactivate it first (see Stage 2's script for "
            "the explicit activation/deactivation flow) and re-run this stage before Stage 2, not after."
        )
    print(f"Confirmed: '{args.component}' is '{states[args.component]}' (not active). Good.")

    controllers = list_controllers(node)
    broadcaster_active = any(
        name == "joint_state_broadcaster" and getattr(state, "label", state) == "active"
        for name, state in controllers.items()
    )
    if not broadcaster_active:
        print("joint_state_broadcaster is not active -- spawning it now (this does not touch the "
              "hardware component's own lifecycle, so motors remain unaffected)...")
        ok = spawn_controller("joint_state_broadcaster")
        if not ok:
            raise SystemExit("Failed to spawn joint_state_broadcaster -- cannot get live joint states safely.")
        time.sleep(1.0)


def phase_liveness_and_crosstalk(node, recorder, args) -> bool:
    banner("Phase 1: liveness and no-crosstalk (motors NOT enabled)")
    confirm_phrase(
        "CONFIRM: motors are NOT energized on this arm (it should be freely backdrivable by hand). "
        "You are about to move each joint by hand, one at a time."
    )

    all_ok = True
    for j in args.joints:
        wait_for_enter(f"Ready to move '{j}' by hand through a comfortable range. Press Enter, then move it now.")
        with TelemetrySession(args.log_dir, "stage1_sensor_liveness", f"liveness_{j}", recorder) as tel:
            time.sleep(args.liveness_window_s)
            samples = recorder.samples()
            tel.note("joint", j)

        if not samples:
            print(f"  [{j}] FAIL: no telemetry received at all")
            all_ok = False
            continue

        moved_values = {jn: [] for jn in args.joints}
        for s in samples:
            for jn in args.joints:
                if jn in s.positions:
                    moved_values[jn].append(s.positions[jn])

        target_range = (max(moved_values[j]) - min(moved_values[j])) if moved_values[j] else 0.0
        target_ok = target_range >= args.liveness_threshold_rad
        print(f"  [{'OK' if target_ok else 'FAIL'}] '{j}' reported range of motion = {target_range:.4f} rad "
              f"(min {args.liveness_threshold_rad:.4f})")

        crosstalk_ok = True
        for other in args.joints:
            if other == j or not moved_values[other]:
                continue
            other_range = max(moved_values[other]) - min(moved_values[other])
            ok = other_range <= args.crosstalk_tolerance_rad
            crosstalk_ok = crosstalk_ok and ok
            if not ok:
                print(f"    [FAIL] '{other}' also moved {other_range:.4f} rad while only '{j}' was touched "
                      f"-- possible CAN-ID / joint-mapping crosstalk")

        observed_ok = confirm(
            f"Automated verdict for '{j}': moved={target_ok}, no-crosstalk={crosstalk_ok}. Did this match "
            f"what you physically did (moved only '{j}', felt normal resistance, nothing else moved)?",
            default=False,
        )
        all_ok = all_ok and target_ok and crosstalk_ok and observed_ok

    return all_ok


def phase_accuracy(node, recorder, args, references: dict) -> bool:
    banner("Phase 2: two-point absolute accuracy spot-check")
    all_ok = True
    for j in args.joints:
        refs = references.get(j)
        if not refs or len(refs) < 2:
            print(f"  '{j}': SKIPPED -- need at least 2 reference points in the reference file, got {len(refs or [])}")
            continue

        reported = []
        for ref in refs[:2]:
            wait_for_enter(
                f"Move '{j}' by hand to: {ref['description']} (documented value {ref['value_rad']:.4f} rad). "
                f"Press Enter once positioned and held steady."
            )
            sample = recorder.wait_for_joints([j], timeout_s=5.0)
            if sample is None:
                print(f"  '{j}': FAIL -- no telemetry available at reference point '{ref['description']}'")
                all_ok = False
                reported.append(None)
                continue
            reported.append(sample[j])
            print(f"    reported = {sample[j]:.4f} rad")

        if None in reported:
            continue

        errors = [abs(r - ref["value_rad"]) for r, ref in zip(reported, refs[:2])]
        abs_ok = all(e <= args.accuracy_tolerance_rad for e in errors)
        for e, ref in zip(errors, refs[:2]):
            print(f"    [{'OK' if e <= args.accuracy_tolerance_rad else 'FAIL'}] '{ref['description']}': "
                  f"|error| = {e:.4f} rad (tolerance {args.accuracy_tolerance_rad:.4f})")

        reported_delta = reported[1] - reported[0]
        reference_delta = refs[1]["value_rad"] - refs[0]["value_rad"]
        delta_error = abs(reported_delta - reference_delta)
        delta_ok = delta_error <= args.accuracy_tolerance_rad
        print(f"    [{'OK' if delta_ok else 'FAIL'}] delta check: reported Δ={reported_delta:.4f} rad vs "
              f"reference Δ={reference_delta:.4f} rad (this is what catches a scale/gain error that a "
              f"single-point check would miss)")

        all_ok = all_ok and abs_ok and delta_ok

    return all_ok


def main():
    args = parse_args()
    safety_banner("Stage 1b", "Encoder liveness, no-crosstalk, and two-point accuracy -- motors never enabled.")

    with RosSession("stage1_sensor_liveness") as session:
        node = session.node
        ensure_safe_precondition(node, args)

        recorder = JointStateRecorder(node, joint_filter=args.joints)
        recorder.start()

        liveness_ok = phase_liveness_and_crosstalk(node, recorder, args)

        accuracy_ok = True
        references = load_references(args.reference_file)
        if references is None:
            print("\nPhase 2 (accuracy) SKIPPED -- no --reference-file supplied. Liveness/crosstalk alone "
                  "is NOT sufficient to close Stage 1's encoder-accuracy requirement; supply a reference "
                  "file with at least two documented physical reference points per joint to complete it.")
        else:
            accuracy_ok = phase_accuracy(node, recorder, args, references)

    overall = liveness_ok and accuracy_ok and references is not None
    banner(f"STAGE 1b RESULT: {'PASS' if overall else 'NOT MET / INCOMPLETE'}", char="#")
    raise SystemExit(0 if overall else 1)


if __name__ == "__main__":
    main()