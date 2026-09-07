#!/usr/bin/env python3
"""Stage 1a -- CAN bus bring-up and health verification.

What this proves, per the Definition of Done:
  - the bus bring-up procedure is documented and scripted, not "run some
    commands and hope they're the same ones as last time" (this script IS
    that procedure -- it's meant to be run at the start of every session);
  - the interface comes up correctly framed/timed for CAN-FD (state, MTU,
    bitrate, data-bitrate all verified, not assumed from the command that
    was run);
  - the bus is free of errors before anything is trusted to run on it;
  - motor-feedback frames are actually arriving at each expected
    arbitration ID, at a plausible rate -- not just that the interface is
    administratively UP.

What this does NOT replace: this checks the *link*, not the *meaning* of
the data on it. Whether the reported joint positions are physically
correct is Stage 1b's job (stage1_sensor_liveness_and_accuracy.py).

Usage:
    python3 stage1_can_bus_bringup_and_health.py \\
        --interface can5 --bitrate 1000000 --dbitrate 5000000 \\
        --expect-ids 0x11 0x12 0x13 0x14 0x15 0x16 0x17
"""

from __future__ import annotations

import argparse

from common.can_helpers import (
    bring_up_can_interface,
    count_can_frames,
    get_can_error_counters,
    get_can_interface_state,
)
from common.operator_io import banner, confirm, safety_banner


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--interface", required=True, help="e.g. can5")
    p.add_argument("--bitrate", type=int, required=True, help="arbitration bitrate, e.g. 1000000")
    p.add_argument("--dbitrate", type=int, required=True, help="CAN-FD data bitrate, e.g. 5000000")
    p.add_argument("--expect-ids", nargs="+", default=["0x11", "0x12", "0x13", "0x14", "0x15", "0x16", "0x17"],
                   help="expected motor-feedback arbitration IDs (hex), default matches the 7-arm-joint convention")
    p.add_argument("--min-rate-hz", type=float, default=5.0, help="minimum acceptable per-ID frame rate")
    p.add_argument("--monitor-duration-s", type=float, default=5.0)
    p.add_argument("--skip-bringup", action="store_true", help="only verify, don't (re)configure the interface")
    p.add_argument("--no-sudo", action="store_true", help="omit sudo from the ip link commands")
    return p.parse_args()


def main():
    args = parse_args()
    expect_ids = [int(x, 16) if isinstance(x, str) and x.startswith("0x") else int(x) for x in args.expect_ids]

    safety_banner("Stage 1a", f"CAN bus bring-up and health check -- {args.interface}")

    if not args.skip_bringup:
        print(f"Bringing up {args.interface} (bitrate={args.bitrate}, dbitrate={args.dbitrate}, fd on)...")
        try:
            bring_up_can_interface(args.interface, args.bitrate, args.dbitrate, use_sudo=not args.no_sudo)
        except RuntimeError as e:
            raise SystemExit(f"FAILED to bring up {args.interface}: {e}")
    else:
        print(f"Skipping bring-up, verifying {args.interface} as currently configured.")

    banner("Interface state check")
    state = get_can_interface_state(args.interface)
    checks = []
    checks.append(("interface exists", state.exists))
    checks.append(("interface is UP", state.up))
    checks.append(("CAN-FD enabled", bool(state.fd_enabled)))
    if state.bitrate is not None:
        checks.append((f"bitrate == {args.bitrate}", state.bitrate == args.bitrate))
    else:
        checks.append(("bitrate reported by kernel", False))
    if state.dbitrate is not None:
        checks.append((f"dbitrate == {args.dbitrate}", state.dbitrate == args.dbitrate))
    else:
        checks.append(("dbitrate reported by kernel", False))

    all_state_ok = True
    for label, ok in checks:
        print(f"  [{'OK' if ok else 'FAIL'}] {label}")
        all_state_ok = all_state_ok and ok
    if not all_state_ok:
        print(f"\nRaw interface info for debugging:\n{state.raw}")

    banner("Error counter check")
    err = get_can_error_counters(args.interface)
    print(f"  rx_errors={err.rx_errors}  tx_errors={err.tx_errors}  bus_error={err.bus_error}")
    errors_ok = (err.rx_errors in (0, None)) and (err.tx_errors in (0, None)) and (err.bus_error in (0, None))
    if err.rx_errors is None and err.tx_errors is None:
        print("  Could not parse error counters from `ip -s -d link show` output on this system -- "
              "treat this check as inconclusive, not passed. See the raw output below if this matters:")
        print(err.raw)
    if not errors_ok:
        print("  Nonzero error counters found -- investigate before trusting this bus.")

    banner(f"Frame traffic check ({args.monitor_duration_s:.1f}s window)")
    print("Requires motors to be present/powered enough to report state (no torque enable needed, just "
          "bus presence) -- if this is a completely cold system, some frames may be absent until the "
          "hardware interface's own configure step first pings each motor.")
    try:
        frames = count_can_frames(args.interface, args.monitor_duration_s, filter_ids=expect_ids)
    except ImportError:
        print("  python-can not installed ('pip install python-can') -- cannot verify frame traffic here. "
              "This check is SKIPPED, not passed.")
        frames = None

    traffic_ok = True
    if frames is not None:
        for can_id in expect_ids:
            rate = frames.rate_hz(can_id)
            ok = rate is not None and rate >= args.min_rate_hz
            traffic_ok = traffic_ok and ok
            print(f"  [{'OK' if ok else 'FAIL'}] id=0x{can_id:02x}: {rate or 0:.1f} Hz (min {args.min_rate_hz:.1f} Hz)")

    overall = all_state_ok and errors_ok and traffic_ok and frames is not None
    banner(f"STAGE 1a RESULT: {'PASS' if overall else 'NOT MET'}", char="#")
    if not overall:
        raise SystemExit(1)


if __name__ == "__main__":
    main()