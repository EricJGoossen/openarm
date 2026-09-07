"""CAN interface helpers for Stage 1. Deliberately independent of ROS --
everything here works from the OS network stack (`ip link`) and raw
SocketCAN, since Stage 1 is specifically about verifying the bus *before*
any ROS-level control stack is trusted to sit on top of it.

Requires `python-can` for frame monitoring (`pip install python-can`) and a
Linux `ip` (iproute2) binary for interface state/bring-up. Neither requires
ROS.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from dataclasses import dataclass, field


@dataclass
class CanInterfaceState:
    exists: bool
    up: bool
    fd_enabled: bool | None
    bitrate: int | None
    dbitrate: int | None
    mtu: int | None
    raw: str = ""


def get_can_interface_state(iface: str) -> CanInterfaceState:
    """Read current state via `ip -details -json link show <iface>`, with a
    plain-text fallback if `-json` isn't supported by the installed
    iproute2 version."""
    try:
        out = subprocess.run(
            ["ip", "-details", "-json", "link", "show", iface],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            data = json.loads(out.stdout)[0]
            linkinfo = data.get("linkinfo", {}) or {}
            info_data = linkinfo.get("info_data", {}) or {}
            bittiming = info_data.get("bittiming", {}) or {}
            data_bittiming = info_data.get("data_bittiming", {}) or {}
            return CanInterfaceState(
                exists=True,
                up="UP" in data.get("flags", []),
                fd_enabled=bool(data_bittiming.get("bitrate")),
                bitrate=bittiming.get("bitrate"),
                dbitrate=data_bittiming.get("bitrate"),
                mtu=data.get("mtu"),
                raw=json.dumps(data),
            )
    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError, IndexError):
        pass

    # Fallback: plain-text `ip -details link show`.
    out = subprocess.run(["ip", "-details", "link", "show", iface], capture_output=True, text=True, timeout=5)
    if out.returncode != 0 or not out.stdout.strip():
        return CanInterfaceState(exists=False, up=False, fd_enabled=None, bitrate=None, dbitrate=None, mtu=None)
    text = out.stdout
    up = "UP" in text.split("\n")[0]
    mtu_match = re.search(r"mtu (\d+)", text)
    bitrate_match = re.search(r"bitrate (\d+)", text)
    dbitrate_match = re.search(r"dbitrate (\d+)", text)
    fd_match = "fd on" in text or "FD ON" in text.upper()
    return CanInterfaceState(
        exists=True,
        up=up,
        fd_enabled=fd_match,
        bitrate=int(bitrate_match.group(1)) if bitrate_match else None,
        dbitrate=int(dbitrate_match.group(1)) if dbitrate_match else None,
        mtu=int(mtu_match.group(1)) if mtu_match else None,
        raw=text,
    )


def bring_up_can_interface(
    iface: str, bitrate: int, dbitrate: int, restart_ms: int = 100, use_sudo: bool = True
) -> None:
    """Idempotent CAN-FD bring-up: down, then up with the given bitrate/
    dbitrate and fd on. Safe to call every session -- this IS the
    "documented, ideally scripted" bring-up procedure Stage 1 asks for.
    """
    prefix = ["sudo"] if use_sudo else []
    subprocess.run(prefix + ["ip", "link", "set", iface, "down"], capture_output=True)
    result = subprocess.run(
        prefix
        + [
            "ip", "link", "set", iface, "type", "can",
            "bitrate", str(bitrate),
            "dbitrate", str(dbitrate),
            "fd", "on",
            "restart-ms", str(restart_ms),
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"failed to configure {iface}: {result.stderr}")
    result = subprocess.run(prefix + ["ip", "link", "set", iface, "up"], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"failed to bring {iface} up: {result.stderr}")


@dataclass
class CanErrorCounters:
    rx_errors: int | None = None
    tx_errors: int | None = None
    bus_error: int | None = None  # bus-off / error-passive style counter, name varies by driver
    raw: str = ""


def get_can_error_counters(iface: str) -> CanErrorCounters:
    """Best-effort parse of `ip -s -d link show <iface>` for error counters.
    Field names/layout vary a bit across CAN drivers, so treat a None field
    as "couldn't find it" rather than "zero" -- don't silently pass on a
    field this couldn't actually read.
    """
    out = subprocess.run(["ip", "-s", "-d", "link", "show", iface], capture_output=True, text=True, timeout=5)
    text = out.stdout
    rx = re.search(r"RX:.*?\n\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)", text, re.S)
    tx = re.search(r"TX:.*?\n\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)", text, re.S)
    bus_err = re.search(r"bus-off\s+(\d+)|bus error\s+(\d+)", text, re.I)
    return CanErrorCounters(
        rx_errors=int(rx.group(3)) if rx else None,  # RX line format: bytes packets errors dropped ...
        tx_errors=int(tx.group(3)) if tx else None,
        bus_error=int(next((g for g in (bus_err.groups() if bus_err else []) if g), 0)) if bus_err else None,
        raw=text,
    )


@dataclass
class FrameCounts:
    counts_by_id: dict[int, int] = field(default_factory=dict)
    total: int = 0
    duration_s: float = 0.0

    def rate_hz(self, can_id: int) -> float | None:
        if self.duration_s <= 0:
            return None
        return self.counts_by_id.get(can_id, 0) / self.duration_s


def count_can_frames(iface: str, duration_s: float, filter_ids: list[int] | None = None) -> FrameCounts:
    """Passively listen on `iface` for `duration_s` seconds and tally frames
    by arbitration ID. Requires `python-can`. Used to confirm expected
    motor-feedback IDs are actually present and arriving at a plausible
    rate -- not just that the interface is administratively UP.
    """
    import can  # local import: keep this module importable even if python-can isn't installed

    counts: dict[int, int] = {}
    bus = can.interface.Bus(channel=iface, bustype="socketcan")
    t_end = time.time() + duration_s
    try:
        while time.time() < t_end:
            msg = bus.recv(timeout=max(0.0, t_end - time.time()))
            if msg is None:
                continue
            if filter_ids is not None and msg.arbitration_id not in filter_ids:
                continue
            counts[msg.arbitration_id] = counts.get(msg.arbitration_id, 0) + 1
    finally:
        bus.shutdown()
    return FrameCounts(counts_by_id=counts, total=sum(counts.values()), duration_s=duration_s)