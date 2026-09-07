"""Telemetry recording. Every real-hardware trial in every stage script
records raw joint-state history, because "the script printed PASS" is never
by itself sufficient evidence that something happened correctly on real
hardware -- the recorded data is what later analysis (including the
independent collision check in `collision_check.py`) actually runs against.

Two independent recorders are provided:

- `JointStateRecorder`: subscribes to a joint-state topic in-process and
  buffers (timestamp, name -> position/velocity/effort) in memory. Used for
  the automated checks each script runs immediately after a trial.
- `RosbagRecorder`: shells out to `ros2 bag record` for a full, independent,
  out-of-process archive of everything on the topics you point it at. This
  is the "belt" to JointStateRecorder's "suspenders" -- if the in-process
  recorder has a bug, the bag is still there.

Both are optional to use individually, but at least `JointStateRecorder`
should be running for the whole duration of any real-hardware trial.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class JointSample:
    t: float  # wall-clock seconds (time.time()), not ROS time -- see note below
    positions: dict[str, float]
    velocities: dict[str, float] = field(default_factory=dict)
    efforts: dict[str, float] = field(default_factory=dict)


class JointStateRecorder:
    """Buffers sensor_msgs/JointState messages with wall-clock receipt time.

    Wall-clock time (not the message's own header stamp) is used deliberately:
    what we care about for latency/simultaneity measurements is when *this
    process observed* the state, which is what matters for e.g. "how long
    after I called request_abort() did motion actually stop" -- mixing that
    with a different clock domain (simulation time, or a differently-synced
    hardware clock) would be a subtle source of exactly the kind of error
    this whole test suite exists to catch.
    """

    def __init__(self, node, topic: str = "/joint_states", joint_filter: list[str] | None = None):
        from sensor_msgs.msg import JointState  # local import: keep module importable without ROS

        self._node = node
        self._joint_filter = set(joint_filter) if joint_filter else None
        self._lock = threading.Lock()
        self._samples: list[JointSample] = []
        self._recording = False
        self._sub = node.create_subscription(JointState, topic, self._cb, 50)

    def _cb(self, msg) -> None:
        if not self._recording:
            return
        t = time.time()
        pos, vel, eff = {}, {}, {}
        for i, name in enumerate(msg.name):
            if self._joint_filter is not None and name not in self._joint_filter:
                continue
            if i < len(msg.position):
                pos[name] = float(msg.position[i])
            if i < len(msg.velocity):
                vel[name] = float(msg.velocity[i])
            if i < len(msg.effort):
                eff[name] = float(msg.effort[i])
        with self._lock:
            self._samples.append(JointSample(t=t, positions=pos, velocities=vel, efforts=eff))

    def start(self) -> None:
        with self._lock:
            self._samples.clear()
            self._recording = True

    def stop(self) -> list[JointSample]:
        with self._lock:
            self._recording = False
            return list(self._samples)

    def samples(self) -> list[JointSample]:
        with self._lock:
            return list(self._samples)

    def latest(self, timeout_s: float = 5.0) -> JointSample | None:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            with self._lock:
                if self._samples:
                    return self._samples[-1]
            time.sleep(0.02)
        return None

    def wait_for_joints(self, joint_names: list[str], timeout_s: float = 10.0) -> dict[str, float] | None:
        """Block until a sample containing every name in `joint_names` has
        arrived, and return that sample's positions. Use this to get a
        trustworthy 'current position' snapshot before planning or ramping,
        rather than assuming the first message has everything you need.
        """
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            s = self.latest(timeout_s=0.5)
            if s and all(n in s.positions for n in joint_names):
                return {n: s.positions[n] for n in joint_names}
            time.sleep(0.05)
        return None

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            data = [
                {"t": s.t, "positions": s.positions, "velocities": s.velocities, "efforts": s.efforts}
                for s in self._samples
            ]
        with open(path, "w") as f:
            json.dump(data, f, indent=1)


class RosbagRecorder:
    """Thin wrapper around `ros2 bag record` for full independent archival.
    Optional -- pass `enabled=False` (or just don't use it) if `ros2 bag` /
    disk space isn't available in a given session, but prefer to leave it on.
    """

    def __init__(self, output_dir: str | Path, topics: list[str] | None = None, enabled: bool = True):
        self._output_dir = Path(output_dir)
        self._topics = topics or ["-a"]  # -a = record everything, simplest default
        self._enabled = enabled
        self._proc: subprocess.Popen | None = None

    def start(self) -> None:
        if not self._enabled:
            return
        self._output_dir.parent.mkdir(parents=True, exist_ok=True)
        cmd = ["ros2", "bag", "record", "-o", str(self._output_dir)] + list(self._topics)
        try:
            self._proc = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        except FileNotFoundError:
            print("  [telemetry] 'ros2 bag record' not found -- continuing without bag archival.")
            self._proc = None
        time.sleep(1.0)  # let recording actually start before the caller does anything

    def stop(self) -> None:
        if self._proc is None:
            return
        self._proc.send_signal(signal.SIGINT)
        try:
            self._proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._proc.kill()
        self._proc = None


class TelemetrySession:
    """Bundles an in-process JointStateRecorder with an optional rosbag, and
    writes everything to a predictable per-trial directory:

        <root>/<stage>/<timestamp>_<label>/joint_states.json
        <root>/<stage>/<timestamp>_<label>/manifest.json
        <root>/<stage>/<timestamp>_<label>/bag/...   (if rosbag enabled)
    """

    def __init__(
        self,
        root: str | Path,
        stage: str,
        label: str,
        recorder: JointStateRecorder,
        use_rosbag: bool = True,
        extra_topics: list[str] | None = None,
    ):
        from .operator_io import timestamp

        self.dir = Path(root) / stage / f"{timestamp()}_{label}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.recorder = recorder
        self.bag = RosbagRecorder(self.dir / "bag", topics=extra_topics, enabled=use_rosbag)
        self._meta: dict[str, Any] = {"stage": stage, "label": label}

    def __enter__(self) -> "TelemetrySession":
        self.bag.start()
        self.recorder.start()
        self._meta["start_wall_time"] = time.time()
        return self

    def __exit__(self, *exc) -> None:
        self._meta["end_wall_time"] = time.time()
        samples = self.recorder.stop()
        self.bag.stop()
        self.recorder.save(self.dir / "joint_states.json")
        with open(self.dir / "manifest.json", "w") as f:
            json.dump(self._meta, f, indent=1)
        self._meta["num_samples"] = len(samples)
        return None

    def note(self, key: str, value: Any) -> None:
        self._meta[key] = value