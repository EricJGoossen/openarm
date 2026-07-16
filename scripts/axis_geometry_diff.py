#!/usr/bin/env python3
"""
Compare axis geometry between OpenArm (left arm) and ssik's Franka
reference (via robot_descriptions), by computing each joint's world-frame
rotation axis via Pinocchio's geometric Jacobian — robust, since it
avoids any hand-derived DH-table arithmetic or needing to know how
Pinocchio internally represents axis-aligned revolute joints.

WHY
---
The failure-correlation script ruled out any single joint's proximity to
ITS OWN limit as the driver of OpenArm's low ssik pass rate — the
tightest-joint tally and per-joint correlations were both roughly uniform
across all seven joints. A structural, geometry-level explanation is the
natural next candidate: if two of OpenArm's axes come back into (near-)
alignment partway through the chain — even with other joints physically
between them — that can create rank-deficient/degenerate conditions
across a much broader region of the joint space than one isolated
singularity would, which fits the ~80-90% failure rate far better than a
narrow dead-zone theory.

This builds a full pairwise angle table across all 7 axes (not just
consecutive ones) for both arms, at the home configuration and at a
handful of random configs, and flags near-parallel (<15 deg) pairs.

USAGE
-----
    uv run python openarm/scripts/axis_geometry_diff.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def _find_workspace_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "setup.sh").exists() and (candidate / "pyproject.toml").exists():
            return candidate
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            return candidate
    return start.parent


REPO_ROOT = _find_workspace_root(Path(__file__).resolve().parent)
OPENARM_URDF = REPO_ROOT / "openarm_assets/src/openarm_assets/models/openarm/urdf/example/v1.urdf"

OPENARM_JOINTS = [f"openarm_left_joint{i}" for i in range(1, 8)]
FRANKA_JOINTS = [f"panda_joint{i}" for i in range(1, 8)]

NEAR_PARALLEL_DEG = 15.0
N_RANDOM_SAMPLES = 8
RNG_SEED = 0


def load_pin():
    try:
        import pinocchio as pin
    except ImportError as e:
        sys.exit(
            f"pinocchio not importable: {e!r}\n"
            f"(wrong PyPI package? `uv remove pinocchio && uv add pin`)"
        )
    if not hasattr(pin, "buildModelFromUrdf"):
        sys.exit(
            f"Wrong `pinocchio` package (no buildModelFromUrdf, version "
            f"{getattr(pin, '__version__', 'unknown')!r}). Fix: "
            f"`uv remove pinocchio && uv add pin`."
        )
    return pin


def joint_axes_at_q(pin, model, data, joint_names, q_full):
    """World-frame unit axis direction for each named joint, at the given
    full configuration — the angular block of that joint's own Jacobian
    column, LOCAL_WORLD_ALIGNED so it's expressed in world orientation.
    Robust to axis sign / internal joint-type representation."""
    pin.forwardKinematics(model, data, q_full)
    pin.computeJointJacobians(model, data, q_full)
    axes = []
    for jn in joint_names:
        jid = model.getJointId(jn)
        J = pin.getJointJacobian(model, data, jid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        col = model.joints[jid].idx_v
        angular = J[3:6, col]
        norm = np.linalg.norm(angular)
        axes.append(angular / norm if norm > 1e-9 else np.zeros(3))
    return np.asarray(axes)  # (7, 3)


def pairwise_angles(axes: np.ndarray) -> np.ndarray:
    """(7,7) matrix of angles in degrees between axis LINES (0-90; sign
    ignored, since a parallel or antiparallel axis pair is kinematically
    the same 'shared axis' condition)."""
    n = axes.shape[0]
    angles = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            cosang = np.clip(abs(np.dot(axes[i], axes[j])), 0, 1)
            angles[i, j] = np.degrees(np.arccos(cosang))
    return angles


def print_angle_table(angles: np.ndarray, joint_names: list[str]) -> list[tuple]:
    n = len(joint_names)
    short = [jn[-1] for jn in joint_names]  # just the trailing digit
    header = "        " + "".join(f"{s:>7}" for s in short)
    print("  " + header)
    near_parallel = []
    for i in range(n):
        row = f"  j{short[i]:<7}"
        for j in range(n):
            if i == j:
                row += f"{'--':>7}"
                continue
            row += f"{angles[i, j]:>7.1f}"
            if i < j and angles[i, j] < NEAR_PARALLEL_DEG:
                near_parallel.append((joint_names[i], joint_names[j], round(angles[i, j], 1)))
        print(row)
    return near_parallel


def analyze_arm(label: str, pin, urdf_path: Path, joint_names: list[str]):
    print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")
    if not urdf_path.is_file():
        print(f"  URDF not found at {urdf_path} — skipping")
        return

    model = pin.buildModelFromUrdf(str(urdf_path))
    data = model.createData()

    missing = [jn for jn in joint_names if model.getJointId(jn) >= model.njoints]
    if missing:
        print(f"  Joint names not found in this URDF: {missing} — skipping")
        return

    q_indices = [model.joints[model.getJointId(jn)].idx_q for jn in joint_names]
    lo = np.asarray(model.lowerPositionLimit)
    hi = np.asarray(model.upperPositionLimit)

    print("\nAt HOME (q=0):")
    q_home = np.zeros(model.nq)
    axes_home = joint_axes_at_q(pin, model, data, joint_names, q_home)
    angles_home = pairwise_angles(axes_home)
    near_parallel_home = print_angle_table(angles_home, joint_names)
    print(
        f"\n  Near-parallel pairs (<{NEAR_PARALLEL_DEG:.0f} deg) at home: "
        f"{near_parallel_home if near_parallel_home else 'none'}"
    )

    print(f"\nAt {N_RANDOM_SAMPLES} random reachable configs:")
    rng = np.random.default_rng(RNG_SEED)
    total_pairs = 0
    all_pairs_seen = set()
    for s in range(N_RANDOM_SAMPLES):
        q_full = np.zeros(model.nq)
        q_local = rng.uniform(lo[q_indices], hi[q_indices])
        for local_idx, global_idx in enumerate(q_indices):
            q_full[global_idx] = q_local[local_idx]
        axes = joint_axes_at_q(pin, model, data, joint_names, q_full)
        angles = pairwise_angles(axes)
        pairs = [
            (joint_names[i], joint_names[j], round(angles[i, j], 1))
            for i in range(len(joint_names))
            for j in range(i + 1, len(joint_names))
            if angles[i, j] < NEAR_PARALLEL_DEG
        ]
        total_pairs += len(pairs)
        for p in pairs:
            all_pairs_seen.add((p[0], p[1]))
        print(f"  sample {s + 1}/{N_RANDOM_SAMPLES}: near-parallel pairs = {pairs}")

    print(
        f"\n  Total near-parallel-pair occurrences across {N_RANDOM_SAMPLES} "
        f"random configs: {total_pairs}\n"
        f"  Distinct joint pairs that went near-parallel at least once: "
        f"{sorted(all_pairs_seen)}"
    )


def main():
    pin = load_pin()

    analyze_arm("OPENARM LEFT ARM", pin, OPENARM_URDF, OPENARM_JOINTS)

    try:
        from robot_descriptions import panda_description
    except ImportError as e:
        sys.exit(
            f"robot_descriptions not importable: {e!r}\n"
            f"uv add --dev robot_descriptions"
        )
    analyze_arm(
        "FRANKA PANDA (control)",
        pin,
        Path(panda_description.URDF_PATH),
        FRANKA_JOINTS,
    )

    print(
        f"\n{'=' * 70}\n"
        "How to read this:\n"
        "- 'Near-parallel pairs' includes non-consecutive joints on purpose\n"
        "  (e.g. joint1 & joint4), since two axes realigning even with other\n"
        "  joints physically between them can still create broad rank-\n"
        "  deficiency across the workspace, not just at one isolated pose.\n"
        "- If OpenArm shows persistent near-parallel pairs across most/all\n"
        "  random samples where Franka shows few or none, that's a real,\n"
        "  structural geometric difference worth reporting precisely.\n"
        "- If both arms show a similar number of near-parallel pairs, the\n"
        "  axis-geometry theory doesn't hold up either, and the search\n"
        "  continues elsewhere."
    )


if __name__ == "__main__":
    main()