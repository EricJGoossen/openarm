#!/usr/bin/env python3
"""
Correlate ssik solve() success/failure on OpenArm with how close each
sampled config's joints are to their OWN declared limits.

WHY
---
The Franka control experiment ruled out a test-methodology bug: Franka
passes 50/50 under the exact same code that gets OpenArm 10/50 (left) and
5/50 (right). So OpenArm's failure rate is real. The open question now is
WHERE it comes from:

- If failures cluster specifically around one joint (e.g. joint4, the
  suspected elbow singularity) being close to ITS limit, that's consistent
  with "the elbow-straight condition is the dominant driver, and it's
  wider than the earlier single-axis sweep suggested."
- If failures look roughly uniform regardless of how close any joint is
  to its limits, that points somewhere else entirely — most likely the
  auto-selected joint-lock choice in ssik's jointlock.seven_r dispatch
  being poorly conditioned for this arm's specific axis arrangement
  across most of the workspace, not a proximity-to-limit effect at all.

This script samples random reachable configs, records a per-joint "margin"
(normalized distance to the nearest of that joint's own limits, 0 = right
at a limit, 0.5 = exactly centered in its range), runs solve(), and
reports margins broken out by success/failure — plus which joint is
"tightest" (smallest margin) per sample, tallied separately for each group.

USAGE
-----
    uv run python openarm/scripts/openarm_failure_correlation.py --side left
    uv run python openarm/scripts/openarm_failure_correlation.py --side right
    uv run python openarm/scripts/openarm_failure_correlation.py --side both --n 100
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Path anchoring — same logic as the other OpenArm scripts in this repo
# ---------------------------------------------------------------------------


def _find_workspace_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "setup.sh").exists() and (candidate / "pyproject.toml").exists():
            return candidate
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            return candidate
    return start.parent


REPO_ROOT = _find_workspace_root(Path(__file__).resolve().parent)
DEFAULT_URDF = REPO_ROOT / "openarm_assets/src/openarm_assets/models/openarm/urdf/example/v1.urdf"

ARM_MODULES = {
    "left": "openarm.openarm_left_ik",
    "right": "openarm.openarm_right_ik",
}
ARM_JOINT_NAMES = {
    "left": [f"openarm_left_joint{i}" for i in range(1, 8)],
    "right": [f"openarm_right_joint{i}" for i in range(1, 8)],
}

RNG_SEED = 0


def load_artifact(side: str):
    return importlib.import_module(ARM_MODULES[side])


def load_pinocchio_model(urdf_path: Path):
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
    if not urdf_path.is_file():
        sys.exit(f"URDF not found at {urdf_path} (REPO_ROOT={REPO_ROOT}).")
    model = pin.buildModelFromUrdf(str(urdf_path))
    data = model.createData()
    return pin, model, data


def joint_q_index(model, joint_name: str) -> int:
    jid = model.getJointId(joint_name)
    if jid >= model.njoints:
        raise ValueError(f"joint {joint_name!r} not found in the URDF/model")
    return model.joints[jid].idx_q


def make_fk_fn(pin, model, data, base_link: str, ee_link: str):
    base_id = model.getFrameId(base_link)
    ee_id = model.getFrameId(ee_link)

    def fk(q_full: np.ndarray) -> np.ndarray:
        pin.forwardKinematics(model, data, q_full)
        pin.updateFramePlacements(model, data)
        T = data.oMf[base_id].inverse() * data.oMf[ee_id]
        return T.homogeneous

    return fk


def joint_margins(q_local: np.ndarray, lo_local: np.ndarray, hi_local: np.ndarray) -> np.ndarray:
    """Per-joint normalized distance to the nearest limit: 0 = sitting
    exactly at a limit, 0.5 = exactly centered in the joint's own range."""
    span = hi_local - lo_local
    dist_to_lo = q_local - lo_local
    dist_to_hi = hi_local - q_local
    return np.minimum(dist_to_lo, dist_to_hi) / span


def analyze_side(side: str, urdf_path: Path, n_samples: int, check_raw: bool):
    print(f"\n{'=' * 70}\n{side.upper()} ARM — {n_samples} samples\n{'=' * 70}")
    arm = load_artifact(side)
    pin, model, data = load_pinocchio_model(urdf_path)
    fk = make_fk_fn(pin, model, data, arm.BASE_LINK, arm.EE_LINK)

    joint_names = ARM_JOINT_NAMES[side]
    q_indices = [joint_q_index(model, jn) for jn in joint_names]
    lo_local = np.array([model.lowerPositionLimit[i] for i in q_indices])
    hi_local = np.array([model.upperPositionLimit[i] for i in q_indices])

    rng = np.random.default_rng(RNG_SEED)
    base_q_full = np.zeros(model.nq)

    all_margins = []  # (n_samples, 7)
    all_success = []  # (n_samples,) bool — default policy unless check_raw
    tightest_joint_idx = []  # (n_samples,) argmin index per sample

    for i in range(n_samples):
        q_local = rng.uniform(lo_local, hi_local)
        q_full = base_q_full.copy()
        for local_idx, global_idx in enumerate(q_indices):
            q_full[global_idx] = q_local[local_idx]
        T_target = fk(q_full)

        t0 = time.perf_counter()
        sols = arm.solve(T_target, respect_limits=not check_raw) if check_raw else arm.solve(T_target)
        elapsed = time.perf_counter() - t0

        margins = joint_margins(q_local, lo_local, hi_local)
        success = bool(sols)

        all_margins.append(margins)
        all_success.append(success)
        tightest_joint_idx.append(int(np.argmin(margins)))

        print(
            f"  {i + 1}/{n_samples}: {'OK' if success else 'FAIL'} "
            f"({elapsed:.2f}s) tightest={joint_names[int(np.argmin(margins))]} "
            f"(margin={margins.min():.3f})",
            flush=True,
        )

    all_margins = np.asarray(all_margins)  # (n, 7)
    all_success = np.asarray(all_success)  # (n,)
    tightest_joint_idx = np.asarray(tightest_joint_idx)

    n_fail = int((~all_success).sum())
    n_ok = int(all_success.sum())
    print(f"\n  {n_ok}/{n_samples} solved, {n_fail}/{n_samples} failed")

    # --- Per-joint margin comparison: success group vs failure group ---
    print(f"\n  {'joint':<22}{'mean margin (OK)':<20}{'mean margin (FAIL)':<20}{'corr w/ success'}")
    print("  " + "-" * 78)
    for j, name in enumerate(joint_names):
        margin_col = all_margins[:, j]
        mean_ok = margin_col[all_success].mean() if n_ok else float("nan")
        mean_fail = margin_col[~all_success].mean() if n_fail else float("nan")
        if all_success.std() > 0 and margin_col.std() > 0:
            corr = float(np.corrcoef(margin_col, all_success.astype(float))[0, 1])
        else:
            corr = float("nan")
        print(f"  {name:<22}{mean_ok:<20.3f}{mean_fail:<20.3f}{corr:.3f}")

    # --- Which joint is "tightest" (closest to its own limit), by group ---
    print(f"\n  Tightest-joint tally (which joint has the smallest margin per sample):")
    print(f"  {'joint':<22}{'count in OK group':<20}{'count in FAIL group'}")
    print("  " + "-" * 62)
    for j, name in enumerate(joint_names):
        count_ok = int(((tightest_joint_idx == j) & all_success).sum())
        count_fail = int(((tightest_joint_idx == j) & ~all_success).sum())
        print(f"  {name:<22}{count_ok:<20}{count_fail}")

    print(
        "\n  How to read this:\n"
        "  - If one joint's 'corr w/ success' is strongly positive (closer to\n"
        "    its own limit == more likely to fail) and/or dominates the FAIL\n"
        "    tightest-joint tally, that joint's proximity to its limit is a\n"
        "    real driver of failure — check whether that matches joint4.\n"
        "  - If correlations are all weak/inconsistent and the tightest-joint\n"
        "    tally looks similar between OK and FAIL groups, failures aren't\n"
        "    explained by proximity to any single joint's limit at all — points\n"
        "    at the auto-selected joint-lock dispatch itself being poorly\n"
        "    conditioned for this arm's axis arrangement broadly, independent\n"
        "    of where in the joint space you are."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--side", choices=["left", "right", "both"], default="both")
    parser.add_argument("--n", type=int, default=60, help="number of random samples per arm")
    parser.add_argument(
        "--raw",
        action="store_true",
        help="use respect_limits=False instead of the default policy",
    )
    args = parser.parse_args()

    sides = ["left", "right"] if args.side == "both" else [args.side]
    for side in sides:
        analyze_side(side, args.urdf, args.n, args.raw)


if __name__ == "__main__":
    main()