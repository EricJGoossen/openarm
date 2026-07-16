#!/usr/bin/env python3
"""
Characterize the ssik dead zone around OpenArm's fully-extended home
configuration (q = zeros), where solve(T_HOME) returns [] for both arms.

This is exploratory measurement, not a pass/fail check, hence a standalone
script rather than a pytest test.

WHAT IT DOES
------------
1. Confirms (or refutes) that the empty result at q=0 is independent of
   URDF joint limits, by calling solve() with respect_limits=True vs False
   and checking they agree. If they disagree, joint limits ARE part of
   the story and the "kinematic singularity" explanation is incomplete.
2. For each joint independently (holding all others at 0), sweeps outward
   from 0 toward that joint's own limit and finds — via a coarse scan
   followed by bisection — the smallest deviation at which solve() starts
   returning a nonempty result again. Reports this per joint, per arm,
   under both respect_limits policies.

This tells you two things directly: (a) whether limits matter at all,
and (b) which joint(s) actually drive the singularity and how far you
need to stay from q=0 along each one to get a valid IK result.

USAGE
-----
    uv run python scripts/sweep_ik_deadzone.py
    uv run python scripts/sweep_ik_deadzone.py --urdf /abs/path/to/v1.urdf
    uv run python scripts/sweep_ik_deadzone.py --side left --joints 4
    uv run python scripts/sweep_ik_deadzone.py --n-coarse 80 --tol 1e-7

Run with `uv run` from the openarm_robot-code workspace root so the
`openarm` package (openarm_left_ik / openarm_right_ik) and Pinocchio are
both importable from the shared workspace venv.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Path anchoring — same logic as tests/test_openarm_ssik.py, kept in sync
# on purpose so "the URDF path" means the same thing in both places.
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

# URDF joint names per arm, in q1..q7 order — used only to look up each
# joint's Pinocchio q-index and its own limits; limit VALUES always come
# from the model, never hardcoded here.
ARM_JOINT_NAMES = {
    "left": [f"openarm_left_joint{i}" for i in range(1, 8)],
    "right": [f"openarm_right_joint{i}" for i in range(1, 8)],
}


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------


def load_artifact(side: str):
    return importlib.import_module(ARM_MODULES[side])


def load_pinocchio_model(urdf_path: Path):
    try:
        import pinocchio as pin
    except ImportError as e:
        sys.exit(
            f"pinocchio is required for this script (to compute FK for "
            f"arbitrary joint configs) and isn't importable: {e!r}\n"
            f"Install it in the workspace venv, e.g.:\n"
            f"    uv add pinocchio\n"
            f"then re-run with `uv run python scripts/sweep_ik_deadzone.py`."
        )
    if not urdf_path.is_file():
        sys.exit(
            f"URDF not found at {urdf_path}\n"
            f"(REPO_ROOT resolved to {REPO_ROOT}). Pass --urdf explicitly "
            f"if this workspace has a different layout."
        )
    model = pin.buildModelFromUrdf(str(urdf_path))
    data = model.createData()
    return pin, model, data


def joint_q_index(model, joint_name: str) -> int:
    jid = model.getJointId(joint_name)
    if jid >= model.njoints:
        raise ValueError(
            f"joint {joint_name!r} not found in the URDF/model — check "
            f"ARM_JOINT_NAMES matches your actual joint names"
        )
    return model.joints[jid].idx_q


def make_fk_fn(pin, model, data, base_link: str, ee_link: str):
    base_id = model.getFrameId(base_link)
    ee_id = model.getFrameId(ee_link)

    def fk(q_full: np.ndarray) -> np.ndarray:
        pin.forwardKinematics(model, data, q_full)
        pin.updateFramePlacements(model, data)
        T_base_ee = data.oMf[base_id].inverse() * data.oMf[ee_id]
        return T_base_ee.homogeneous

    return fk


# ---------------------------------------------------------------------------
# Step 1: is this actually independent of joint limits?
# ---------------------------------------------------------------------------


def check_limits_independence(arm, T_home: np.ndarray) -> None:
    sols_true = arm.solve(T_home, respect_limits=True)
    sols_false = arm.solve(T_home, respect_limits=False)
    print(
        f"  solve(T_HOME, respect_limits=True)  -> {len(sols_true)} solution(s)\n"
        f"  solve(T_HOME, respect_limits=False) -> {len(sols_false)} solution(s)"
    )
    if bool(sols_true) != bool(sols_false):
        print(
            "  MISMATCH: limits policy changes the outcome at q=0 — this "
            "IS partly a limits story, not a pure singularity. Investigate "
            "which joint's limit is doing this before trusting the sweep "
            "below."
        )
    elif not sols_true and not sols_false:
        print(
            "  Both agree: EMPTY regardless of respect_limits. Confirms "
            "the failure is NOT a joint-limit filtering artifact — the "
            "algebraic solver itself finds zero candidates at this pose, "
            "independent of any limits policy."
        )
    else:
        print("  Both agree: NONEMPTY under both policies. No dead zone at exactly q=0.")


# ---------------------------------------------------------------------------
# Step 2: per-joint sweep to find the edge of the dead zone
# ---------------------------------------------------------------------------


def find_boundary(
    arm,
    fk,
    base_q: np.ndarray,
    joint_idx: int,
    target: float,
    respect_limits: bool,
    n_coarse: int,
    tol: float,
    bisect_iters: int = 30,
):
    """Scan from 0 toward `target` (a signed limit value): find the first
    coarse sample where solve() succeeds, then bisect between the last
    failure and first success for a precise boundary.

    Returns (boundary_value_or_None, status_str).
    """
    if abs(target) < 1e-9:
        return None, "zero-width range (limit equals 0 in this direction)"

    samples = np.linspace(0.0, target, n_coarse + 1)[1:]  # skip 0, already known to fail
    last_fail = 0.0
    first_success = None
    for v in samples:
        q = base_q.copy()
        q[joint_idx] = v
        T = fk(q)
        sols = arm.solve(T, respect_limits=respect_limits)
        if sols:
            first_success = v
            break
        last_fail = v

    if first_success is None:
        return None, f"still failing across the full range to {target:.4f} rad"

    lo, hi = last_fail, first_success
    for _ in range(bisect_iters):
        mid = 0.5 * (lo + hi)
        q = base_q.copy()
        q[joint_idx] = mid
        T = fk(q)
        sols = arm.solve(T, respect_limits=respect_limits)
        if sols:
            hi = mid
        else:
            lo = mid
        if abs(hi - lo) < tol:
            break
    return hi, "found"


def sweep_side(side: str, urdf_path: Path, joint_selection, n_coarse: int, tol: float):
    print(f"\n{'=' * 70}\n{side.upper()} ARM\n{'=' * 70}")
    arm = load_artifact(side)
    pin, model, data = load_pinocchio_model(urdf_path)
    fk = make_fk_fn(pin, model, data, arm.BASE_LINK, arm.EE_LINK)

    joint_names = ARM_JOINT_NAMES[side]
    q_indices = [joint_q_index(model, name) for name in joint_names]
    lo_limits = [model.lowerPositionLimit[i] for i in q_indices]
    hi_limits = [model.upperPositionLimit[i] for i in q_indices]

    T_home = np.asarray(arm.T_HOME)

    print("\nStep 1: is the q=0 failure a joint-limit artifact?")
    check_limits_independence(arm, T_home)

    # base_q is a FULL model configuration (every joint in the combined
    # URDF: both arms + fingers), initialized to zero; only this arm's
    # own 7 indices ever get perturbed below. Other joints don't affect
    # this arm's base->ee FK (unrelated kinematic branch), so leaving
    # them at zero is fine.
    base_q_full = np.zeros(model.nq)

    print("\nStep 2: per-joint sweep (holding all other joints at 0)")
    header = f"{'joint':<10}{'dir':<7}{'limits=True':<20}{'limits=False':<20}{'match'}"
    print(header)
    print("-" * len(header))

    for local_idx, name in enumerate(joint_names):
        if joint_selection is not None and (local_idx + 1) not in joint_selection:
            continue
        q_idx = q_indices[local_idx]
        lo, hi = lo_limits[local_idx], hi_limits[local_idx]

        for direction_name, target in (("lower", lo), ("upper", hi)):
            if abs(target) < 1e-9:
                continue  # this joint has no range in this direction (e.g. joint4 lower=0)

            b_true, status_true = find_boundary(
                arm, fk, base_q_full, q_idx, target, True, n_coarse, tol
            )
            b_false, status_false = find_boundary(
                arm, fk, base_q_full, q_idx, target, False, n_coarse, tol
            )

            def fmt(b, status):
                return f"{b:.5f} rad" if b is not None else status

            if (b_true is None) != (b_false is None):
                match = "DIFFERS"
            elif b_true is None:
                match = "yes"  # both failed the same way across the full range
            else:
                match = "yes" if abs(b_true - b_false) < 10 * tol else "DIFFERS"

            print(
                f"{name:<10}{direction_name:<7}"
                f"{fmt(b_true, status_true):<20}{fmt(b_false, status_false):<20}{match}"
            )


def parse_joint_selection(arg):
    if arg is None:
        return None
    return {int(x) for x in arg.split(",")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--side", choices=["left", "right", "both"], default="both")
    parser.add_argument(
        "--joints",
        type=str,
        default=None,
        help="comma-separated joint numbers (1-7) to sweep; default: all",
    )
    parser.add_argument("--n-coarse", type=int, default=40)
    parser.add_argument("--tol", type=float, default=1e-6)
    args = parser.parse_args()

    joint_selection = parse_joint_selection(args.joints)
    sides = ["left", "right"] if args.side == "both" else [args.side]

    for side in sides:
        sweep_side(side, args.urdf, joint_selection, args.n_coarse, args.tol)


if __name__ == "__main__":
    main()