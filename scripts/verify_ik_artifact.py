#!/usr/bin/env python3
"""
Generic round-trip verifier: test ANY ssik-built artifact against ANY
URDF with the exact same methodology used throughout this investigation
(unseeded default + raw round-trip, same RNG seed), so results are
directly comparable to:

    OpenArm (original URDF):  10/50 (left), 5/50 (right)
    Franka Panda (control):   50/50

USAGE
-----
    # after: ssik build v1_asymmetric_test.urdf --base openarm_left_link0 \\
    #            --ee openarm_left_hand_tcp
    # and renaming the output so it doesn't collide with the real artifact,
    # e.g. openarm_left_ik_asym.py, placed somewhere importable:

    uv run python openarm/scripts/verify_symmetry_hypothesis.py \\
        --module openarm_left_ik_asym \\
        --urdf openarm_assets/.../v1_asymmetric_test.urdf \\
        --joints openarm_left_joint1,openarm_left_joint2,openarm_left_joint3,openarm_left_joint4,openarm_left_joint5,openarm_left_joint6,openarm_left_joint7

If --module isn't importable directly (e.g. it's a loose file, not in the
package), pass --module-path pointing at its containing directory and
it'll be added to sys.path first.
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path

import numpy as np

N_RANDOM_CONFIGS = 50
RNG_SEED = 0
FK_ATOL_ROT = 1e-4


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


def pin_fk(pin, model, data, base_link, ee_link, q_full):
    pin.forwardKinematics(model, data, q_full)
    pin.updateFramePlacements(model, data)
    base_id = model.getFrameId(base_link)
    ee_id = model.getFrameId(ee_link)
    T = data.oMf[base_id].inverse() * data.oMf[ee_id]
    return T.homogeneous


def fk_residual(T_a, T_b) -> float:
    return float(np.linalg.norm(T_a - T_b, ord="fro"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", required=True, help="importable module name of the built artifact")
    parser.add_argument("--module-path", type=Path, default=None, help="dir to add to sys.path for --module")
    parser.add_argument("--urdf", type=Path, required=True, help="URDF this artifact was built against")
    parser.add_argument(
        "--joints",
        required=True,
        help="comma-separated joint names, in q1..qN order, for this arm's chain",
    )
    parser.add_argument("--n", type=int, default=N_RANDOM_CONFIGS)
    args = parser.parse_args()

    if args.module_path:
        sys.path.insert(0, str(args.module_path))

    try:
        arm = importlib.import_module(args.module)
    except ImportError as e:
        sys.exit(f"Couldn't import {args.module!r}: {e!r}")

    print(f"{args.module}: BASE_LINK={arm.BASE_LINK!r} EE_LINK={arm.EE_LINK!r} DOF={arm.DOF}")

    pin = load_pin()
    if not args.urdf.is_file():
        sys.exit(f"URDF not found at {args.urdf}")
    model = pin.buildModelFromUrdf(str(args.urdf))
    data = model.createData()

    joint_names = args.joints.split(",")
    if len(joint_names) != arm.DOF:
        sys.exit(
            f"--joints has {len(joint_names)} names but {args.module}.DOF={arm.DOF} "
            f"— these must match 1:1 in order"
        )
    missing = [jn for jn in joint_names if model.getJointId(jn) >= model.njoints]
    if missing:
        sys.exit(f"Joint names not found in this URDF: {missing}")

    q_indices = [model.joints[model.getJointId(jn)].idx_q for jn in joint_names]
    lo_full = np.asarray(model.lowerPositionLimit)
    hi_full = np.asarray(model.upperPositionLimit)

    # Sanity check: does T_HOME match this URDF's own FK at q=0?
    q_zero_full = np.zeros(model.nq)
    T_pin_home = pin_fk(pin, model, data, arm.BASE_LINK, arm.EE_LINK, q_zero_full)
    home_residual = fk_residual(T_pin_home, np.asarray(arm.T_HOME))
    print(f"T_HOME residual vs this URDF's Pinocchio FK: {home_residual:.2e}")

    rng = np.random.default_rng(RNG_SEED)
    n_default_ok = 0
    n_raw_ok = 0
    for i in range(args.n):
        q_local = rng.uniform(lo_full[q_indices], hi_full[q_indices])
        q_full = np.zeros(model.nq)
        for local_idx, global_idx in enumerate(q_indices):
            q_full[global_idx] = q_local[local_idx]
        T_target = pin_fk(pin, model, data, arm.BASE_LINK, arm.EE_LINK, q_full)

        t0 = time.perf_counter()
        default_sols = arm.solve(T_target)
        raw_sols = arm.solve(T_target, respect_limits=False)
        elapsed = time.perf_counter() - t0

        if default_sols:
            n_default_ok += 1
        if raw_sols:
            n_raw_ok += 1

        print(
            f"  {i + 1}/{args.n}: default={'found' if default_sols else 'EMPTY'}, "
            f"raw={'found' if raw_sols else 'EMPTY'} ({elapsed:.3f}s)",
            flush=True,
        )

    print(f"\n{'=' * 60}\nRESULT\n{'=' * 60}")
    print(f"Default-policy pass rate: {n_default_ok}/{args.n}")
    print(f"Raw (no limits) pass rate: {n_raw_ok}/{args.n}")
    print(
        "\nCompare against:\n"
        "  OpenArm original URDF:  10/50 (left), 5/50 (right)\n"
        "  Franka Panda (control): 50/50\n"
        "\nIf this desymmetrized build lands much closer to Franka's 50/50 "
        "than to OpenArm's original 10/50, that's strong confirmation of "
        "the symmetric-DH hypothesis. If it's still down near 10/50, the "
        "hypothesis doesn't hold (at least not from these specific "
        "parameters alone), and something else is going on."
    )


if __name__ == "__main__":
    main()