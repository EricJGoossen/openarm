#!/usr/bin/env python3
"""
Control experiment: run the exact same validation methodology used against
OpenArm's ssik-built artifacts, against ssik's own PREBUILT `franka_panda_ik`
instead, using an independently-sourced Franka Panda URDF (via the
`robot_descriptions` package — the same source ssik's own test suite uses
for its fixture-parity checks against upstream URDFs).

WHY THIS MATTERS
----------------
OpenArm's artifacts showed ~10-20% pass rates on random reachable configs
under both default and seeded solve(), despite near-machine-precision FK
residuals whenever a solution WAS found. Before concluding anything about
OpenArm's geometry or ssik's solver coverage, we need a control: does a
known-good, widely-used 7R arm (Franka Panda) pass at a much higher rate
under this EXACT same test code and methodology?

- Franka passes at a high rate here -> the harness is sound. OpenArm's low
  rate is real and specific to OpenArm (geometry, URDF authoring, or a
  genuine solver-coverage gap on this particular kinematic arrangement).
- Franka ALSO fails broadly here -> there's a bug in the shared test
  methodology itself (FK convention, joint indexing, embedding, etc.), and
  none of the OpenArm conclusions drawn so far should be trusted until
  that's found and fixed.

SETUP
-----
    uv add --dev robot_descriptions   # one-time; pulls in a validated
                                       # Franka Panda URDF + meshes
    uv run python scripts/franka_control_check.py

Requires `pin` (real Pinocchio) and `ssik` already installed in the venv,
same as the rest of this investigation.
"""

from __future__ import annotations

import sys
import time

import numpy as np

N_RANDOM_CONFIGS = 50
RNG_SEED = 0
FK_ATOL_ROT = 1e-4
JOINT_DIST_NEAR_SEED = 0.1

# Standard Franka Panda joint names (franka_description convention).
# If robot_descriptions' Panda URDF uses different names, the frame/joint
# existence check below will say so explicitly instead of crashing opaquely.
PANDA_JOINT_NAMES = [f"panda_joint{i}" for i in range(1, 8)]


def _rng():
    return np.random.default_rng(RNG_SEED)


def _is_valid_se3(T: np.ndarray) -> bool:
    if T.shape != (4, 4):
        return False
    R = T[:3, :3]
    return (
        np.allclose(R @ R.T, np.eye(3), atol=1e-6)
        and np.isclose(np.linalg.det(R), 1.0, atol=1e-6)
        and np.allclose(T[3, :], [0, 0, 0, 1])
    )


def _fk_residual(T_a: np.ndarray, T_b: np.ndarray) -> float:
    return float(np.linalg.norm(T_a - T_b, ord="fro"))


def load_ssik_franka():
    try:
        from ssik.prebuilt import franka_panda_ik
    except ImportError as e:
        sys.exit(
            f"Couldn't import ssik.prebuilt.franka_panda_ik: {e!r}\n"
            f"Make sure `ssik` is installed in this venv (it already should "
            f"be, since it's what built the OpenArm artifacts)."
        )
    return franka_panda_ik


def load_pinocchio_model_for_panda():
    try:
        import pinocchio as pin
    except ImportError as e:
        sys.exit(
            f"pinocchio not importable: {e!r}\n"
            f"(if this is the wrong `pinocchio` PyPI package again: "
            f"`uv remove pinocchio && uv add pin`)"
        )
    if not hasattr(pin, "buildModelFromUrdf"):
        sys.exit(
            f"This is the wrong `pinocchio` package (no buildModelFromUrdf, "
            f"version {getattr(pin, '__version__', 'unknown')!r}). Fix: "
            f"`uv remove pinocchio && uv add pin`."
        )

    try:
        from robot_descriptions import panda_description
    except ImportError as e:
        sys.exit(
            f"robot_descriptions (or its Panda description) not importable: "
            f"{e!r}\nInstall with:\n    uv add --dev robot_descriptions\n"
            f"then re-run."
        )

    urdf_path = panda_description.URDF_PATH
    print(f"Using Franka Panda URDF from robot_descriptions: {urdf_path}")
    model = pin.buildModelFromUrdf(urdf_path)
    data = model.createData()
    return pin, model, data, urdf_path


def check_frames_and_joints(model, base_link: str, ee_link: str):
    problems = []
    if not model.existFrame(base_link):
        problems.append(f"BASE_LINK {base_link!r} not found as a frame in this URDF")
    if not model.existFrame(ee_link):
        problems.append(f"EE_LINK {ee_link!r} not found as a frame in this URDF")

    missing_joints = [jn for jn in PANDA_JOINT_NAMES if model.getJointId(jn) >= model.njoints]
    if missing_joints:
        problems.append(f"joint names not found: {missing_joints}")

    if problems:
        all_frame_names = [f.name for f in model.frames]
        all_joint_names = [model.names[i] for i in range(model.njoints)]
        sys.exit(
            "Frame/joint name mismatch between ssik's franka_panda_ik and "
            "this URDF:\n  " + "\n  ".join(problems) + "\n\n"
            f"Frames available in this URDF containing 'base' or 'link0': "
            f"{[n for n in all_frame_names if 'base' in n.lower() or 'link0' in n.lower()]}\n"
            f"Frames available containing 'ee' or 'link8' or 'hand' or 'tcp': "
            f"{[n for n in all_frame_names if any(s in n.lower() for s in ('ee', 'link8', 'hand', 'tcp'))]}\n"
            f"All joint names in this URDF: {all_joint_names}\n"
            f"Update BASE_LINK/EE_LINK/PANDA_JOINT_NAMES at the top of this "
            f"script to match, then re-run."
        )


def arm_q_indices(model):
    return [model.joints[model.getJointId(jn)].idx_q for jn in PANDA_JOINT_NAMES]


def embed_arm_q(full_q_template, q_indices, arm_local_q):
    full_q = full_q_template.copy()
    for local_idx, global_idx in enumerate(q_indices):
        full_q[global_idx] = arm_local_q[local_idx]
    return full_q


def pin_fk(pin, model, data, base_link, ee_link, q_full):
    pin.forwardKinematics(model, data, q_full)
    pin.updateFramePlacements(model, data)
    base_id = model.getFrameId(base_link)
    ee_id = model.getFrameId(ee_link)
    T_base_ee = data.oMf[base_id].inverse() * data.oMf[ee_id]
    return T_base_ee.homogeneous


# ---------------------------------------------------------------------------
# The three checks, mirrored from tests/test_openarm_ssik.py
# ---------------------------------------------------------------------------


def check_home_pose(arm, pin, model, data, q_indices):
    print("\n--- Check 1: does T_HOME match independent Pinocchio FK at q=0? ---")
    q_zero_full = np.zeros(model.nq)
    T_pin = pin_fk(pin, model, data, arm.BASE_LINK, arm.EE_LINK, q_zero_full)
    T_ssik = np.asarray(arm.T_HOME)
    residual = _fk_residual(T_pin, T_ssik)
    print(f"  T_HOME residual vs Pinocchio: {residual:.2e} ({'OK' if residual < FK_ATOL_ROT else 'MISMATCH'})")
    return residual < FK_ATOL_ROT


def check_round_trip(arm, pin, model, data, q_indices, lo, hi):
    print(f"\n--- Check 2: unseeded round-trip over {N_RANDOM_CONFIGS} random reachable configs ---")
    rng = _rng()
    n_default_ok = 0
    n_raw_ok = 0
    for i in range(N_RANDOM_CONFIGS):
        q_true_full = rng.uniform(lo, hi)
        T_target = pin_fk(pin, model, data, arm.BASE_LINK, arm.EE_LINK, q_true_full)

        t0 = time.perf_counter()
        default_sols = arm.solve(T_target)
        raw_sols = arm.solve(T_target, respect_limits=False)
        elapsed = time.perf_counter() - t0

        if raw_sols:
            n_raw_ok += 1
        if default_sols:
            n_default_ok += 1

        print(
            f"  {i + 1}/{N_RANDOM_CONFIGS}: default={'found' if default_sols else 'EMPTY'}, "
            f"raw={'found' if raw_sols else 'EMPTY'} ({elapsed:.3f}s)",
            flush=True,
        )

    print(
        f"\n  Summary: {n_default_ok}/{N_RANDOM_CONFIGS} default-policy solved, "
        f"{n_raw_ok}/{N_RANDOM_CONFIGS} raw (limits-ignored) solved"
    )
    return n_default_ok, n_raw_ok


def check_seeded_at_truth(arm, pin, model, data, q_indices, lo, hi):
    print(f"\n--- Check 3: seeded-at-truth over {N_RANDOM_CONFIGS} random reachable configs ---")
    rng = _rng()
    n_empty = 0
    n_valid = 0
    n_near_seed = 0
    worst_residual_when_valid = 0.0

    for i in range(N_RANDOM_CONFIGS):
        q_true_full = rng.uniform(lo, hi)
        T_target = pin_fk(pin, model, data, arm.BASE_LINK, arm.EE_LINK, q_true_full)
        q_true_local = q_true_full[q_indices]

        t0 = time.perf_counter()
        sols = arm.solve(T_target, max_solutions=1, q_seed=q_true_local, respect_limits=False)
        elapsed = time.perf_counter() - t0

        if not sols:
            n_empty += 1
            print(f"  {i + 1}/{N_RANDOM_CONFIGS}: EMPTY ({elapsed:.3f}s)", flush=True)
            continue

        q_solved = np.asarray(sols[0].q)
        joint_dist = np.linalg.norm(np.mod(q_solved - q_true_local + np.pi, 2 * np.pi) - np.pi)
        full_q_check = embed_arm_q(q_true_full, q_indices, q_solved)
        T_check = pin_fk(pin, model, data, arm.BASE_LINK, arm.EE_LINK, full_q_check)
        residual = _fk_residual(T_check, T_target)

        is_valid = residual < FK_ATOL_ROT
        is_near_seed = joint_dist < JOINT_DIST_NEAR_SEED
        print(
            f"  {i + 1}/{N_RANDOM_CONFIGS}: found, residual={residual:.2e} "
            f"({'valid' if is_valid else 'INVALID'}), joint_dist={joint_dist:.3f} rad "
            f"({'near seed' if is_near_seed else 'DIFFERENT BRANCH'}) ({elapsed:.3f}s)",
            flush=True,
        )
        if is_valid:
            n_valid += 1
            worst_residual_when_valid = max(worst_residual_when_valid, residual)
            if is_near_seed:
                n_near_seed += 1

    print(
        f"\n  Summary: {n_empty}/{N_RANDOM_CONFIGS} empty, "
        f"{n_valid}/{N_RANDOM_CONFIGS} kinematically valid "
        f"(worst residual when valid: {worst_residual_when_valid:.2e}), "
        f"{n_near_seed}/{N_RANDOM_CONFIGS} of those also close to the seed branch"
    )
    return n_empty, n_valid, n_near_seed


def main():
    arm = load_ssik_franka()
    print(f"franka_panda_ik: BASE_LINK={arm.BASE_LINK!r} EE_LINK={arm.EE_LINK!r} DOF={arm.DOF}")

    pin, model, data, urdf_path = load_pinocchio_model_for_panda()
    check_frames_and_joints(model, arm.BASE_LINK, arm.EE_LINK)
    q_indices = arm_q_indices(model)

    lo_full = np.asarray(model.lowerPositionLimit)
    hi_full = np.asarray(model.upperPositionLimit)

    home_ok = check_home_pose(arm, pin, model, data, q_indices)
    n_default_ok, n_raw_ok = check_round_trip(arm, pin, model, data, q_indices, lo_full, hi_full)
    n_empty, n_valid, n_near_seed = check_seeded_at_truth(
        arm, pin, model, data, q_indices, lo_full, hi_full
    )

    print(f"\n{'=' * 70}\nFINAL COMPARISON vs OpenArm\n{'=' * 70}")
    print(f"T_HOME matches Pinocchio:            {'YES' if home_ok else 'NO'}")
    print(f"Unseeded default-policy pass rate:   {n_default_ok}/{N_RANDOM_CONFIGS}")
    print(f"Unseeded raw (no limits) pass rate:  {n_raw_ok}/{N_RANDOM_CONFIGS}")
    print(f"Seeded-at-truth kinematic validity:  {n_valid}/{N_RANDOM_CONFIGS}")
    print(f"Seeded-at-truth near-seed branch:    {n_near_seed}/{N_RANDOM_CONFIGS}")
    print(
        "\nIf these rates are dramatically higher than OpenArm's "
        "(10/50, 5/50 there), the test methodology is sound and OpenArm's "
        "low rate is real and arm-specific. If Franka ALSO fails broadly "
        "here, the shared test code has a bug and none of the OpenArm "
        "conclusions should be trusted yet."
    )


if __name__ == "__main__":
    main()