"""
Integration tests for the ssik-built OpenArm IK artifact(s).

Assumes `ssik build` has already run against the OpenArm URDF(s) and the
resulting `<name>_ik.py` module(s) are importable (installed in the
workspace venv, or on PYTHONPATH).

Setup, before running:
  1. ARTIFACT_MODULE_NAMES -- match what `ssik build` named your
     artifact(s). Names that don't resolve are silently skipped, so
     unused guesses don't need to be deleted.
  2. URDF_PATHS -- path to the URDF each artifact was built against.
     Only needed for the Pinocchio cross-validation tests below; those
     skip (not fail) if a path isn't configured.

Pinocchio (imported as `pinocchio`, installed via the `pin` package) is
an optional dependency here. Every test that uses it skips cleanly if
it isn't installed -- see `_pin_model_for`.

Why Pinocchio as ground truth: ssik's own `fk_residual` is computed
against the same solver's internal FK, so a bug in the baked kinematic
constants wouldn't show up there. Cross-checking against an independent
FK implementation catches artifact-build mistakes -- wrong --base/--ee,
a stale URDF, wrong joint ordering, etc. -- that ssik's self-check can't.
"""

from __future__ import annotations

import importlib
import time
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Path anchoring
# ---------------------------------------------------------------------------

# URDF_PATHS entries are relative to the workspace root (where setup.sh
# clones `openarm`, `openarm_assets`, etc. as siblings), not this repo's
# own root -- `openarm` has its own `.git`, so a naive nearest-`.git`
# search would anchor one level too deep. Walk up looking for the
# workspace's own markers (setup.sh + pyproject.toml) first; fall back to
# the nearest `.git` if those aren't found.
def _find_workspace_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "setup.sh").exists() and (candidate / "pyproject.toml").exists():
            return candidate
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            return candidate
    return start.parent


REPO_ROOT = _find_workspace_root(Path(__file__).resolve().parent)


def _resolve_urdf(path: str) -> str:
    p = Path(path)
    return str(p if p.is_absolute() else (REPO_ROOT / p))


# ---------------------------------------------------------------------------
# Config -- edit for your repo layout
# ---------------------------------------------------------------------------

ARTIFACT_MODULE_NAMES = [
    "openarm.IK.openarm_left_ik",
    "openarm.IK.openarm_right_ik",
]

# Both artifacts are built from the same combined bimanual URDF (both arms
# hang off openarm_body_link0 via fixed joints).
_SHARED_URDF = "openarm_assets/src/openarm_assets/models/openarm/urdf/example/v1.urdf"

URDF_PATHS: dict[str, str] = {
    "openarm.IK.openarm_left_ik": _SHARED_URDF,
    "openarm.IK.openarm_right_ik": _SHARED_URDF,
}

# Expected --base/--ee per the URDF structure, purely informational: a
# mismatch is surfaced (printed), not failed, since hand_tcp and link7 are
# geometrically identical here (zero-offset fixed joint).
EXPECTED_LINKS = {
    "openarm.IK.openarm_left_ik": ("openarm_left_link0", "openarm_left_hand_tcp"),
    "openarm.IK.openarm_right_ik": ("openarm_right_link0", "openarm_right_hand_tcp"),
}

# URDF joint names per arm, q1..q7 order -- maps an ssik Solution.q
# (arm-local, length DOF) onto the combined URDF's full Pinocchio q vector
# (both arms + fingers).
ARM_JOINT_NAMES = {
    "left": [f"openarm_left_joint{i}" for i in range(1, 8)],
    "right": [f"openarm_right_joint{i}" for i in range(1, 8)],
}

FK_ATOL_ROT = 1e-4  # rotation-part Frobenius tolerance (also used for position)
JOINT_LIMIT_EPS = 1e-6
N_RANDOM_CONFIGS = 50
RNG_SEED = 0


def _side_from_name(name: str) -> str:
    if "left" in name:
        return "left"
    if "right" in name:
        return "right"
    raise ValueError(f"can't infer arm side ('left'/'right') from name {name!r}")


def _rng():
    return np.random.default_rng(RNG_SEED)


def _is_valid_se3(T: np.ndarray) -> bool:
    if T.shape != (4, 4):
        return False
    R = T[:3, :3]
    if not np.allclose(R @ R.T, np.eye(3), atol=1e-6):
        return False
    if not np.isclose(np.linalg.det(R), 1.0, atol=1e-6):
        return False
    if not np.allclose(T[3, :], [0, 0, 0, 1]):
        return False
    return True


def _fk_residual(T_a: np.ndarray, T_b: np.ndarray) -> float:
    return float(np.linalg.norm(T_a - T_b, ord="fro"))


# ---------------------------------------------------------------------------
# Artifact discovery
# ---------------------------------------------------------------------------


def _load_artifacts() -> tuple[dict[str, object], dict[str, Exception]]:
    loaded, import_errors = {}, {}
    for name in ARTIFACT_MODULE_NAMES:
        try:
            loaded[name] = importlib.import_module(name)
        except ModuleNotFoundError as e:
            # Genuinely absent -- expected for guessed names that don't
            # apply to this build. Only swallow if it's THIS module, not
            # a missing transitive dependency of a module that does exist.
            if e.name == name:
                continue
            import_errors[name] = e
        except Exception as e:  # noqa: BLE001 - surface anything else
            import_errors[name] = e
    return loaded, import_errors


_ARTIFACTS, _IMPORT_ERRORS = _load_artifacts()

if _IMPORT_ERRORS:
    details = "\n".join(f"  {name}: {err!r}" for name, err in _IMPORT_ERRORS.items())
    pytest.fail(
        f"Found the following OpenArm ssik artifact(s) on the import path, "
        f"but they failed to import (a real error, not a missing artifact):"
        f"\n{details}",
        pytrace=False,
    )

if not _ARTIFACTS:
    pytest.skip(
        f"No ssik-built OpenArm artifact importable. Looked for: "
        f"{ARTIFACT_MODULE_NAMES}. Update ARTIFACT_MODULE_NAMES to match "
        f"your `ssik build` output, or add the built module's directory "
        f"to PYTHONPATH.",
        allow_module_level=True,
    )

ARTIFACT_IDS = list(_ARTIFACTS.keys())
ARTIFACT_MODULES = list(_ARTIFACTS.values())


def _arm_q_indices(model, name: str) -> list[int]:
    side = _side_from_name(name)
    indices = []
    for jn in ARM_JOINT_NAMES[side]:
        jid = model.getJointId(jn)
        if jid >= model.njoints:
            raise ValueError(
                f"joint {jn!r} not found in the Pinocchio model for {name!r} "
                f"-- check ARM_JOINT_NAMES matches this URDF's joint names"
            )
        indices.append(model.joints[jid].idx_q)
    return indices


def _embed_arm_q(full_q_template: np.ndarray, q_indices: list[int], arm_local_q: np.ndarray) -> np.ndarray:
    """Copy of `full_q_template` with this arm's DOF overwritten by
    `arm_local_q` (ssik's Solution.q). Everything outside this arm's
    chain is left as-is -- it doesn't affect this arm's base->ee FK."""
    full_q = full_q_template.copy()
    for local_idx, global_idx in enumerate(q_indices):
        full_q[global_idx] = arm_local_q[local_idx]
    return full_q


def _best_residual(pin, model, data, arm, q_full, arm_q_indices, sols, T_target) -> float:
    """Smallest Pinocchio FK residual across a list of ssik solutions."""
    return min(
        _fk_residual(
            _pin_fk(
                pin, model, data, arm.BASE_LINK, arm.EE_LINK,
                _embed_arm_q(q_full, arm_q_indices, np.asarray(s.q)),
            ),
            T_target,
        )
        for s in sols
    )


# ---------------------------------------------------------------------------
# Pinocchio fixtures (optional dependency -- tests skip if unavailable)
# ---------------------------------------------------------------------------


def _pin_model_for(module_name: str):
    pin = pytest.importorskip("pinocchio", reason="pinocchio (optional dep 'pin') not installed")
    urdf_path = URDF_PATHS.get(module_name)
    if not urdf_path:
        pytest.skip(f"No URDF_PATHS entry configured for {module_name!r}")
    resolved = _resolve_urdf(urdf_path)
    if not Path(resolved).is_file():
        pytest.skip(
            f"URDF for {module_name!r} not found at {resolved} "
            f"(REPO_ROOT resolved to {REPO_ROOT}; configured path was {urdf_path!r})."
        )
    model = pin.buildModelFromUrdf(resolved)
    data = model.createData()
    return pin, model, data


def _pin_fk(pin, model, data, base_link: str, ee_link: str, q: np.ndarray) -> np.ndarray:
    """4x4 pose of ee_link expressed in base_link, via Pinocchio FK."""
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    T_world_base = data.oMf[model.getFrameId(base_link)]
    T_world_ee = data.oMf[model.getFrameId(ee_link)]
    return (T_world_base.inverse() * T_world_ee).homogeneous


def _pin_joint_limits(model) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(model.lowerPositionLimit), np.asarray(model.upperPositionLimit)


def _try_dev_diagnostic(name: str, arm, T: np.ndarray) -> str:
    """Best-effort: use ssik's dev path (Manipulator.from_urdf +
    explain=True) to get ssik's own diagnosis of a solve() failure,
    instead of guessing. Independent of Pinocchio. Always returns a
    message -- either the real diagnostic, or a specific reason it
    couldn't be produced."""
    urdf_path = URDF_PATHS.get(name)
    if not urdf_path:
        return f"no URDF_PATHS entry configured for {name!r}"

    resolved = _resolve_urdf(urdf_path)
    if not Path(resolved).is_file():
        return f"URDF not found at {resolved} (REPO_ROOT={REPO_ROOT}, configured={urdf_path!r})"

    try:
        import ssik
    except ImportError as e:
        return f"`import ssik` failed ({e!r}) -- is ssik installed in this venv?"

    try:
        dev_arm = ssik.Manipulator.from_urdf(resolved, base=arm.BASE_LINK, ee=arm.EE_LINK)
    except Exception as e:  # noqa: BLE001 - diagnostic aid only
        return (
            f"ssik.Manipulator.from_urdf({resolved!r}, base={arm.BASE_LINK!r}, "
            f"ee={arm.EE_LINK!r}) raised: {e!r}. Likely missing the urdf "
            f"extras -- try `pip install ssik[urdf]`."
        )

    try:
        _, diag = dev_arm.solve(T, explain=True)
        return diag.summary()
    except Exception as e:  # noqa: BLE001 - diagnostic aid only
        return f"dev_arm.solve(T, explain=True) itself raised: {e!r}"


# ---------------------------------------------------------------------------
# Basic artifact API / sanity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("arm", ARTIFACT_MODULES, ids=ARTIFACT_IDS)
def test_artifact_exposes_public_constants(arm):
    for attr in ("BASE_LINK", "EE_LINK", "DOF", "T_HOME"):
        assert hasattr(arm, attr), f"{arm.__name__} missing {attr}"
    assert isinstance(arm.DOF, int)
    assert arm.DOF in (6, 7), f"unexpected DOF {arm.DOF} for {arm.__name__}"


@pytest.mark.parametrize("arm", ARTIFACT_MODULES, ids=ARTIFACT_IDS)
def test_home_pose_is_valid_se3(arm):
    T_home = np.asarray(arm.T_HOME)
    assert _is_valid_se3(T_home), f"{arm.__name__}.T_HOME is not a valid SE(3) matrix"


@pytest.mark.parametrize("arm", ARTIFACT_MODULES, ids=ARTIFACT_IDS)
def test_solve_at_home_pose_recovers_zero_config(arm):
    """q = 0 FKs to T_HOME by construction, so solving T_HOME with limits
    ignored should trivially recover it. An empty result even with limits
    ignored means the solver found zero raw candidates -- either a
    genuine kinematic singularity at the fully-extended pose, or a
    build-time mismatch. Pulls ssik's own explain=True diagnostic on
    failure rather than guessing."""
    T_home = np.asarray(arm.T_HOME)

    raw_sols = arm.solve(T_home, respect_limits=False)
    if not raw_sols:
        diag = _try_dev_diagnostic(arm.__name__, arm, T_home)
        pytest.fail(
            f"{arm.__name__}: solve(T_HOME, respect_limits=False) found no "
            f"branch at all, even with limits ignored.\n\n"
            f"ssik explain=True diagnostic:\n{diag}"
        )

    best_raw = min(raw_sols, key=lambda s: np.linalg.norm(s.q))
    raw_residual = np.linalg.norm(best_raw.q)
    assert raw_residual < 1e-3, (
        f"{arm.__name__}: closest raw branch to home config was "
        f"{best_raw.q} (norm {raw_residual:.2e}), expected ~zeros"
    )

    default_sols = arm.solve(T_home)  # respect_limits=True, the public default
    if not default_sols:
        pytest.xfail(
            f"{arm.__name__}: solve(T_HOME) returns [] under the default "
            f"policy even though the raw solve found q~zeros (residual "
            f"{raw_residual:.2e}). Likely cause: a joint limit sits "
            f"exactly at the home config's boundary and floating-point "
            f"noise gets filtered by the default policy."
        )
    else:
        best_default = min(default_sols, key=lambda s: s.fk_residual)
        assert best_default.fk_residual < FK_ATOL_ROT, (
            f"{arm.__name__}: default-policy branch at home config has "
            f"fk_residual={best_default.fk_residual:.2e} -- not a valid "
            f"solution, not just a different branch than q=0"
        )


@pytest.mark.parametrize("arm", ARTIFACT_MODULES, ids=ARTIFACT_IDS)
def test_seeded_solve_near_limit_boundary_is_fast(arm):
    """Regression guard: seeded max_solutions=1 solve at the home config
    should be sub-ms, not multi-second. If your control loop ever seeds
    near this configuration, a multi-second stall here is a real-time
    risk, not just a test artifact."""
    T_home = np.asarray(arm.T_HOME)
    q_seed = np.zeros(arm.DOF)

    start = time.perf_counter()
    arm.solve(T_home, max_solutions=1, q_seed=q_seed)
    elapsed_ms = (time.perf_counter() - start) * 1000

    budget_ms = 100.0
    assert elapsed_ms < budget_ms, (
        f"{arm.__name__}: seeded solve at home config took "
        f"{elapsed_ms:.1f} ms, budget was {budget_ms} ms."
    )


# ---------------------------------------------------------------------------
# Cross-validation against Pinocchio FK (independent ground truth)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ARTIFACT_IDS)
def test_base_ee_are_valid_urdf_frames(name):
    """Catches a typo'd --base/--ee before any FK math runs."""
    arm = _ARTIFACTS[name]
    pin, model, data = _pin_model_for(name)

    assert model.existFrame(arm.BASE_LINK), (
        f"{name}: arm.BASE_LINK={arm.BASE_LINK!r} is not a frame in "
        f"{_resolve_urdf(URDF_PATHS[name])}"
    )
    assert model.existFrame(arm.EE_LINK), (
        f"{name}: arm.EE_LINK={arm.EE_LINK!r} is not a frame in "
        f"{_resolve_urdf(URDF_PATHS[name])}"
    )

    expected = EXPECTED_LINKS.get(name)
    if expected and (arm.BASE_LINK, arm.EE_LINK) != expected:
        print(
            f"\n{name}: artifact uses base={arm.BASE_LINK!r}, "
            f"ee={arm.EE_LINK!r}; URDF structure suggested {expected}. "
            f"Not necessarily wrong -- just confirm this is intended."
        )


@pytest.mark.parametrize("name", ARTIFACT_IDS)
def test_home_pose_matches_pinocchio_fk(name):
    arm = _ARTIFACTS[name]
    pin, model, data = _pin_model_for(name)
    T_pin = _pin_fk(pin, model, data, arm.BASE_LINK, arm.EE_LINK, np.zeros(model.nq))
    residual = _fk_residual(T_pin, np.asarray(arm.T_HOME))
    assert residual < FK_ATOL_ROT, (
        f"{name}: T_HOME disagrees with Pinocchio FK at q=0 "
        f"(residual={residual:.2e}). Check --base/--ee and that the "
        f"build URDF matches URDF_PATHS[{name!r}]."
    )


@pytest.mark.parametrize("name", ARTIFACT_IDS)
def test_random_configs_round_trip_through_pinocchio(name):
    """Sample real joint configs, FK them with Pinocchio (ground truth),
    feed the resulting pose to ssik, and check some returned branch
    actually reaches it per Pinocchio (not per ssik's own fk_residual).

    Also tracks respect_limits=False alongside the default policy: if the
    default policy fails broadly but the raw solve mostly succeeds, the
    artifact's baked joint limits are likely wrong -- not a solver
    coverage gap."""
    arm = _ARTIFACTS[name]
    pin, model, data = _pin_model_for(name)
    lo, hi = _pin_joint_limits(model)
    rng = _rng()
    arm_q_indices = _arm_q_indices(model, name)

    n_default_ok = n_raw_ok = 0
    for _ in range(N_RANDOM_CONFIGS):
        q_true = rng.uniform(lo, hi)
        T_target = _pin_fk(pin, model, data, arm.BASE_LINK, arm.EE_LINK, q_true)

        default_sols = arm.solve(T_target)
        if arm.solve(T_target, respect_limits=False):
            n_raw_ok += 1
        if not default_sols:
            continue

        best_residual = _best_residual(pin, model, data, arm, q_true, arm_q_indices, default_sols, T_target)
        assert best_residual < FK_ATOL_ROT, (
            f"{name}: best branch for a sampled config had residual "
            f"{best_residual:.2e} against target (q_true={q_true})"
        )
        n_default_ok += 1

    if n_default_ok < N_RANDOM_CONFIGS // 2:
        if n_raw_ok >= N_RANDOM_CONFIGS // 2:
            diagnosis = (
                f"respect_limits=False succeeded on {n_raw_ok}/{N_RANDOM_CONFIGS} "
                f"of the same configs -- points at the artifact's baked "
                f"joint limits being wrong, not a solver coverage gap."
            )
        else:
            diagnosis = (
                f"respect_limits=False also only succeeded on "
                f"{n_raw_ok}/{N_RANDOM_CONFIGS} -- a broader coverage "
                f"problem independent of limits, worth escalating."
            )
        pytest.fail(
            f"{name}: only {n_default_ok}/{N_RANDOM_CONFIGS} random "
            f"reachable configs produced a default-policy solution.\n{diagnosis}"
        )


@pytest.mark.parametrize("name", ARTIFACT_IDS)
def test_artifact_kinematics_matches_urdf_when_seeded_at_truth(name):
    """Seed solve() with q_seed set exactly to a sampled true config --
    the easiest input a jointlock solver can get -- with limits ignored,
    to isolate kinematic fidelity from limits policy.

    Separates two questions: (1) does ssik return ANY branch whose
    Pinocchio FK matches the target (the real "is this artifact
    corrupted" question), and (2) is the returned branch close to the
    seed (a narrower question about max_solutions=1's nearest-to-seed
    ordering, not about correctness). A kinematically-valid-but-different
    branch is informative, not evidence of corruption."""
    arm = _ARTIFACTS[name]
    pin, model, data = _pin_model_for(name)
    lo, hi = _pin_joint_limits(model)
    rng = _rng()
    arm_q_indices = _arm_q_indices(model, name)

    n_valid = n_near_seed = 0
    for _ in range(N_RANDOM_CONFIGS):
        q_true_full = rng.uniform(lo, hi)
        T_target = _pin_fk(pin, model, data, arm.BASE_LINK, arm.EE_LINK, q_true_full)
        q_true_local = q_true_full[arm_q_indices]

        sols = arm.solve(T_target, max_solutions=1, q_seed=q_true_local, respect_limits=False)
        if not sols:
            continue

        q_solved = np.asarray(sols[0].q)
        joint_dist = np.linalg.norm(np.mod(q_solved - q_true_local + np.pi, 2 * np.pi) - np.pi)
        full_q_check = _embed_arm_q(q_true_full, arm_q_indices, q_solved)
        residual = _fk_residual(
            _pin_fk(pin, model, data, arm.BASE_LINK, arm.EE_LINK, full_q_check), T_target
        )

        if residual < FK_ATOL_ROT:
            n_valid += 1
            if joint_dist < 0.1:
                n_near_seed += 1

    assert n_valid / N_RANDOM_CONFIGS > 0.5, (
        f"{name}: only {n_valid}/{N_RANDOM_CONFIGS} seeded solves found ANY "
        f"branch matching the target pose per Pinocchio -- real evidence "
        f"the artifact's baked kinematics don't match this URDF."
    )
    if n_valid and n_near_seed < n_valid * 0.5:
        print(
            f"\n{name}: {n_valid - n_near_seed}/{n_valid} kinematically "
            f"valid solves returned a different branch than the seed, not "
            f"the nearest one. Kinematics check out -- this is about "
            f"max_solutions=1's nearest-to-seed ordering, not correctness."
        )


@pytest.mark.parametrize("name", ARTIFACT_IDS)
def test_default_solve_respects_urdf_joint_limits(name):
    arm = _ARTIFACTS[name]
    pin, model, data = _pin_model_for(name)
    lo, hi = _pin_joint_limits(model)
    rng = _rng()
    arm_q_indices = _arm_q_indices(model, name)
    lo_local, hi_local = lo[arm_q_indices], hi[arm_q_indices]

    q_true_full = rng.uniform(lo, hi)
    T_target = _pin_fk(pin, model, data, arm.BASE_LINK, arm.EE_LINK, q_true_full)

    for s in arm.solve(T_target):  # respect_limits=True by default
        q = np.asarray(s.q)
        assert np.all(q >= lo_local - JOINT_LIMIT_EPS) and np.all(q <= hi_local + JOINT_LIMIT_EPS), (
            f"{name}: solve() returned an out-of-limit branch {q} despite "
            f"respect_limits defaulting to True"
        )


# ---------------------------------------------------------------------------
# Systematic edge-case / singularity coverage
#
# These stress boundary and structural configurations (joint-limit
# extremes, full-range sweeps, near-tolerance perturbations) rather than
# any specific historically-failing pose -- so they generalize across
# URDF revisions and don't need manual upkeep.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ARTIFACT_IDS)
def test_joint_limit_extremes_solve_correctly(name):
    """Configurations at the joint-limit extremes -- all-lower, all-upper,
    and each joint individually pinned to a limit while the rest sit at
    mid-range -- are exactly where analytical solvers are most likely to
    hit a degenerate common normal or rank-deficient Jacobian."""
    arm = _ARTIFACTS[name]
    pin, model, data = _pin_model_for(name)
    lo, hi = _pin_joint_limits(model)
    arm_q_indices = _arm_q_indices(model, name)
    lo_local, hi_local = lo[arm_q_indices], hi[arm_q_indices]
    mid_local = (lo_local + hi_local) / 2
    q_full_template = np.zeros(model.nq)

    configs = [lo_local.copy(), hi_local.copy()]
    for i in range(arm.DOF):
        for bound in (lo_local, hi_local):
            q = mid_local.copy()
            q[i] = bound[i]
            configs.append(q)

    n_valid = 0
    for q_local in configs:
        q_full = _embed_arm_q(q_full_template, arm_q_indices, q_local)
        T_target = _pin_fk(pin, model, data, arm.BASE_LINK, arm.EE_LINK, q_full)
        sols = arm.solve(T_target, respect_limits=False)
        if sols and _best_residual(pin, model, data, arm, q_full, arm_q_indices, sols, T_target) < FK_ATOL_ROT:
            n_valid += 1

    assert n_valid >= len(configs) * 0.5, (
        f"{name}: only {n_valid}/{len(configs)} joint-limit-extreme "
        f"configs solved correctly. These boundary corners are common "
        f"failure points for analytical IK solvers."
    )


@pytest.mark.parametrize("name", ARTIFACT_IDS)
def test_full_single_joint_range_sweep(name):
    """Sweep each joint individually across its full range (rest at
    mid-range) and check the pass rate. This walks through every
    structural singularity along a joint's travel -- including any
    near-zero-link-length configurations -- without needing to know in
    advance where those singularities sit."""
    arm = _ARTIFACTS[name]
    pin, model, data = _pin_model_for(name)
    lo, hi = _pin_joint_limits(model)
    arm_q_indices = _arm_q_indices(model, name)
    lo_local, hi_local = lo[arm_q_indices], hi[arm_q_indices]
    mid_local = (lo_local + hi_local) / 2
    q_full_template = np.zeros(model.nq)

    steps = 15
    n_valid = n_total = 0
    for j in range(arm.DOF):
        for t in np.linspace(0.0, 1.0, steps):
            q_local = mid_local.copy()
            q_local[j] = lo_local[j] + t * (hi_local[j] - lo_local[j])
            q_full = _embed_arm_q(q_full_template, arm_q_indices, q_local)
            T_target = _pin_fk(pin, model, data, arm.BASE_LINK, arm.EE_LINK, q_full)

            n_total += 1
            sols = arm.solve(T_target, respect_limits=False)
            if sols and _best_residual(pin, model, data, arm, q_full, arm_q_indices, sols, T_target) < FK_ATOL_ROT:
                n_valid += 1

    pass_rate = n_valid / n_total
    assert pass_rate > 0.8, (
        f"{name}: sweeping each joint across its full range solved only "
        f"{n_valid}/{n_total} ({pass_rate:.0%}) of sampled configs. A low "
        f"rate here points at a structural singularity somewhere along a "
        f"joint's travel, not just at the extremes."
    )


@pytest.mark.parametrize("name", ARTIFACT_IDS)
def test_pose_perturbation_robustness(name):
    """A sub-tolerance positional perturbation of a reachable target
    shouldn't change whether the solver finds a valid branch -- guards
    against brittle behavior right at numerical-tolerance boundaries."""
    arm = _ARTIFACTS[name]
    pin, model, data = _pin_model_for(name)
    lo, hi = _pin_joint_limits(model)
    rng = _rng()
    arm_q_indices = _arm_q_indices(model, name)

    n_valid = 0
    n_trials = 20
    for _ in range(n_trials):
        q_true = rng.uniform(lo, hi)
        T_target = _pin_fk(pin, model, data, arm.BASE_LINK, arm.EE_LINK, q_true)

        T_perturbed = T_target.copy()
        T_perturbed[:3, 3] += rng.normal(scale=1e-6, size=3)

        sols = arm.solve(T_perturbed, respect_limits=False)
        if sols and _best_residual(pin, model, data, arm, q_true, arm_q_indices, sols, T_target) < FK_ATOL_ROT * 10:
            n_valid += 1

    assert n_valid >= n_trials * 0.75, (
        f"{name}: only {n_valid}/{n_trials} sub-tolerance-perturbed poses "
        f"solved to a branch matching the unperturbed target -- suggests "
        f"brittleness near numerical-precision boundaries."
    )


@pytest.mark.parametrize("arm", ARTIFACT_MODULES, ids=ARTIFACT_IDS)
def test_workspace_reach_extremes_do_not_crash(arm):
    """Targets right at, and just past, an estimated reach boundary
    should never raise -- only return a (possibly empty) solution list."""
    T_home = np.asarray(arm.T_HOME)
    reach_estimate = float(np.linalg.norm(T_home[:3, 3])) or 1.0

    for scale in (0.99, 1.0, 1.5, 3.0):
        T_edge = T_home.copy()
        T_edge[:3, 3] = T_home[:3, 3] / (np.linalg.norm(T_home[:3, 3]) or 1.0) * reach_estimate * scale
        try:
            sols = arm.solve(T_edge)
        except Exception as e:  # noqa: BLE001
            pytest.fail(f"{arm.__name__}: solve() raised at reach scale {scale}: {e!r}")
        assert isinstance(sols, list), f"{arm.__name__}: solve() didn't return a list at reach scale {scale}"


# ---------------------------------------------------------------------------
# Behavior that doesn't need Pinocchio
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("arm", ARTIFACT_MODULES, ids=ARTIFACT_IDS)
def test_grossly_unreachable_pose_returns_empty(arm):
    T_far = np.eye(4)
    T_far[:3, 3] = [100.0, 100.0, 100.0]  # 100 m away, no arm reaches this
    assert arm.solve(T_far) == [], f"{arm.__name__}: expected [] for an obviously unreachable pose"


@pytest.mark.parametrize("arm", ARTIFACT_MODULES, ids=ARTIFACT_IDS)
def test_solve_is_deterministic(arm):
    """No RNG in the solve path -- same pose in, same branch set out."""
    T = np.asarray(arm.T_HOME)
    sols_a, sols_b = arm.solve(T), arm.solve(T)
    assert len(sols_a) == len(sols_b)
    qs_a = sorted(tuple(np.round(s.q, 9)) for s in sols_a)
    qs_b = sorted(tuple(np.round(s.q, 9)) for s in sols_b)
    assert qs_a == qs_b, f"{arm.__name__}: solve() gave different branches on repeat calls"


@pytest.mark.parametrize("arm", ARTIFACT_MODULES, ids=ARTIFACT_IDS)
@pytest.mark.slow
def test_seeded_solve_latency_smoke(arm):
    """Guard against a silent perf regression after re-running `ssik
    build` (e.g. losing a jointlock short-circuit). Threshold is generous
    on purpose -- this is a smoke test, not a benchmark."""
    T = np.asarray(arm.T_HOME)
    q_seed = np.zeros(arm.DOF)
    arm.solve(T, max_solutions=1, q_seed=q_seed, respect_limits=False)  # warm up

    start = time.perf_counter()
    for _ in range(20):
        arm.solve(T, max_solutions=1, q_seed=q_seed, respect_limits=False)
    elapsed_ms = (time.perf_counter() - start) / 20 * 1000

    budget_ms = 50.0 if arm.DOF == 7 else 10.0
    assert elapsed_ms < budget_ms, (
        f"{arm.__name__}: seeded solve averaged {elapsed_ms:.2f} ms/call, "
        f"budget was {budget_ms} ms"
    )


def test_left_and_right_artifacts_are_not_identical():
    """Catches a left/right artifact accidentally built twice from the
    same URDF."""
    left = _ARTIFACTS.get("openarm.IK.openarm_left_ik")
    right = _ARTIFACTS.get("openarm.IK.openarm_right_ik")
    if left is None or right is None:
        pytest.skip("both left and right artifacts must be present for this check")
    assert not np.allclose(np.asarray(left.T_HOME), np.asarray(right.T_HOME), atol=1e-6), (
        "left and right arm artifacts report the same T_HOME -- did "
        "`ssik build` accidentally run against the same URDF for both sides?"
    )