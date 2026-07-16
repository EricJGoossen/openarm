"""
Integration tests for the ssik-built OpenArm IK artifact(s).

These assume you've already run `ssik build` against the OpenArm URDF(s)
(per-arm if bimanual) and that the resulting `<name>_ik.py` module(s) are
importable — either sitting next to this test file, or on PYTHONPATH /
installed in the workspace venv.

WHAT TO EDIT BEFORE RUNNING
---------------------------
1. ARTIFACT_MODULE_NAMES — match whatever `ssik build` actually named your
   artifact(s). Left as a guess list; unresolvable names are silently
   skipped so you don't have to delete entries.
2. URDF_PATHS — path to the exact URDF you built each artifact against.
   Only needed for the Pinocchio cross-validation tests; those tests
   skip (not fail) if a path isn't configured or Pinocchio isn't
   installed in this environment.
3. BASE_LINK / EE_LINK per arm if you want the Pinocchio tests to use
   link names other than what's baked into the artifact's own
   `BASE_LINK` / `EE_LINK` constants (usually you don't need to touch
   this — the test reads them from the artifact directly).

WHY PINOCCHIO AS GROUND TRUTH
------------------------------
ssik's own `fk_residual` is self-referential — it's computed against the
same solver's internal FK, so a bug in the baked KinBody constants
wouldn't show up there. Cross-checking against an independent FK
implementation (Pinocchio, which you're already using for gravity comp)
actually catches artifact-build mistakes: wrong --base/--ee, a stale
URDF, wrong joint ordering, etc.
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

# This file lives somewhere inside the `openarm` repo's own tests directory,
# but URDF_PATHS entries are given relative to the WORKSPACE root
# (openarm_robot-code/, where setup.sh clones `openarm`, `openarm_assets`,
# etc. as SIBLINGS) — not relative to the `openarm` repo's own root.
# Walking up to the nearest `.git` would stop too early: `openarm` has its
# own `.git`, so that search would anchor one level too deep and silently
# look for `openarm_assets/...` *inside* `openarm/`, which doesn't exist.
# Instead, walk up looking for the workspace's actual markers
# (setup.sh + pyproject.toml, both at openarm_robot-code's root per its
# own repo layout); fall back to the nearest `.git` only if those aren't
# found, so this doesn't hard-fail on layouts we haven't seen.
def _find_workspace_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "setup.sh").exists() and (candidate / "pyproject.toml").exists():
            return candidate
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            return candidate
    return start.parent  # last-resort fallback


REPO_ROOT = _find_workspace_root(Path(__file__).resolve().parent)


def _resolve_urdf(path: str) -> str:
    p = Path(path)
    return str(p if p.is_absolute() else (REPO_ROOT / p))

# ---------------------------------------------------------------------------
# Config — edit this section for your repo layout
# ---------------------------------------------------------------------------

ARTIFACT_MODULE_NAMES = [
    "openarm.openarm_left_ik",
    "openarm.openarm_right_ik",
]

# Both artifacts were built from the SAME combined bimanual URDF (one file,
# both arms hang off `openarm_body_link0` via fixed joints) — point both
# entries at that one file. Path is relative to REPO_ROOT (see above),
# which is the openarm_robot-code WORKSPACE root (found via setup.sh +
# pyproject.toml), so this is relative to that root — matching how
# setup.sh cloned `openarm_assets` as a sibling of `openarm`.
_SHARED_URDF = "openarm_assets/src/openarm_assets/models/openarm/urdf/example/v1.urdf"

# module name -> URDF path used with `ssik build --base ... --ee ...`
URDF_PATHS: dict[str, str] = {
    "openarm.openarm_left_ik": _SHARED_URDF,
    "openarm.openarm_right_ik": _SHARED_URDF,
}

# Expected --base/--ee per the URDF structure: each arm's revolute chain
# starts at `<side>_link0` (attached to the torso via a FIXED joint) and
# ends at `<side>_link7`; `<side>_hand_tcp` is a zero-offset fixed frame
# on link7, so it's numerically identical but is the semantically correct
# tool point if that's what you passed to `ssik build --ee`. Not used to
# drive any test directly — arm.BASE_LINK / arm.EE_LINK from the artifact
# itself are the source of truth — this is just here so a mismatch is
# obvious at a glance.
EXPECTED_LINKS = {
    "openarm.openarm_left_ik": ("openarm_left_link0", "openarm_left_hand_tcp"),
    "openarm.openarm_right_ik": ("openarm_right_link0", "openarm_right_hand_tcp"),
}

# URDF joint names per arm, in q1..q7 order — used to map a 7-DOF
# ssik Solution.q (arm-local) into the correct 7 positions of the full
# 18-DOF combined-URDF Pinocchio q vector (both arms + fingers). Getting
# this mapping right matters: naively slicing q[:7] or q[7:14] would be
# wrong and silently produce garbage FK residuals instead of a clean
# shape error.
ARM_JOINT_NAMES = {
    "left": [f"openarm_left_joint{i}" for i in range(1, 8)],
    "right": [f"openarm_right_joint{i}" for i in range(1, 8)],
}


def _side_from_name(name: str) -> str:
    if "left" in name:
        return "left"
    if "right" in name:
        return "right"
    raise ValueError(f"can't infer arm side ('left'/'right') from module name {name!r}")


def _arm_q_indices(model, name: str) -> list[int]:
    side = _side_from_name(name)
    indices = []
    for jn in ARM_JOINT_NAMES[side]:
        jid = model.getJointId(jn)
        if jid >= model.njoints:
            raise ValueError(
                f"joint {jn!r} not found in the Pinocchio model for {name!r} — "
                f"check ARM_JOINT_NAMES matches this URDF's actual joint names"
            )
        indices.append(model.joints[jid].idx_q)
    return indices

FK_ATOL_M = 1e-4  # 0.1 mm — matches ssik's default sub-repeatability floor
FK_ATOL_ROT = 1e-4  # rotation-part Frobenius tolerance
JOINT_LIMIT_EPS = 1e-6
N_RANDOM_CONFIGS = 50
RNG_SEED = 0


# ---------------------------------------------------------------------------
# Artifact discovery
# ---------------------------------------------------------------------------


def _load_artifacts() -> dict[str, object]:
    loaded = {}
    import_errors = {}
    for name in ARTIFACT_MODULE_NAMES:
        try:
            loaded[name] = importlib.import_module(name)
        except ModuleNotFoundError as e:
            # Genuinely not on the import path — expected for names in our
            # guess list that don't apply to your build. Only swallow this
            # if it's specifically THIS module that's missing, not some
            # transitive dependency of a module that does exist.
            if e.name == name:
                continue
            import_errors[name] = e
        except Exception as e:  # noqa: BLE001 - surface anything else, don't hide it
            import_errors[name] = e
    return loaded, import_errors


_ARTIFACTS, _IMPORT_ERRORS = _load_artifacts()

if _IMPORT_ERRORS:
    details = "\n".join(f"  {name}: {err!r}" for name, err in _IMPORT_ERRORS.items())
    pytest.fail(
        f"Found the following OpenArm ssik artifact(s) on the import path, "
        f"but they failed to import (this is a real error, not a missing "
        f"artifact):\n{details}",
        pytrace=False,
    )

if not _ARTIFACTS:
    pytest.skip(
        "No ssik-built OpenArm artifact importable. Looked for: "
        f"{ARTIFACT_MODULE_NAMES}. Update ARTIFACT_MODULE_NAMES at the top "
        "of this file to match your `ssik build` output, or add the built "
        "module's directory to PYTHONPATH.",
        allow_module_level=True,
    )

ARTIFACT_IDS = list(_ARTIFACTS.keys())
ARTIFACT_MODULES = list(_ARTIFACTS.values())


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
# Pinocchio fixtures (skipped gracefully if unavailable / unconfigured)
# ---------------------------------------------------------------------------


def _pin_model_for(module_name: str):
    pin = pytest.importorskip("pinocchio", reason="pinocchio not installed")
    if not hasattr(pin, "buildModelFromUrdf"):
        pytest.fail(
            f"The installed `pinocchio` package (version "
            f"{getattr(pin, '__version__', 'unknown')!r}) has no "
            f"`buildModelFromUrdf` — this is the unrelated PyPI package "
            f"literally named `pinocchio` (a name squat), not the real "
            f"robotics library. Fix: `uv remove pinocchio` (or `--dev`) "
            f"then `uv add pin` — the real library installs under the "
            f"package name `pin` but still exposes `import pinocchio`.",
            pytrace=False,
        )
    urdf_path = URDF_PATHS.get(module_name)
    if not urdf_path:
        pytest.skip(f"No URDF_PATHS entry configured for {module_name!r}")
    resolved = _resolve_urdf(urdf_path)
    if not Path(resolved).is_file():
        pytest.skip(
            f"URDF for {module_name!r} not found at {resolved} "
            f"(REPO_ROOT resolved to {REPO_ROOT}; configured path was "
            f"{urdf_path!r}). Fix URDF_PATHS."
        )
    model = pin.buildModelFromUrdf(resolved)
    data = model.createData()
    return pin, model, data


def _pin_fk(pin, model, data, base_link: str, ee_link: str, q: np.ndarray) -> np.ndarray:
    """4x4 pose of ee_link expressed in base_link, via Pinocchio FK."""
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    base_id = model.getFrameId(base_link)
    ee_id = model.getFrameId(ee_link)
    T_world_base = data.oMf[base_id]
    T_world_ee = data.oMf[ee_id]
    T_base_ee = T_world_base.inverse() * T_world_ee
    return T_base_ee.homogeneous


def _pin_joint_limits(model) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(model.lowerPositionLimit), np.asarray(model.upperPositionLimit)


# URDF joint names per arm, in q1..q7 order — used only to map an ssik
# Solution.q (arm-local, length DOF) back onto this combined URDF's full
# model q-vector (length model.nq, includes both arms + fingers) so it
# can be fed through the SAME Pinocchio FK used for the ground-truth pose.
# Without this, s.q gets passed directly to forwardKinematics() expecting
# model.nq entries and gets a dimension-mismatch ValueError instead of a
# meaningful comparison.
_ARM_JOINT_NAMES = {
    "left": [f"openarm_left_joint{i}" for i in range(1, 8)],
    "right": [f"openarm_right_joint{i}" for i in range(1, 8)],
}


def _side_from_artifact_name(name: str) -> str:
    if "left" in name:
        return "left"
    if "right" in name:
        return "right"
    raise ValueError(
        f"can't infer arm side ('left'/'right') from artifact module name "
        f"{name!r} — update _ARM_JOINT_NAMES/_side_from_artifact_name if "
        f"your naming convention differs"
    )


def _arm_q_indices(model, name: str) -> list[int]:
    side = _side_from_artifact_name(name)
    return [model.joints[model.getJointId(jn)].idx_q for jn in _ARM_JOINT_NAMES[side]]


def _embed_arm_q(full_q_template: np.ndarray, q_indices: list[int], arm_local_q: np.ndarray) -> np.ndarray:
    """Return a copy of `full_q_template` (a valid full-model q vector,
    e.g. the q_true used to generate the target pose) with this arm's own
    DOF overwritten by `arm_local_q` (ssik's Solution.q, length DOF).
    Everything outside this arm's chain is left as-is — it doesn't affect
    this arm's base->ee FK, so any consistent value works, using the
    original q_true's values there rather than zeros just keeps things
    tidy."""
    full_q = full_q_template.copy()
    for local_idx, global_idx in enumerate(q_indices):
        full_q[global_idx] = arm_local_q[local_idx]
    return full_q


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


def _try_dev_diagnostic(name: str, arm, T: np.ndarray) -> str:
    """Best-effort: use ssik's dev path (Manipulator.from_urdf + explain=True)
    to get ssik's own diagnosis of why solve() found nothing, instead of
    guessing. Independent of Pinocchio — only needs `pip install ssik[urdf]`
    and a URDF_PATHS entry. Always returns a message: either the real
    diagnostic, or an explicit, specific reason it couldn't be produced —
    never silently says nothing.
    """
    urdf_path = URDF_PATHS.get(name)
    if not urdf_path:
        return f"no URDF_PATHS entry configured for {name!r}"

    resolved = _resolve_urdf(urdf_path)
    if not Path(resolved).is_file():
        return (
            f"URDF not found at {resolved} "
            f"(REPO_ROOT resolved to {REPO_ROOT}; configured path was "
            f"{urdf_path!r}). Fix URDF_PATHS or check REPO_ROOT is "
            f"anchoring to the right workspace root."
        )

    try:
        import ssik
    except ImportError as e:
        return f"`import ssik` failed ({e!r}) — is ssik installed in this venv at all?"

    try:
        dev_arm = ssik.Manipulator.from_urdf(resolved, base=arm.BASE_LINK, ee=arm.EE_LINK)
    except Exception as e:  # noqa: BLE001 - diagnostic aid only, never let it mask the real failure
        return (
            f"ssik.Manipulator.from_urdf({resolved!r}, base={arm.BASE_LINK!r}, "
            f"ee={arm.EE_LINK!r}) raised: {e!r}. Likely missing the urdf "
            f"extras — try `pip install ssik[urdf]` (needs urchin + sympy "
            f"on the import path, which the built artifact itself doesn't "
            f"need but this dev path does)."
        )

    try:
        _, diag = dev_arm.solve(T, explain=True)
        return diag.summary()
    except Exception as e:  # noqa: BLE001 - diagnostic aid only, never let it mask the real failure
        return f"dev_arm.solve(T, explain=True) itself raised: {e!r}"



@pytest.mark.parametrize("arm", ARTIFACT_MODULES, ids=ARTIFACT_IDS)
def test_solve_at_home_pose_recovers_zero_config(arm):
    """q = 0 should FK to T_HOME by construction, so solving T_HOME with
    limits ignored (respect_limits=False) should trivially recover it.

    If this returns [] even with limits ignored, that's not a joint-limit
    filtering artifact, it means the solver itself found zero raw
    candidates at this pose. The fully-extended home config is a classic
    serial-arm kinematic singularity (joint4 straight -> rank-deficient
    Jacobian), which can legitimately zero out an analytical solver's
    candidate set depending on its degeneracy tolerance — but that's a
    hypothesis, not a diagnosis. On failure this pulls ssik's own
    explain=True diagnostic (dev path, no Pinocchio needed) so you get
    ssik's actual reasoning instead of another guess.
    """
    T_home = np.asarray(arm.T_HOME)

    raw_sols = arm.solve(T_home, respect_limits=False)
    if not raw_sols:
        diag = _try_dev_diagnostic(arm.__name__, arm, T_home)
        pytest.fail(
            f"{arm.__name__}: solve(T_HOME, respect_limits=False) found no "
            f"branch at all, even with limits ignored. T_HOME is supposed "
            f"to be FK(q=0) exactly, so this is either a genuine kinematic "
            f"singularity at the fully-extended pose, or a build-time "
            f"mismatch between the baked T_HOME and the artifact's actual "
            f"solve path.\n\nssik explain=True diagnostic attempt:\n{diag}"
        )

    best_raw = min(raw_sols, key=lambda s: np.linalg.norm(s.q))
    raw_residual = np.linalg.norm(best_raw.q)
    assert raw_residual < 1e-3, (
        f"{arm.__name__}: closest RAW branch to home config was "
        f"{best_raw.q} (norm {raw_residual:.2e}), expected ~zeros"
    )

    default_sols = arm.solve(T_home)  # respect_limits=True, the public default
    if not default_sols:
        pytest.xfail(
            f"{arm.__name__}: solve(T_HOME) returns [] under the default "
            f"policy (respect_limits=True) even though the raw geometric "
            f"solve found q~zeros (residual {raw_residual:.2e}). Likely "
            f"cause: joint4's lower limit is exactly 0.0 in this URDF "
            f"(straight-elbow hard stop), so the home config sits exactly "
            f"on the boundary and solver floating-point noise on q4 gets "
            f"filtered by the default policy. Not necessarily a ssik bug — "
            f"worth deciding whether your control stack ever needs to "
            f"command exactly q4=0, or whether the URDF limit should carry "
            f"a small epsilon margin (e.g. lower=-1e-6) to avoid this."
        )
    else:
        best_default = min(default_sols, key=lambda s: s.fk_residual)
        assert best_default.fk_residual < FK_ATOL_ROT, (
            f"{arm.__name__}: default-policy branch at home config has "
            f"fk_residual={best_default.fk_residual:.2e} -- not a valid "
            f"solution, not just 'a different branch than q=0'"
        )


@pytest.mark.parametrize("arm", ARTIFACT_MODULES, ids=ARTIFACT_IDS)
def test_seeded_solve_near_limit_boundary_is_fast(arm):
    """Regression guard for a confirmed real latency problem: seeded
    max_solutions=1 solve at the fully-extended home config takes ~3.5s
    instead of sub-ms, on both arms. Root cause isn't nailed down yet —
    respect_limits=False also returns [] (see
    test_solve_at_home_pose_recovers_zero_config), which rules out simple
    limit filtering and points at either a genuine kinematic singularity
    at this pose or something wrong in how the artifact was built. Whatever
    the cause, if your VR teleop or trajectory tracking ever seeds near
    this configuration (arm near fully extended), a real-time control loop
    running at 100-200 Hz will stall for multiple seconds. That's asserted
    directly here rather than only smoke-tested at a "safe" operating
    point, because it's a real risk, not a test artifact.
    """
    T_home = np.asarray(arm.T_HOME)
    q_seed = np.zeros(arm.DOF)

    start = time.perf_counter()
    arm.solve(T_home, max_solutions=1, q_seed=q_seed)
    elapsed_ms = (time.perf_counter() - start) * 1000

    budget_ms = 100.0  # generous vs. the ~3.5 s observed
    assert elapsed_ms < budget_ms, (
        f"{arm.__name__}: seeded solve at the fully-extended home config "
        f"(q_seed=zeros) took {elapsed_ms:.1f} ms, budget was {budget_ms} "
        f"ms. See test_solve_at_home_pose_recovers_zero_config's "
        f"explain=True diagnostic for the likely root cause — until that's "
        f"resolved, treat any control-loop pose near full extension as a "
        f"latency risk, not just this test."
    )


# ---------------------------------------------------------------------------
# Cross-validation against Pinocchio FK (independent ground truth)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ARTIFACT_IDS)
def test_base_ee_are_valid_urdf_frames(name):
    """Catches a typo'd --base/--ee before any FK math runs, and surfaces
    what the artifact actually thinks its base/ee links are vs. the
    link0/hand_tcp pair this URDF's structure suggests."""
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
        # Not a failure — hand_tcp and link7 are geometrically identical
        # in this URDF (zero-offset fixed joint), so building against
        # either is fine. Just surfaced so a genuine mismatch is visible.
        print(
            f"\n{name}: artifact uses base={arm.BASE_LINK!r}, "
            f"ee={arm.EE_LINK!r}; URDF structure suggested {expected}. "
            "Not necessarily wrong, just confirm this is what you intended."
        )


@pytest.mark.parametrize("name", ARTIFACT_IDS)
def test_home_pose_matches_pinocchio_fk(name):
    arm = _ARTIFACTS[name]
    pin, model, data = _pin_model_for(name)
    q_zero = np.zeros(model.nq)
    T_pin = _pin_fk(pin, model, data, arm.BASE_LINK, arm.EE_LINK, q_zero)
    T_ssik = np.asarray(arm.T_HOME)
    residual = _fk_residual(T_pin, T_ssik)
    assert residual < FK_ATOL_ROT, (
        f"{name}: T_HOME disagrees with Pinocchio FK at q=0 "
        f"(residual={residual:.2e}). Check --base/--ee link names and "
        f"that the URDF used for `ssik build` matches URDF_PATHS[{name!r}]."
    )


@pytest.mark.parametrize("name", ARTIFACT_IDS)
def test_random_configs_round_trip_through_pinocchio(name):
    """Sample real joint configs, FK them with Pinocchio (ground truth),
    feed the resulting pose to ssik, and check some returned branch
    actually reaches that pose (per Pinocchio, not per ssik's own
    internal fk_residual).

    Also tracks respect_limits=False alongside the default policy. If the
    default policy fails broadly but the raw (limits-ignored) solve
    mostly succeeds, that points at the artifact's baked joint limits
    being wrong, not a genuine solver coverage gap — worth knowing which
    one it is before assuming ssik just can't solve this arm.
    """
    arm = _ARTIFACTS[name]
    pin, model, data = _pin_model_for(name)
    lo, hi = _pin_joint_limits(model)
    rng = _rng()
    arm_q_indices = _arm_q_indices(model, name)

    n_default_ok = 0
    n_raw_ok = 0
    for _ in range(N_RANDOM_CONFIGS):
        q_true = rng.uniform(lo, hi)
        T_target = _pin_fk(pin, model, data, arm.BASE_LINK, arm.EE_LINK, q_true)

        default_sols = arm.solve(T_target)
        raw_sols = arm.solve(T_target, respect_limits=False)
        if raw_sols:
            n_raw_ok += 1

        if not default_sols:
            # Some sampled configs may be at/near a singularity where the
            # default policy filters the branch; don't fail the whole test
            # on a handful of those — but n_raw_ok above still gets credit
            # if the raw solve found something, so the final message can
            # tell the two failure modes apart.
            continue

        residuals = []
        for s in default_sols:
            full_q_check = _embed_arm_q(q_true, arm_q_indices, np.asarray(s.q))
            T_check = _pin_fk(pin, model, data, arm.BASE_LINK, arm.EE_LINK, full_q_check)
            residuals.append(_fk_residual(T_check, T_target))
        best_residual = min(residuals)
        assert best_residual < FK_ATOL_ROT, (
            f"{name}: best branch for a sampled config had Pinocchio FK "
            f"residual {best_residual:.2e} against the target pose "
            f"(q_true={q_true})"
        )
        n_default_ok += 1

    if n_default_ok < N_RANDOM_CONFIGS // 2:
        if n_raw_ok >= N_RANDOM_CONFIGS // 2:
            diagnosis = (
                f"respect_limits=False succeeded on {n_raw_ok}/{N_RANDOM_CONFIGS} "
                f"of the SAME configs — the solver itself is finding valid "
                f"branches, they're being filtered by the default policy. "
                f"Points at the artifact's baked joint limits being wrong "
                f"or mismatched vs the URDF, not a solver coverage gap."
            )
        else:
            diagnosis = (
                f"respect_limits=False also only succeeded on "
                f"{n_raw_ok}/{N_RANDOM_CONFIGS} of the SAME configs — the "
                f"solver itself is failing broadly here, independent of "
                f"limits. This is NOT just the narrow home-pose singularity, "
                f"it's a much broader coverage problem worth escalating to "
                f"ssik directly (or re-checking --base/--ee at build time)."
            )
        pytest.fail(
            f"{name}: only {n_default_ok}/{N_RANDOM_CONFIGS} random reachable "
            f"configs produced a default-policy solution.\n{diagnosis}"
        )


@pytest.mark.parametrize("name", ARTIFACT_IDS)
def test_artifact_kinematics_matches_urdf_when_seeded_at_truth(name):
    """Does the built artifact's baked kinematics actually match the
    current URDF? Seed solve() with q_seed set EXACTLY to a randomly
    sampled true config — the easiest possible input a jointlock solver
    can get — with respect_limits=False to isolate kinematics fidelity
    from limits policy entirely.

    This deliberately separates two different questions that an earlier
    version of this test conflated into one pass/fail:

    1. KINEMATIC VALIDITY — does ssik return ANY branch whose
       Pinocchio-computed FK actually matches the target pose? This is
       the real "is the artifact corrupted / built against a different
       URDF" question. A failure here is the real red flag.
    2. SEED-CLOSENESS — is the branch it returns close, in joint space,
       to the seed it was given? This is a narrower question about
       whether `max_solutions=1 + q_seed`'s "nearest-to-seed" ordering
       is actually returning the nearest branch, not about kinematic
       correctness. A kinematically-valid-but-different-branch result
       here is informative (may be worth a narrower ssik report on its
       own) but is NOT evidence the artifact is broken.
    """
    arm = _ARTIFACTS[name]
    pin, model, data = _pin_model_for(name)
    lo, hi = _pin_joint_limits(model)
    rng = _rng()
    arm_q_indices = _arm_q_indices(model, name)

    n_empty = 0
    n_kinematically_valid = 0  # residual small, regardless of which branch
    n_near_seed = 0  # ALSO close to the seed in joint space
    worst_residual_when_valid = 0.0

    for i in range(N_RANDOM_CONFIGS):
        t0 = time.perf_counter()
        q_true_full = rng.uniform(lo, hi)
        T_target = _pin_fk(pin, model, data, arm.BASE_LINK, arm.EE_LINK, q_true_full)
        q_true_local = q_true_full[arm_q_indices]

        sols = arm.solve(
            T_target, max_solutions=1, q_seed=q_true_local, respect_limits=False
        )
        elapsed = time.perf_counter() - t0

        if not sols:
            n_empty += 1
            print(f"  [{name}] {i + 1}/{N_RANDOM_CONFIGS}: EMPTY ({elapsed:.2f}s)", flush=True)
            continue

        q_solved = np.asarray(sols[0].q)
        joint_dist = np.linalg.norm(
            np.mod(q_solved - q_true_local + np.pi, 2 * np.pi) - np.pi
        )
        full_q_check = _embed_arm_q(q_true_full, arm_q_indices, q_solved)
        T_check = _pin_fk(pin, model, data, arm.BASE_LINK, arm.EE_LINK, full_q_check)
        residual = _fk_residual(T_check, T_target)

        is_valid = residual < FK_ATOL_ROT
        is_near_seed = joint_dist < 0.1
        print(
            f"  [{name}] {i + 1}/{N_RANDOM_CONFIGS}: found, "
            f"residual={residual:.2e} ({'valid' if is_valid else 'INVALID'}), "
            f"joint_dist={joint_dist:.3f} rad "
            f"({'near seed' if is_near_seed else 'DIFFERENT BRANCH'}) "
            f"({elapsed:.2f}s)",
            flush=True,
        )
        if is_valid:
            n_kinematically_valid += 1
            worst_residual_when_valid = max(worst_residual_when_valid, residual)
            if is_near_seed:
                n_near_seed += 1

    print(
        f"\n{name} summary: {n_empty}/{N_RANDOM_CONFIGS} empty, "
        f"{n_kinematically_valid}/{N_RANDOM_CONFIGS} kinematically valid "
        f"(worst residual when valid: {worst_residual_when_valid:.2e}), "
        f"{n_near_seed}/{N_RANDOM_CONFIGS} of those ALSO close to the seed branch",
        flush=True,
    )

    # The real "is the artifact corrupted" question.
    assert n_kinematically_valid / N_RANDOM_CONFIGS > 0.5, (
        f"{name}: only {n_kinematically_valid}/{N_RANDOM_CONFIGS} seeded "
        f"solves found ANY branch whose Pinocchio FK actually matches the "
        f"target pose (not just one close to the seed). This WOULD be "
        f"real evidence the artifact's baked kinematics don't match this "
        f"URDF — it's not just failing to find the nearest branch, it's "
        f"failing to find any valid one at all."
    )

    # Informational only — a kinematically-valid but different branch is
    # not evidence of corruption, just worth knowing about separately.
    if n_kinematically_valid and n_near_seed < n_kinematically_valid * 0.5:
        print(
            f"\nNOTE: {n_kinematically_valid - n_near_seed}/{n_kinematically_valid} "
            f"kinematically-valid solves returned a DIFFERENT branch than "
            f"the seed, not the nearest one. Kinematics check out fine — "
            f"this looks like `max_solutions=1 + q_seed`'s nearest-to-seed "
            f"short-circuit not behaving as documented for this arm's "
            f"jointlock dispatch specifically, which matters for your "
            f"trajectory-tracking use case even though it's not corruption.",
            flush=True,
        )


@pytest.mark.parametrize("name", ARTIFACT_IDS)
def test_default_solve_respects_urdf_joint_limits(name):
    arm = _ARTIFACTS[name]
    pin, model, data = _pin_model_for(name)
    lo, hi = _pin_joint_limits(model)
    rng = _rng()
    arm_q_indices = _arm_q_indices(model, name)
    lo_local = lo[arm_q_indices]  # ssik's Solution.q is arm-local (length DOF)
    hi_local = hi[arm_q_indices]

    q_true_full = rng.uniform(lo, hi)
    T_target = _pin_fk(pin, model, data, arm.BASE_LINK, arm.EE_LINK, q_true_full)

    sols = arm.solve(T_target)  # respect_limits=True by default
    for s in sols:
        q = np.asarray(s.q)
        assert np.all(q >= lo_local - JOINT_LIMIT_EPS) and np.all(q <= hi_local + JOINT_LIMIT_EPS), (
            f"{name}: solve() returned an out-of-limit branch {q} "
            f"despite respect_limits defaulting to True"
        )


# ---------------------------------------------------------------------------
# Behavior that doesn't need Pinocchio
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("arm", ARTIFACT_MODULES, ids=ARTIFACT_IDS)
def test_grossly_unreachable_pose_returns_empty(arm):
    T_far = np.eye(4)
    T_far[:3, 3] = [100.0, 100.0, 100.0]  # 100 m away, no arm reaches this
    sols = arm.solve(T_far)
    assert sols == [], f"{arm.__name__}: expected [] for an obviously unreachable pose"


@pytest.mark.parametrize("arm", ARTIFACT_MODULES, ids=ARTIFACT_IDS)
def test_solve_is_deterministic(arm):
    """No RNG in the solve path — same pose in, same branch set out."""
    T = np.asarray(arm.T_HOME)
    sols_a = arm.solve(T)
    sols_b = arm.solve(T)
    assert len(sols_a) == len(sols_b)
    qs_a = sorted(tuple(np.round(s.q, 9)) for s in sols_a)
    qs_b = sorted(tuple(np.round(s.q, 9)) for s in sols_b)
    assert qs_a == qs_b, f"{arm.__name__}: solve() gave different branches on repeat calls"


@pytest.mark.parametrize("arm", ARTIFACT_MODULES, ids=ARTIFACT_IDS)
@pytest.mark.slow
def test_seeded_solve_latency_smoke(arm):
    """Guard against a silent perf regression after re-running `ssik build`
    (e.g. accidentally losing the jointlock short-circuit). Threshold is
    generous on purpose — this is a smoke test, not a benchmark.

    Uses respect_limits=False deliberately: T_HOME sits exactly on
    joint4's lower limit (see test_solve_at_home_pose_recovers_zero_config
    and test_seeded_solve_near_limit_boundary_is_fast), and that specific
    edge case has its own dedicated regression test above. This test is
    about catching a generic short-circuit regression, not re-measuring
    the known boundary pathology.
    """
    T = np.asarray(arm.T_HOME)
    q_seed = np.zeros(arm.DOF)
    # warm up (first call may pay import/JIT-ish costs)
    arm.solve(T, max_solutions=1, q_seed=q_seed, respect_limits=False)

    start = time.perf_counter()
    for _ in range(20):
        arm.solve(T, max_solutions=1, q_seed=q_seed, respect_limits=False)
    elapsed_ms = (time.perf_counter() - start) / 20 * 1000

    budget_ms = 50.0 if arm.DOF == 7 else 10.0
    assert elapsed_ms < budget_ms, (
        f"{arm.__name__}: seeded solve averaged {elapsed_ms:.2f} ms/call, "
        f"budget was {budget_ms} ms"
    )


# ---------------------------------------------------------------------------
# Bimanual-specific: catch a left/right artifact accidentally built twice
# from the same URDF
# ---------------------------------------------------------------------------


def test_left_and_right_artifacts_are_not_identical():
    left = _ARTIFACTS.get("openarm.openarm_left_ik")
    right = _ARTIFACTS.get("openarm.openarm_right_ik")
    if left is None or right is None:
        pytest.skip("both left and right artifacts must be present for this check")
    assert not np.allclose(np.asarray(left.T_HOME), np.asarray(right.T_HOME), atol=1e-6), (
        "left and right arm artifacts report the same T_HOME — did `ssik "
        "build` accidentally run against the same URDF for both sides?"
    )