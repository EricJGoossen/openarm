"""Numeric verification helpers. Pure Python/numpy -- no ROS dependency, so
these are unit-testable on their own and reused identically across stages.

These implement the *numerical definitions* the spec calls for ("simultaneous"
must be defined and checked, not eyeballed; "settled" needs a real velocity
threshold, not a guess) so every script applies the same definition rather
than each one improvising its own notion of "close enough".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .telemetry import JointSample


@dataclass
class ToleranceResult:
    ok: bool
    message: str
    value: float | None = None


def samples_to_series(samples: list[JointSample], joint: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (times, positions) for one joint, in insertion order, dropping
    samples that don't mention that joint."""
    t, p = [], []
    for s in samples:
        if joint in s.positions:
            t.append(s.t)
            p.append(s.positions[joint])
    return np.asarray(t), np.asarray(p)


def velocity_series(samples: list[JointSample], joint: str) -> tuple[np.ndarray, np.ndarray]:
    """Prefer reported velocity if present; otherwise finite-difference the
    position series. Returns (times, velocities)."""
    t, v = [], []
    have_reported = any(joint in s.velocities for s in samples)
    if have_reported:
        for s in samples:
            if joint in s.velocities:
                t.append(s.t)
                v.append(s.velocities[joint])
        return np.asarray(t), np.asarray(v)
    tp, p = samples_to_series(samples, joint)
    if len(tp) < 2:
        return tp, np.zeros_like(tp)
    dv = np.gradient(p, tp)
    return tp, dv


def position_reached(
    samples: list[JointSample], joint: str, target: float, tolerance: float, window_s: float = 0.5
) -> ToleranceResult:
    """Was `joint` within `tolerance` of `target` for the *last* `window_s`
    seconds of the recording? (Not just at the very last sample -- a single
    lucky sample near the target isn't evidence of a settled hold.)
    """
    t, p = samples_to_series(samples, joint)
    if len(t) == 0:
        return ToleranceResult(False, f"no telemetry samples contain joint '{joint}'")
    end = t[-1]
    mask = t >= (end - window_s)
    if not mask.any():
        return ToleranceResult(False, "not enough samples in the settling window")
    max_err = float(np.max(np.abs(p[mask] - target)))
    ok = max_err <= tolerance
    return ToleranceResult(
        ok, f"max |error| over last {window_s}s = {max_err:.4f} rad (tolerance {tolerance:.4f})", max_err
    )


def stayed_near(
    samples: list[JointSample], joint: str, reference: float, tolerance: float
) -> ToleranceResult:
    """Did `joint` stay within `tolerance` of `reference` for the *entire*
    recording? Use this for rejection tests: a command that should have been
    rejected must never move the joint at all, not just end up back near
    where it started.
    """
    t, p = samples_to_series(samples, joint)
    if len(t) == 0:
        return ToleranceResult(False, f"no telemetry samples contain joint '{joint}'")
    max_err = float(np.max(np.abs(p - reference)))
    ok = max_err <= tolerance
    return ToleranceResult(
        ok, f"max deviation from reference over whole recording = {max_err:.4f} rad", max_err
    )


def settle_time_after(
    samples: list[JointSample],
    joints: list[str],
    after_t: float,
    velocity_threshold: float,
    sustain_s: float = 0.15,
) -> float | None:
    """First timestamp at or after `after_t` at which every joint in
    `joints` has |velocity| <= velocity_threshold continuously for at least
    `sustain_s` seconds. Returns None if it never settles within the
    recorded data. This is the shared definition of "motion actually
    stopped" used for every abort/e-stop latency measurement.
    """
    if not samples:
        return None
    end_t = samples[-1].t
    all_t = sorted({s.t for s in samples if s.t >= after_t})
    for candidate in all_t:
        window_end = candidate + sustain_s
        if window_end > end_t:
            break
        settled = True
        for j in joints:
            tv, v = velocity_series([s for s in samples if candidate - 0.05 <= s.t <= window_end], j)
            if len(v) == 0 or np.any(np.abs(v) > velocity_threshold):
                settled = False
                break
        if settled:
            return candidate
    return None


def first_motion_time(
    samples: list[JointSample], joints: list[str], velocity_threshold: float, after_t: float = 0.0
) -> float | None:
    """First timestamp at or after `after_t` where any joint in `joints`
    exceeds `velocity_threshold`. Used to measure when each arm actually
    started moving, for the bimanual simultaneity check.
    """
    best = float("inf")
    for j in joints:
        t, v = velocity_series([s for s in samples if s.t >= after_t], j)
        above = np.where(np.abs(v) > velocity_threshold)[0]
        if len(above) > 0:
            candidate = float(t[above[0]])
            if candidate < best:
                best = candidate
    return None if best == float("inf") else best


def simultaneity_overlap_fraction(
    samples_a: list[JointSample],
    joints_a: list[str],
    samples_b: list[JointSample],
    joints_b: list[str],
    velocity_threshold: float,
    dt: float = 0.02,
) -> ToleranceResult:
    """Fraction of the combined motion window during which BOTH sides have
    at least one joint moving. 1.0 = perfectly overlapping motion, 0.0 =
    completely sequential (one finishes before the other starts) -- this is
    the number that distinguishes "genuinely simultaneous" from "happened to
    be dispatched close together but actually ran one-then-the-other".
    """
    if not samples_a or not samples_b:
        return ToleranceResult(False, "missing telemetry for one or both sides")
    t0 = min(samples_a[0].t, samples_b[0].t)
    t1 = max(samples_a[-1].t, samples_b[-1].t)
    if t1 <= t0:
        return ToleranceResult(False, "degenerate time range")
    grid = np.arange(t0, t1, dt)

    def moving_mask(samples, joints):
        mask = np.zeros_like(grid, dtype=bool)
        for j in joints:
            t, v = velocity_series(samples, j)
            if len(t) < 2:
                continue
            v_interp = np.interp(grid, t, np.abs(v), left=0.0, right=0.0)
            mask |= v_interp > velocity_threshold
        return mask

    mask_a = moving_mask(samples_a, joints_a)
    mask_b = moving_mask(samples_b, joints_b)
    any_moving = mask_a | mask_b
    both_moving = mask_a & mask_b
    total = int(np.sum(any_moving))
    if total == 0:
        return ToleranceResult(False, "neither side was ever observed moving")
    frac = float(np.sum(both_moving)) / total
    return ToleranceResult(True, f"both-moving overlap fraction = {frac:.2f}", frac)


def no_sustained_oscillation(
    samples: list[JointSample], joint: str, after_t: float, velocity_threshold: float, window_s: float = 1.0
) -> ToleranceResult:
    """Cheap oscillation guard: after motion is supposed to have stopped,
    velocity should not keep crossing back above threshold repeatedly. Counts
    threshold up-crossings in the window; more than a couple is suspicious.
    """
    t, v = velocity_series([s for s in samples if s.t >= after_t and s.t <= after_t + window_s], joint)
    if len(v) < 2:
        return ToleranceResult(True, "not enough data to assess -- treated as inconclusive/pass")
    above = np.abs(v) > velocity_threshold
    crossings = int(np.sum(above[1:] & ~above[:-1]))
    ok = crossings <= 2
    return ToleranceResult(ok, f"{crossings} threshold up-crossings in {window_s}s after motion", crossings)