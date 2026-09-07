"""Independent, offline collision verification for bimanual trials.

This is the concrete implementation of the spec's requirement that a
planner's collision-free guarantee be re-checked against what *actually
executed* on hardware, not assumed to transfer from planning correctness.
It works entirely from recorded telemetry -- it never talks to hardware or
ROS -- so it runs identically whether you're checking a trial from five
minutes ago or five weeks ago, and it uses the same collision model the
planner itself uses, so "collision-free" means the same thing in both
places.

Deliberately does not import anything ROS-specific, so it can be run as a
pure offline analysis step on saved `joint_states.json` files.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .telemetry import JointSample

if TYPE_CHECKING:
    from openarm.robot import ArmGroup 
    from openarm.config import OpenarmConfig

# Type alias: a zero-arg factory that returns a live ArmGroup-like object
# with `.arm(name).joint_names`, `.set_joint_positions(q)`, and
# `.check_collisions()`. Passed in rather than imported directly so this
# module doesn't hard-code a dependency on one specific robot package.
ArmGroupFactory = Callable[[], ArmGroup]


@dataclass
class CollisionCheckResult:
    ok: bool
    checked_timestamps: int
    violation_times: list[float] = field(default_factory=list)
    details: list[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        if self.ok:
            return f"collision-free at all {self.checked_timestamps} sampled instants"
        return (
            f"collision detected at {len(self.violation_times)} / "
            f"{self.checked_timestamps} sampled instants "
            f"(first at t={self.violation_times[0]:.3f}s)"
        )


def _resample(times: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Linear interpolation with edge-hold, for putting two independently-
    sampled joint-state streams onto one common time grid."""
    return np.interp(grid, times, values, left=values[0], right=values[-1])


def build_joint_matrix(
    samples: list[JointSample], joint_names: list[str], grid: np.ndarray
) -> np.ndarray:
    """Resample every joint in `joint_names` from `samples` onto `grid`.
    Returns an (len(grid), len(joint_names)) array."""
    out = np.zeros((len(grid), len(joint_names)))
    for j, name in enumerate(joint_names):
        t, p = [], []
        for s in samples:
            if name in s.positions:
                t.append(s.t)
                p.append(s.positions[name])
        if len(t) < 2:
            raise ValueError(f"joint '{name}' has fewer than 2 telemetry samples -- can't resample")
        out[:, j] = _resample(np.asarray(t), np.asarray(p), grid)
    return out


def check_pair_collision_free(
    arm_group_factory: ArmGroupFactory,
    left_joint_names: list[str],
    left_samples: list[JointSample],
    right_joint_names: list[str],
    right_samples: list[JointSample],
    dt: float = 0.02,
    left_key: str = "left",
    right_key: str = "right",
) -> CollisionCheckResult:
    """Replay two arms' recorded telemetry through a fresh collision-checked
    model at a shared time grid, and report every instant (if any) where the
    model reports a collision between the two arms.

    `arm_group_factory` must build a *fresh* ArmGroup-like object each call
    (don't reuse one that a live execution might still be touching) exposing
    at minimum: `arm_group[key].dof`, and `arm_group.check_collisions()`
    after positions are set via `arm_group.arm(key).set_joint_positions(q)`
    or equivalent. See `default_openarm_factory` below for the concrete
    wiring against this codebase's `Openarm`/`ArmGroup`.
    """
    t0 = max(left_samples[0].t, right_samples[0].t)
    t1 = min(left_samples[-1].t, right_samples[-1].t)
    if t1 <= t0:
        return CollisionCheckResult(
            ok=False, checked_timestamps=0, details=["left/right telemetry windows don't overlap"]
        )
    grid = np.arange(t0, t1, dt)

    q_left = build_joint_matrix(left_samples, left_joint_names, grid)
    q_right = build_joint_matrix(right_samples, right_joint_names, grid)

    arm_group = arm_group_factory()

    violations: list[float] = []
    details: list[str] = []
    for i, t in enumerate(grid):
        arm_group[left_key].set_joint_positions(q_left[i])
        arm_group[right_key].set_joint_positions(q_right[i])
        collisions = arm_group.check_collisions()
        if collisions:
            violations.append(float(t))
            details.append(f"t={t:.3f}s: {collisions}")

    return CollisionCheckResult(
        ok=(len(violations) == 0),
        checked_timestamps=len(grid),
        violation_times=violations,
        details=details[:20],  # cap detail volume; violation_times has the full list
    )


def default_openarm_factory(config_factory: Callable[[], OpenarmConfig] | None = None) -> ArmGroupFactory:
    """Returns a factory building a fresh `ArmGroup` from this codebase's
    `openarm.robot.Openarm`, for the common case. Pass a custom
    `config_factory` (e.g. `lambda: OpenarmConfig.from_yaml(path)`) if the
    default config isn't what a given session used.
    """

    def factory():
        from openarm.config import OpenarmConfig
        from openarm.robot import Openarm

        cfg = config_factory() if config_factory is not None else OpenarmConfig.default()
        robot = Openarm(config=cfg)
        return robot._arm_group  # noqa: SLF001 -- intentional: this is an offline analysis tool,
        # not application code, and ArmGroup itself doesn't need to be a
        # public attribute for every other purpose.

    return factory