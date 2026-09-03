"""
Small, direct unit tests for the code that lives in `openarm` itself
(`Openarm`, `OpenarmConfig`, `_ArmScope`) -- as opposed to `mj_manipulator`,
which has its own test suite.

These are fast and narrow: each test checks one piece of the wrapper's
plumbing (config shape, dict-like arm access, closest-arm selection, abort
flag bookkeeping) in isolation from full planning/collision behavior, which
is covered by test_kinematic_safety.py and test_collision_safety.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from openarm.config import OpenarmConfig
from openarm.robot import Openarm


@pytest.fixture(scope="module")
def robot():
    """A real Openarm instance. Skips the module if the model can't be
    resolved in this environment (run
    `python -m openarm_assets.assembly --sides bimanual` first)."""
    try:
        return Openarm(config=OpenarmConfig.default())
    except FileNotFoundError as e:
        pytest.skip(
            f"Openarm model not available in this environment: {e}",
            allow_module_level=True,
        )


@pytest.fixture(autouse=True)
def _reset_robot(robot):
    """Known state before every test, so joint manipulation doesn't leak
    between tests in this module."""
    robot.reset()
    yield


# ---------------------------------------------------------------------------
# OpenarmConfig
# ---------------------------------------------------------------------------


class TestConfig:
    def test_default_config_builds(self):
        """OpenarmConfig.default() should construct without raising -- this
        alone would have caught the earlier AttributeError from the
        wrapper-class kinematic-limits bug."""
        cfg = OpenarmConfig.default()
        assert cfg.left_arm is not None
        assert cfg.right_arm is not None

    @pytest.mark.parametrize("side", ["left_arm", "right_arm"])
    def test_arm_kinematic_limits_are_positive_and_full_dof(self, side):
        """Each arm spec must carry per-joint velocity/acceleration limits
        for every joint -- these are what keep planned trajectories within
        what the real actuators can safely do. A zero or missing limit here
        would silently make the safety checks in test_kinematic_safety.py
        meaningless."""
        cfg = OpenarmConfig.default()
        arm_spec = getattr(cfg, side)
        limits = arm_spec.kinematic_limits
        dof = len(cfg.joint_names(arm_spec))

        assert limits.velocity.shape == (dof,)
        assert limits.acceleration.shape == (dof,)
        assert np.all(limits.velocity > 0), "every joint must have a positive velocity limit"
        assert np.all(limits.acceleration > 0), "every joint must have a positive acceleration limit"

    def test_joint_names_are_correctly_prefixed(self):
        """Regression guard for the double-underscore prefix bug noted in
        OpenarmConfig.joint_names: 'openarm_left' + '_joint1' should give
        'openarm_left_joint1', not 'openarm_left__joint1'."""
        cfg = OpenarmConfig.default()
        names = cfg.joint_names(cfg.left_arm)
        assert all("__" not in n for n in names), f"double-underscore joint name(s): {names}"
        assert all(n.startswith("openarm_left_joint") for n in names)


# ---------------------------------------------------------------------------
# Openarm.arms -- Mapping passthrough
# ---------------------------------------------------------------------------


class TestArmsMapping:
    """Regression guard for the ArmGroup-as-Mapping fix, exercised through
    the public `Openarm.arms` / `robot.left` / `robot.right` surface rather
    than mj_manipulator's own ArmGroup tests."""

    def test_len_and_iteration(self, robot):
        assert len(robot.arms) == 2
        assert set(robot.arms) == {"left", "right"}

    def test_getitem_and_keys_and_items(self, robot):
        assert "left" in robot.arms.keys()
        assert robot.arms["left"] is robot.arms.get("left")
        names = {name for name, _arm in robot.arms.items()}
        assert names == {"left", "right"}

    def test_arm_scope_matches_group_lookup(self, robot):
        """robot.left.arm and robot.right.arm should be exactly the arms
        addressed by robot.arms['left'] / ['right'] -- the two access paths
        must not diverge."""
        assert robot.left.arm is robot.arms["left"]
        assert robot.right.arm is robot.arms["right"]


# ---------------------------------------------------------------------------
# plan_reach_to_pose -- closest-arm selection
# ---------------------------------------------------------------------------


class TestPlanReachToPose:
    @staticmethod
    def _moves(traj, atol=1e-6) -> bool:
        """True if a trajectory's final position differs meaningfully from
        its first -- i.e. this arm was actually commanded, as opposed to
        being held at its running configuration."""
        return not np.allclose(traj.positions[0], traj.positions[-1], atol=atol)

    def test_picks_the_nearer_arm(self, robot):
        """plan_reach_to_pose should route the goal to whichever arm's
        current end-effector is closest to the first waypoint. The group
        plan still returns a trajectory for both arms (per
        ArmGroup.plan_to_poses: 'unnamed arms hold'), so the check is on
        which arm actually moves, not on which trajectory is None.

        Keeps each arm's own current orientation and nudges only position,
        so the target is a small, easily reachable step rather than an
        arbitrary (possibly unreachable) orientation.
        """
        near_left = robot.left.arm.get_ee_pose().copy()
        near_left[:3, 3] += np.array([0.03, 0.0, 0.0])
        result = robot.plan_reach_to_pose([near_left])
        assert result is not None and result.success
        assert self._moves(result.left)
        assert not self._moves(result.right)

        robot.reset()

        near_right = robot.right.arm.get_ee_pose().copy()
        near_right[:3, 3] += np.array([0.03, 0.0, 0.0])
        result = robot.plan_reach_to_pose([near_right])
        assert result is not None and result.success
        assert self._moves(result.right)
        assert not self._moves(result.left)

    def test_raises_on_empty_goal(self, robot):
        with pytest.raises(ValueError):
            robot.plan_reach_to_pose([])


# ---------------------------------------------------------------------------
# Abort flag bookkeeping (not full mid-execution behavior -- see
# test_system_integration.py for that)
# ---------------------------------------------------------------------------


class TestAbortFlag:
    def test_request_and_clear_abort_toggle_the_flag(self, robot):
        assert robot.is_abort_requested() is False
        robot.request_abort()
        assert robot.is_abort_requested() is True
        robot.clear_abort()
        assert robot.is_abort_requested() is False

    def test_abort_flag_starts_clear_after_reset(self, robot):
        """reset() must not leave a stale abort flag set from a previous
        test/operation -- an operator clearing a fault and resetting the
        robot should get a robot that's actually willing to move again."""
        robot.request_abort()
        robot.clear_abort()
        robot.reset()
        assert robot.is_abort_requested() is False