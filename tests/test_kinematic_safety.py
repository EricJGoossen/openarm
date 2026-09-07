"""
GATE FILE -- velocity/acceleration safety for real-hardware execution.

Every trajectory `Openarm` hands to an executor (sim or real) has already
been through `ArmGroup.retime` / TOPP-RA, which is supposed to guarantee the
result never exceeds each joint's configured `KinematicLimits`. Nothing else
in the pipeline re-checks that guarantee before commanding real motors, so
if it doesn't hold, an over-speed or over-accel command reaches actual
hardware with no other safety net in this codebase.

These tests independently re-verify that guarantee against the *actual*
`Trajectory.velocities` / `Trajectory.accelerations` arrays returned by
`Openarm.plan_to_configuration` / `plan_ee_to_pose`, across single-arm,
bimanual, small-motion, and large-random-swing goals. If any of these fail,
do not run the corresponding motion on the real robot -- the planned
trajectory is telling the joints to move faster or accelerate harder than
they're rated for.
"""

from __future__ import annotations

import numpy as np
import pytest

from openarm.config import OpenarmConfig
from openarm.robot import Openarm

# Small numerical tolerance for floating-point/solver noise -- NOT a safety
# margin. Anything beyond this fraction of the configured limit is a real
# violation, not rounding error.
REL_TOL = 1e-3

N_RANDOM_SEEDS = 8


@pytest.fixture(scope="module")
def robot():
    try:
        return Openarm(config=OpenarmConfig.default())
    except FileNotFoundError as e:
        pytest.skip(
            f"Openarm model not available in this environment: {e}",
            allow_module_level=True,
        )


@pytest.fixture(autouse=True)
def _reset_robot(robot):
    robot.reset()
    yield


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_within_limits(traj, limits, label):
    vmax = np.max(np.abs(traj.velocities), axis=0)
    amax = np.max(np.abs(traj.accelerations), axis=0)
    v_over = vmax > limits.velocity * (1 + REL_TOL)
    a_over = amax > limits.acceleration * (1 + REL_TOL)

    assert not v_over.any(), (
        f"{label}: VELOCITY limit exceeded on joint index(es) {np.where(v_over)[0].tolist()} "
        f"-- reached {vmax[v_over].tolist()} rad/s vs limit {limits.velocity[v_over].tolist()} rad/s"
    )
    assert not a_over.any(), (
        f"{label}: ACCELERATION limit exceeded on joint index(es) {np.where(a_over)[0].tolist()} "
        f"-- reached {amax[a_over].tolist()} rad/s^2 vs limit {limits.acceleration[a_over].tolist()} rad/s^2"
    )


def _assert_starts_and_ends_at_rest(traj, label, atol=1e-6):
    """A trajectory that's still moving at its last waypoint hands the
    executor a robot in motion with nothing further commanded -- on real
    hardware that's an uncontrolled coast. Both endpoints should be at
    zero velocity."""
    assert np.allclose(traj.velocities[0], 0.0, atol=atol), (
        f"{label}: trajectory does not start at rest: {traj.velocities[0]}"
    )
    assert np.allclose(traj.velocities[-1], 0.0, atol=atol), (
        f"{label}: trajectory does not end at rest: {traj.velocities[-1]}"
    )


def _random_goal(rng, lo, hi, margin=0.05):
    """A random joint configuration safely inside the joint-limit range
    (margin keeps it off the exact limit, which can make IK finicky)."""
    lo, hi = np.asarray(lo), np.asarray(hi)
    span = hi - lo
    return lo + margin * span + rng.random(len(lo)) * span * (1 - 2 * margin)

def _assert_positions_within_limits(traj, lo, hi, margin_rad, label):
    """Independent position-limit check, mirroring `_assert_within_limits`'
    treatment of velocity/acceleration but for the retimed/split trajectory's
    actual joint positions at every waypoint.
 
    `margin_rad` is an ABSOLUTE margin in radians (matching
    openarm_impedance_control's `joint_position_margin` convention: a fixed
    0.05 rad on every joint regardless of that joint's total range) -- do not
    confuse this with `_random_goal`'s `margin` parameter above, which is a
    FRACTION of each joint's span and is used only for goal generation, not
    for this check.
    """
    lo, hi = np.asarray(lo), np.asarray(hi)
    pos = traj.positions
    below = pos < (lo + margin_rad) - REL_TOL
    above = pos > (hi - margin_rad) + REL_TOL
    assert not below.any(), (
        f"{label}: position BELOW the admissible lower limit (with {margin_rad} rad margin) "
        f"at waypoint(s) {np.where(below.any(axis=1))[0].tolist()[:5]}"
    )
    assert not above.any(), (
        f"{label}: position ABOVE the admissible upper limit (with {margin_rad} rad margin) "
        f"at waypoint(s) {np.where(above.any(axis=1))[0].tolist()[:5]}"
    )


# ---------------------------------------------------------------------------
# Single-arm: plan_to_configuration
# ---------------------------------------------------------------------------


class TestSingleArmJointSpaceLimits:
    @pytest.mark.parametrize("side", ["left", "right"])
    def test_small_motion_respects_limits(self, robot, side):
        arm = getattr(robot, side).arm
        q0 = arm.get_joint_positions()
        goal = q0.copy()
        goal[0] += 0.2

        result = robot.plan_to_configuration({side: goal}, seed=0)
        assert result is not None and result.success, "planner failed on a small, easy motion"

        traj = getattr(result, side)
        _assert_within_limits(traj, arm.config.kinematic_limits, f"{side} small motion")
        _assert_starts_and_ends_at_rest(traj, f"{side} small motion")

    @pytest.mark.parametrize("side", ["left", "right"])
    @pytest.mark.parametrize("seed", range(N_RANDOM_SEEDS))
    def test_random_large_swing_respects_limits(self, robot, side, seed):
        """Fuzz across large, randomized single-arm reconfigurations --
        this is where a retimer is most likely to be pushed hard enough to
        reveal a limit violation that a single hand-picked example would
        miss."""
        arm = getattr(robot, side).arm
        lo, hi = arm.get_joint_limits()
        rng = np.random.default_rng(seed)
        goal = _random_goal(rng, lo, hi)

        result = robot.plan_to_configuration({side: goal}, seed=seed, timeout=15.0)
        if result is None or not result.success:
            pytest.skip("planner could not reach this random goal (not a kinematic-limits question)")

        traj = getattr(result, side)
        _assert_within_limits(traj, arm.config.kinematic_limits, f"{side} seed={seed}")
        _assert_starts_and_ends_at_rest(traj, f"{side} seed={seed}")


# ---------------------------------------------------------------------------
# Bimanual: plan_to_configuration with both arms
# ---------------------------------------------------------------------------


class TestBimanualJointSpaceLimits:
    @pytest.mark.parametrize("seed", range(N_RANDOM_SEEDS))
    def test_random_bimanual_goal_respects_per_arm_limits(self, robot, seed):
        """The combined 14-DOF path is retimed against concatenated
        per-arm limits and then split back into individual arm
        trajectories -- confirm each split-out trajectory still respects
        its *own* arm's limits, not just the combined vector's."""
        rng = np.random.default_rng(seed)
        lo_l, hi_l = robot.left.arm.get_joint_limits()
        lo_r, hi_r = robot.right.arm.get_joint_limits()
        goal_l = _random_goal(rng, lo_l, hi_l)
        goal_r = _random_goal(rng, lo_r, hi_r)

        result = robot.plan_to_configuration({"left": goal_l, "right": goal_r}, seed=seed, timeout=20.0)
        if result is None or not result.success:
            pytest.skip("planner could not reach this random goal pair")

        _assert_within_limits(result.left, robot.left.arm.config.kinematic_limits, f"bimanual left seed={seed}")
        _assert_within_limits(result.right, robot.right.arm.config.kinematic_limits, f"bimanual right seed={seed}")
        _assert_starts_and_ends_at_rest(result.left, f"bimanual left seed={seed}")
        _assert_starts_and_ends_at_rest(result.right, f"bimanual right seed={seed}")

    def test_both_arms_share_one_time_base(self, robot):
        """Sanity check underlying the limits guarantee above: the two
        arms must share duration and waypoint count, or 'retimed against
        concatenated limits' doesn't actually apply per-timestep the way
        the other tests in this file assume."""
        goal_l = robot.left.arm.get_joint_positions().copy()
        goal_l[0] += 0.3
        goal_r = robot.right.arm.get_joint_positions().copy()
        goal_r[0] += 0.3

        result = robot.plan_to_configuration({"left": goal_l, "right": goal_r}, seed=0)
        assert result is not None and result.success
        assert result.left.num_waypoints == result.right.num_waypoints
        assert np.isclose(result.left.duration, result.right.duration)
        assert np.array_equal(result.left.timestamps, result.right.timestamps)


# ---------------------------------------------------------------------------
# Cartesian goals: plan_ee_to_pose
# ---------------------------------------------------------------------------


class TestCartesianGoalLimits:
    @pytest.mark.parametrize("side", ["left", "right"])
    @pytest.mark.parametrize("delta", [
        [0.05, 0.0, 0.0],
        [0.0, 0.05, 0.0],
        [0.0, 0.0, 0.05],
        [0.05, -0.05, 0.05],
    ])
    def test_small_cartesian_move_respects_limits(self, robot, side, delta):
        """Cartesian-goal plans go through a different code path
        (plan_ee_to_pose -> ArmGroup.plan_to_poses) before the same
        _package_plan retime/split step -- verify it inherits the same
        safety guarantee."""
        arm = getattr(robot, side).arm
        goal_pose = arm.get_ee_pose().copy()
        goal_pose[:3, 3] += np.array(delta)

        result = robot.plan_ee_to_pose({side: goal_pose}, seed=0)
        if result is None or not result.success:
            pytest.skip("planner could not reach this Cartesian goal")

        traj = getattr(result, side)
        _assert_within_limits(traj, arm.config.kinematic_limits, f"{side} cartesian delta={delta}")
        _assert_starts_and_ends_at_rest(traj, f"{side} cartesian delta={delta}")


# ---------------------------------------------------------------------------
# Test position limits at every waypoint, not just velocity/acceleration
# ---------------------------------------------------------------------------
 

class TestPositionLimitsAtEveryWaypoint:
    """Companion to TestSingleArmJointSpaceLimits / TestBimanualJointSpaceLimits:
    same trajectories, same random seeds, checking `.positions` against joint
    limits (with the real runtime margin) at every waypoint, instead of
    `.velocities` / `.accelerations`.
    """
 
    MARGIN_RAD = 0.05  # matches openarm_impedance_control's joint_position_margin
 
    @pytest.mark.parametrize("side", ["left", "right"])
    @pytest.mark.parametrize("seed", range(N_RANDOM_SEEDS))
    def test_single_arm_random_swing_respects_position_limits(self, robot, side, seed):
        arm = getattr(robot, side).arm
        lo, hi = arm.get_joint_limits()
        rng = np.random.default_rng(seed)
        goal = _random_goal(rng, lo, hi)
 
        result = robot.plan_to_configuration({side: goal}, seed=seed, timeout=15.0)
        if result is None or not result.success:
            pytest.skip("planner could not reach this random goal (not a position-limits question)")
 
        traj = getattr(result, side)
        _assert_positions_within_limits(traj, lo, hi, self.MARGIN_RAD, f"{side} seed={seed}")
 
    @pytest.mark.parametrize("seed", range(N_RANDOM_SEEDS))
    def test_bimanual_random_goal_respects_position_limits(self, robot, seed):
        rng = np.random.default_rng(seed)
        lo_l, hi_l = robot.left.arm.get_joint_limits()
        lo_r, hi_r = robot.right.arm.get_joint_limits()
        goal_l = _random_goal(rng, lo_l, hi_l)
        goal_r = _random_goal(rng, lo_r, hi_r)
 
        result = robot.plan_to_configuration({"left": goal_l, "right": goal_r}, seed=seed, timeout=20.0)
        if result is None or not result.success:
            pytest.skip("planner could not reach this random goal pair")
 
        _assert_positions_within_limits(result.left, lo_l, hi_l, self.MARGIN_RAD, f"bimanual left seed={seed}")
        _assert_positions_within_limits(result.right, lo_r, hi_r, self.MARGIN_RAD, f"bimanual right seed={seed}")
 
    @pytest.mark.parametrize("side", ["left", "right"])
    @pytest.mark.parametrize(
        "delta", [[0.05, 0.0, 0.0], [0.0, 0.05, 0.0], [0.0, 0.0, 0.05], [0.05, -0.05, 0.05]]
    )
    def test_cartesian_goal_respects_position_limits(self, robot, side, delta):
        arm = getattr(robot, side).arm
        lo, hi = arm.get_joint_limits()
        goal_pose = arm.get_ee_pose().copy()
        goal_pose[:3, 3] += np.array(delta)
 
        result = robot.plan_ee_to_pose({side: goal_pose}, seed=0)
        if result is None or not result.success:
            pytest.skip("planner could not reach this Cartesian goal")
 
        traj = getattr(result, side)
        _assert_positions_within_limits(traj, lo, hi, self.MARGIN_RAD, f"{side} cartesian delta={delta}")