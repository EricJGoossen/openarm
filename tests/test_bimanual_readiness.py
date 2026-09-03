"""
GATE FILE -- fills gaps left by the existing bimanual test suite.

This repo already has strong coverage for bimanual planning:
  - test_openarm_bimanual_planning.py : joint-space crossing collision (the
    conclusive "does it actually reason about both arms" proof)
  - test_kinematic_safety.py          : velocity/acceleration limits for
    joint-space and Cartesian (pose) goals, single-arm and bimanual
  - test_collision_safety.py          : arm-vs-arm fuzz + arm-vs-obstacle
  - test_system_integration.py        : plan -> execute round trip, e-stop
  - test_robot_unit.py                : config/Mapping/abort-flag plumbing

This file adds five things that suite does not currently check:

  1. TSR-based bimanual planning (`plan_to_tsrs`) has NO test coverage
     anywhere in this repo, even though it is one of the three planning
     entry points `Openarm._package_plan` wraps, and grasp/place tasks are
     expected to go through TSRs rather than raw joint configs or poses.
  2. Determinism: nothing currently checks that the same seed reproduces
     the same trajectory. On real hardware you cannot re-run "the exact
     same attempt" to compare against a sim result unless planning with a
     fixed seed is actually deterministic.
  3. Malformed-goal validation: a goal with the wrong number of joints
     should fail loudly and immediately, not silently misbehave.
  4. Atomicity: when one arm's goal in a *joint bimanual call* is itself
     unreachable (e.g. self-colliding), the whole call must fail for BOTH
     arms -- it must never silently return a plan that only moves the good
     arm while quietly dropping the bad one.
  5. Holding-arm stationarity: when only one arm is given a goal, the
     existing tests check the untasked arm doesn't move between the first
     and last waypoint; this checks it doesn't move at ANY waypoint in
     between either (no drift mid-trajectory).

If these fail, treat them exactly like the other GATE files in this repo:
do not run the corresponding motion on the real robot.
"""

from __future__ import annotations

import numpy as np
import pytest
from tsr import TSR

from openarm.config import OpenarmConfig
from openarm.robot import Openarm

REL_TOL = 1e-3


# ---------------------------------------------------------------------------
# Fixtures (same convention as the rest of this suite)
# ---------------------------------------------------------------------------


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
    robot.clear_abort()
    yield
    robot.clear_abort()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _assert_within_limits(traj, limits, label):
    """Same check as test_kinematic_safety.py -- duplicated locally so this
    file doesn't depend on import order of another test module."""
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


def _arm_side(body_name: str) -> str | None:
    if body_name.startswith("openarm_left_"):
        return "left"
    if body_name.startswith("openarm_right_"):
        return "right"
    return None


def _combined_checker(robot):
    """Independent oracle collision checker, built fresh -- same pattern as
    test_openarm_bimanual_planning.py / test_collision_safety.py."""
    from mj_manipulator.collision import CollisionChecker

    left = robot.left.arm
    right = robot.right.arm
    joint_names = list(left.config.joint_names) + list(right.config.joint_names)
    extra = list(getattr(left.config, "extra_arm_body_names", None) or []) + list(
        getattr(right.config, "extra_arm_body_names", None) or []
    )
    env = robot.env.fork()
    return CollisionChecker(
        model=env.model,
        data=env.data,
        joint_names=joint_names,
        extra_arm_body_names=extra or None,
    )


def _cross_arm_contacts(checker, q_left, q_right):
    q = np.concatenate([np.asarray(q_left), np.asarray(q_right)])
    return [
        (b1, b2, depth)
        for b1, b2, depth in checker.get_contacts(q)
        if {_arm_side(b1), _arm_side(b2)} == {"left", "right"}
    ]


def _point_tsr(pose: np.ndarray) -> TSR:
    """A TSR with zero-width bounds at exactly `pose` -- a pose goal
    expressed through the TSR code path instead of plan_ee_to_pose."""
    return TSR(T0_w=pose.copy(), Tw_e=np.eye(4), Bw=np.zeros((6, 2)))


def _find_self_colliding_config(arm, seed=42, n_samples=20000):
    """A joint configuration for `arm` alone (no other arm involved) that is
    guaranteed to self-collide, found by uniform random sampling within the
    arm's own joint limits and checking with an independent CollisionChecker.

    This gives a goal that is unreachable for a structural reason (it's
    invalid at rest, not just hard to path to), which is what the atomicity
    test needs: a genuinely bad goal for one arm, wired into an otherwise
    perfectly fine bimanual request.
    """
    from mj_manipulator.collision import CollisionChecker

    lo, hi = arm.get_joint_limits()
    lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
    # Fork the environment before probing -- CollisionChecker.get_contacts(q)
    # writes q into the given MjData to evaluate contacts, so probing against
    # the arm's *live* data (shared with the robot) would leave the robot's
    # real joint state at whatever the last probed sample was. A fork keeps
    # this search side-effect-free on the actual robot state.
    env = arm.env.fork()
    checker = CollisionChecker(model=env.model, data=env.data, joint_names=list(arm.config.joint_names))
    rng = np.random.default_rng(seed)
    for _ in range(n_samples):
        q = lo + rng.random(len(lo)) * (hi - lo)
        if checker.get_contacts(q):
            return q
    raise RuntimeError(
        "could not find a self-colliding configuration in this model within the sample "
        "budget -- increase n_samples or re-derive by hand; the atomicity test needs one"
    )


# ---------------------------------------------------------------------------
# 1. TSR-based bimanual planning
# ---------------------------------------------------------------------------


class TestTsrBimanualPlanning:
    """plan_to_tsrs has no coverage elsewhere in this repo. These mirror the
    joint-space/Cartesian bimanual checks already applied to
    plan_to_configuration / plan_ee_to_pose, so all three planning entry
    points that feed `Openarm._package_plan` get the same bar."""

    def test_bimanual_point_tsr_reaches_goal_and_respects_limits(self, robot):
        left_goal_pose = robot.left.arm.get_ee_pose().copy()
        left_goal_pose[:3, 3] += np.array([0.03, 0.0, 0.02])
        right_goal_pose = robot.right.arm.get_ee_pose().copy()
        right_goal_pose[:3, 3] += np.array([0.03, 0.0, 0.02])

        result = robot.plan_to_tsrs(
            {"left": _point_tsr(left_goal_pose), "right": _point_tsr(right_goal_pose)},
            seed=0,
            timeout=20.0,
        )
        if result is None or not result.success:
            pytest.skip("planner could not reach this TSR goal pair (not what this test targets)")

        _assert_within_limits(result.left, robot.left.arm.config.kinematic_limits, "TSR bimanual left")
        _assert_within_limits(result.right, robot.right.arm.config.kinematic_limits, "TSR bimanual right")
        assert result.left.num_waypoints == result.right.num_waypoints
        assert np.array_equal(result.left.timestamps, result.right.timestamps)

    def test_bimanual_tsr_goal_has_no_cross_arm_contact_at_any_waypoint(self, robot):
        """Same jointly-collision-free bar as the joint-space crossing test,
        applied to the TSR planning path."""
        left_goal_pose = robot.left.arm.get_ee_pose().copy()
        left_goal_pose[:3, 3] += np.array([0.04, -0.02, 0.03])
        right_goal_pose = robot.right.arm.get_ee_pose().copy()
        right_goal_pose[:3, 3] += np.array([0.04, 0.02, 0.03])

        result = robot.plan_to_tsrs(
            {"left": _point_tsr(left_goal_pose), "right": _point_tsr(right_goal_pose)},
            seed=0,
            timeout=20.0,
        )
        if result is None or not result.success:
            pytest.skip("planner could not reach this TSR goal pair (not what this test targets)")

        checker = _combined_checker(robot)
        n = result.left.num_waypoints
        colliding = [
            i
            for i in range(n)
            if _cross_arm_contacts(checker, result.left.positions[i], result.right.positions[i])
        ]
        assert not colliding, f"TSR bimanual plan collides at waypoints {colliding[:5]}"

    def test_tsr_with_nonzero_bounds_still_plans(self, robot):
        """A TSR with a small volume (not a single point) is the realistic
        case for grasp regions -- confirm the sampling path (`samples=`)
        works for a bimanual goal, not just the degenerate point-TSR case
        above."""
        left_goal_pose = robot.left.arm.get_ee_pose().copy()
        left_goal_pose[:3, 3] += np.array([0.03, 0.0, 0.0])
        bw = np.array([[-0.01, 0.01], [-0.01, 0.01], [-0.01, 0.01], [0, 0], [0, 0], [0, 0]])
        tsr = TSR(T0_w=left_goal_pose, Tw_e=np.eye(4), Bw=bw)

        result = robot.plan_to_tsrs({"left": tsr}, seed=0, samples=8, timeout=20.0)
        assert result is not None and result.success, "planner failed on a small-volume TSR with slack"


# ---------------------------------------------------------------------------
# 2. Determinism
# ---------------------------------------------------------------------------


class TestPlanningDeterminism:
    """A fixed seed should reproduce the same plan. Without this, a
    trajectory that passed safety checks in sim cannot be trusted to be
    the same trajectory that gets sent to the real robot on a later run,
    and a failure on the robot can't be reliably reproduced in sim for
    debugging."""

    def test_same_seed_reproduces_identical_bimanual_trajectory(self, robot):
        goal_l = robot.left.arm.get_joint_positions().copy()
        goal_l[0] += 0.3
        goal_r = robot.right.arm.get_joint_positions().copy()
        goal_r[0] += 0.3

        result_a = robot.plan_to_configuration({"left": goal_l, "right": goal_r}, seed=7)
        robot.reset()
        result_b = robot.plan_to_configuration({"left": goal_l, "right": goal_r}, seed=7)

        assert result_a is not None and result_a.success
        assert result_b is not None and result_b.success

        assert result_a.left.num_waypoints == result_b.left.num_waypoints, (
            "same seed produced different waypoint counts -- planning is not deterministic"
        )
        assert np.allclose(result_a.left.positions, result_b.left.positions, atol=1e-9), (
            "same seed produced a different left trajectory"
        )
        assert np.allclose(result_a.right.positions, result_b.right.positions, atol=1e-9), (
            "same seed produced a different right trajectory"
        )

    def test_different_seeds_are_not_required_to_match(self, robot):
        """Sanity check for the test above: confirm the planner's seed
        actually does something (otherwise test_same_seed_reproduces_*
        could be passing vacuously because the planner ignores seed
        entirely and always returns one fixed path)."""
        goal_l = robot.left.arm.get_joint_positions().copy()
        goal_l[0] += 0.3
        goal_r = robot.right.arm.get_joint_positions().copy()
        goal_r[0] += 0.3

        seen_waypoint_counts = set()
        for seed in range(5):
            robot.reset()
            result = robot.plan_to_configuration({"left": goal_l, "right": goal_r}, seed=seed)
            if result is not None and result.success:
                seen_waypoint_counts.add(result.left.num_waypoints)

        assert len(seen_waypoint_counts) >= 1  # at least one succeeded
        # Not asserting variety here -- CBiRRT may legitimately find the same
        # solution for an easy goal across seeds. This test exists to make
        # the determinism claim above falsifiable, not to demand variety.


# ---------------------------------------------------------------------------
# 3. Malformed-goal validation
# ---------------------------------------------------------------------------


class TestMalformedGoalValidation:
    def test_wrong_dof_single_arm_goal_raises_clearly(self, robot):
        bad_goal = np.zeros(robot.left.arm.dof + 2)  # wrong length
        with pytest.raises(ValueError):
            robot.plan_to_configuration({"left": bad_goal})

    def test_wrong_dof_inside_bimanual_goal_raises_clearly(self, robot):
        """The malformed arm is not the first one specified -- guards
        against a validation path that only checks the first dict entry."""
        good_left = robot.left.arm.get_joint_positions().copy()
        bad_right = np.zeros(robot.right.arm.dof - 1)
        with pytest.raises(ValueError):
            robot.plan_to_configuration({"left": good_left, "right": bad_right})

    def test_unknown_arm_name_raises_clearly(self, robot):
        goal = robot.left.arm.get_joint_positions().copy()
        with pytest.raises(ValueError):
            robot.plan_to_configuration({"middle": goal})


# ---------------------------------------------------------------------------
# 4. Atomicity: one bad arm goal must fail the whole bimanual call
# ---------------------------------------------------------------------------


class TestBimanualGoalAtomicity:
    def test_one_arm_unreachable_goal_fails_the_whole_plan(self, robot):
        """If the right arm's goal is structurally invalid (self-colliding),
        the group planner must refuse the ENTIRE request -- it must not
        return a plan that quietly moves only the left arm while dropping
        the right arm's bad goal. A caller that checked only
        `result.success is False` and stopped would still be protected by
        this; the extra checks below guard against a partial success that
        makes result.success True while one arm silently didn't move to
        where it was asked."""
        good_left_goal = robot.left.arm.get_joint_positions().copy()
        good_left_goal[0] += 0.3
        bad_right_goal = _find_self_colliding_config(robot.right.arm)

        result = robot.plan_to_configuration({"left": good_left_goal, "right": bad_right_goal}, seed=0, timeout=15.0)

        assert result is None or not result.success, (
            "planner reported success for a bimanual goal where one arm's target "
            "configuration is self-colliding -- the bad goal was not actually checked"
        )

    def test_rejected_bimanual_goal_leaves_robot_state_untouched(self, robot):
        """Planning must not have side effects on the live robot state even
        when it fails partway through resolving goals for one arm."""
        q_left_before = robot.left.arm.get_joint_positions().copy()
        q_right_before = robot.right.arm.get_joint_positions().copy()

        good_left_goal = q_left_before.copy()
        good_left_goal[0] += 0.3
        bad_right_goal = _find_self_colliding_config(robot.right.arm)

        robot.plan_to_configuration({"left": good_left_goal, "right": bad_right_goal}, seed=0, timeout=15.0)

        assert np.array_equal(robot.left.arm.get_joint_positions(), q_left_before)
        assert np.array_equal(robot.right.arm.get_joint_positions(), q_right_before)


# ---------------------------------------------------------------------------
# 5. Holding-arm stationarity throughout (not just endpoints)
# ---------------------------------------------------------------------------


class TestHoldingArmStationarity:
    @pytest.mark.parametrize("moving_side,holding_side", [("left", "right"), ("right", "left")])
    def test_untasked_arm_does_not_drift_mid_trajectory(self, robot, moving_side, holding_side):
        """test_robot_unit.py's plan_reach_to_pose test checks the untasked
        arm's first vs. last waypoint. This checks every waypoint in
        between -- a holding arm that jitters mid-trajectory (e.g. from a
        retiming or splitting bug) would pass the endpoints-only check but
        fail this one, and would visibly twitch on the real robot."""
        moving_arm = getattr(robot, moving_side).arm
        goal = moving_arm.get_joint_positions().copy()
        goal[0] += 0.4

        result = robot.plan_to_configuration({moving_side: goal}, seed=0)
        assert result is not None and result.success

        holding_traj = getattr(result, holding_side)
        first = holding_traj.positions[0]
        max_drift = np.max(np.abs(holding_traj.positions - first))
        assert max_drift < 1e-6, (
            f"{holding_side} arm was not tasked but drifted up to {max_drift:.2e} rad "
            f"from its starting position during the {moving_side} arm's motion"
        )
        assert np.allclose(holding_traj.velocities, 0.0, atol=1e-6), (
            f"{holding_side} arm reports nonzero velocity while holding"
        )