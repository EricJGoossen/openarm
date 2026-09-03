"""
GATE FILE -- collision safety for real-hardware execution.

Two collision guarantees matter for running bimanual plans on the real
robot, and neither is checked anywhere else in this test suite:

1. ARM-VS-ARM: a plan that moves both arms at once must never let them
   touch each other at any *synchronized* timestep, even when each arm's
   own path is individually collision-free at its endpoints.
2. ARM-VS-ENVIRONMENT: a plan must never drive an arm through a fixed
   obstacle in the scene, even when the start and goal configurations are
   both clear of it.

Both checks are done with an oracle `CollisionChecker` built independently
from the planner (fresh model fork, no planner internals reused), so a
shared bug between planner and checker can't make these tests pass
vacuously -- same approach as
`test_openarm_bimanual_planning.py::_combined_checker`.

If these fail, the corresponding plan is telling the arms to occupy the
same space as each other or as something fixed in the cell -- do not
execute it on real hardware.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mj_manipulator.collision import CollisionChecker
from openarm.config import OpenarmConfig
from openarm.robot import Openarm

from _obstacle_model import (
    OBSTACLE_POS,
    OBSTACLE_RADIUS,
    RIGHT_ARM_OBSTACLE_GOAL_SCALE,
    build_obstacle_model_xml,
)

N_RANDOM_PAIRS = 12


# ---------------------------------------------------------------------------
# Fixtures
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
    yield


@pytest.fixture(scope="module")
def robot_with_obstacle(tmp_path_factory):
    """A second Openarm instance loaded from a model with one static
    obstacle sphere added to the scene (see _obstacle_model.py)."""
    model_path = build_obstacle_model_xml(str(tmp_path_factory.mktemp("obstacle_model") / "bimanual_obstacle.xml"))
    cfg = OpenarmConfig.default()
    cfg.model_path = Path(model_path)
    try:
        r = Openarm(config=cfg)
    except FileNotFoundError as e:
        pytest.skip(f"Openarm model not available in this environment: {e}", allow_module_level=True)
    r.reset()
    return r


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _combined_checker(robot):
    """Independent oracle collision checker covering both arms, built
    fresh from a fork of the live scene (not reused planner state)."""
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


def _arm_side(body_name: str) -> str | None:
    if body_name.startswith("openarm_left_"):
        return "left"
    if body_name.startswith("openarm_right_"):
        return "right"
    return None


def _cross_arm_contacts(checker, q_left, q_right):
    q = np.concatenate([np.asarray(q_left), np.asarray(q_right)])
    return [
        (b1, b2, depth)
        for b1, b2, depth in checker.get_contacts(q)
        if {_arm_side(b1), _arm_side(b2)} == {"left", "right"}
    ]


def _random_goal(rng, lo, hi, margin=0.05):
    lo, hi = np.asarray(lo), np.asarray(hi)
    span = hi - lo
    return lo + margin * span + rng.random(len(lo)) * span * (1 - 2 * margin)


# ---------------------------------------------------------------------------
# Arm-vs-arm: fuzz across many random bimanual goal pairs
# ---------------------------------------------------------------------------


class TestArmVsArmCollision:
    @pytest.mark.parametrize("seed", range(N_RANDOM_PAIRS))
    def test_random_bimanual_plan_has_no_cross_arm_contact_at_any_waypoint(self, robot, seed):
        """Sample a random reachable goal for each arm and check EVERY
        synchronized waypoint of the resulting plan, not just the
        endpoints. This is deliberately not one hand-crafted adversarial
        pair (see test_openarm_bimanual_planning.py for that) -- it's a
        broad fuzz across many combinations, closer to the range of goals
        a real task sequence would actually send."""
        rng = np.random.default_rng(1000 + seed)
        lo_l, hi_l = robot.left.arm.get_joint_limits()
        lo_r, hi_r = robot.right.arm.get_joint_limits()
        goal_l = _random_goal(rng, lo_l, hi_l)
        goal_r = _random_goal(rng, lo_r, hi_r)

        result = robot.plan_to_configuration({"left": goal_l, "right": goal_r}, seed=seed, timeout=20.0)
        if result is None or not result.success:
            pytest.skip("planner could not reach this random goal pair")

        checker = _combined_checker(robot)
        n = result.left.num_waypoints
        assert n == result.right.num_waypoints

        colliding_waypoints = []
        for i in range(n):
            contacts = _cross_arm_contacts(checker, result.left.positions[i], result.right.positions[i])
            if contacts:
                colliding_waypoints.append((i, contacts))

        assert not colliding_waypoints, (
            f"seed={seed}: {len(colliding_waypoints)}/{n} synchronized waypoints have a "
            f"left-vs-right contact, e.g. at waypoint {colliding_waypoints[0][0]}: "
            f"{colliding_waypoints[0][1]}"
        )

    def test_goal_and_ready_pose_are_self_consistent(self, robot):
        """Regression guard for the oracle itself: the all-zeros ready
        pose must read as collision-free, or every test above is
        comparing against a broken baseline."""
        checker = _combined_checker(robot)
        q_ready_l = robot.left.arm.get_joint_positions()
        q_ready_r = robot.right.arm.get_joint_positions()
        assert _cross_arm_contacts(checker, q_ready_l, q_ready_r) == []


# ---------------------------------------------------------------------------
# Arm-vs-environment: static obstacle in the scene
# ---------------------------------------------------------------------------


class TestArmVsEnvironmentCollision:
    def _obstacle_goal(self, robot_with_obstacle):
        lo, hi = robot_with_obstacle.right.arm.get_joint_limits()
        return np.array(lo) * RIGHT_ARM_OBSTACLE_GOAL_SCALE

    def test_obstacle_is_absent_at_endpoints_present_on_naive_path(self, robot_with_obstacle):
        """Confirms this is a genuine trap before trusting the planner
        result below: the obstacle must NOT be in contact at the ready
        pose or the goal (so a planner that just checked endpoints would
        wrongly call this safe), but naive straight-line interpolation
        between them MUST clip it at least once."""
        right = robot_with_obstacle.right.arm
        q_ready = right.get_joint_positions()
        q_goal = self._obstacle_goal(robot_with_obstacle)

        env = robot_with_obstacle.env.fork()
        checker = CollisionChecker(
            model=env.model,
            data=env.data,
            joint_names=list(right.config.joint_names),
        )

        def obstacle_hit(q):
            return any(
                "test_obstacle" in b1 or "test_obstacle" in b2 for b1, b2, _ in checker.get_contacts(np.asarray(q))
            )

        assert not obstacle_hit(q_ready), "test setup invalid: obstacle already touches the ready pose"
        assert not obstacle_hit(q_goal), "test setup invalid: obstacle already touches the goal"

        naive_path = [q_ready + (q_goal - q_ready) * t for t in np.linspace(0.0, 1.0, 25)]
        assert any(obstacle_hit(q) for q in naive_path), (
            "test setup invalid: naive straight-line motion never touches the obstacle, "
            "so this isn't testing obstacle avoidance at all -- re-derive OBSTACLE_POS/"
            "OBSTACLE_RADIUS in _obstacle_model.py"
        )

    def test_planner_avoids_obstacle_at_every_waypoint(self, robot_with_obstacle):
        """The actual safety guarantee: given the trap confirmed above,
        every waypoint of the planned (not naive) trajectory must be clear
        of the obstacle -- not just the two endpoints."""
        right = robot_with_obstacle.right.arm
        q_goal = self._obstacle_goal(robot_with_obstacle)

        env = robot_with_obstacle.env.fork()
        checker = CollisionChecker(
            model=env.model,
            data=env.data,
            joint_names=list(right.config.joint_names),
        )

        failures = []
        for seed in range(5):
            result = robot_with_obstacle.plan_to_configuration({"right": q_goal}, seed=seed)
            if result is None or not result.success:
                continue
            traj = result.right
            for i in range(traj.num_waypoints):
                contacts = [
                    c for c in checker.get_contacts(traj.positions[i]) if "test_obstacle" in c[0] or "test_obstacle" in c[1]
                ]
                if contacts:
                    failures.append((seed, i, contacts))

        assert failures == [], (
            f"planned trajectory intersects the static obstacle at "
            f"{len(failures)} (seed, waypoint) point(s), e.g. seed={failures[0][0]} "
            f"waypoint={failures[0][1]}: {failures[0][2]}"
        )

    def test_obstacle_model_matches_expected_geometry(self, robot_with_obstacle):
        """Sanity check that the injected obstacle actually landed where
        _obstacle_model.py says it should -- if the generated model's
        layout changes, this fails loudly instead of the tests above
        silently testing nothing."""
        model = robot_with_obstacle.env.model
        import mujoco

        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "test_obstacle")
        assert body_id >= 0, "test_obstacle body was not added to the model"
        pos = model.body_pos[body_id]
        assert np.allclose(pos, OBSTACLE_POS, atol=1e-6)

        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "test_obstacle_geom")
        assert geom_id >= 0
        assert np.isclose(model.geom_size[geom_id][0], OBSTACLE_RADIUS)