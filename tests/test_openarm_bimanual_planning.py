"""
Conclusive test for bimanual (two-arm-aware) motion planning on Openarm.

WHAT "BIMANUAL SUPPORT" MEANS HERE
----------------------------------
The bar this file sets is deliberately strict: it is NOT enough for the robot
to move both arms. The *current* `Openarm.go_to()` already moves both arms --
but it does so by looping over them and calling `arm.plan_to_configuration()`
independently (see robot.py). Each single-arm plan validates against a frozen
snapshot of the other arm, so two arms whose paths cross in time can still
collide. That is precisely the gap a real bimanual planner closes.

So "bimanual support" is defined here as:

    A single planning call that reasons about BOTH arms at once and returns a
    pair of time-synchronized trajectories which are collision-free at every
    shared timestep -- INCLUDING left-arm-vs-right-arm collisions -- even for a
    goal pair whose naive independent (straight-line) execution would drive the
    two arms through each other.

The centerpiece is `test_crossing_goal_is_solved_without_arm_arm_collision`:
it constructs an adversarial start/goal pair, first PROVES the scenario is a
genuine trap (independent straight-line motion collides), then requires the
bimanual planner to return a jointly-collision-free solution. A merely
"move both arms" implementation cannot pass it; only a planner that accounts
for the other arm's motion can.

TARGET API (as designed -- implement to make these pass)
--------------------------------------------------------
    result = robot.plan_bimanual(q_left_goal, q_right_goal, seed=..., timeout=...)

    result is None            -> planning failed
    result.left  : Trajectory -> left arm's joint trajectory
    result.right : Trajectory -> right arm's joint trajectory
    (both share one time parameterization: equal duration + waypoint count)

If `plan_bimanual` doesn't exist yet, the whole module skips (so it sits green-
pending in CI until you build it), rather than erroring. Delete the skip guard
in `bimanual` once the method lands to turn these into hard requirements.

FORCING A REAL CROSS-ARM COLLISION
----------------------------------
Like the sibling collision-detection test, we avoid relying on incidental URDF
geometry. We drive one arm to a configuration that reaches deep into the
other arm's workspace and confirm, via the planner's own collision checker,
that the crossing configuration genuinely interpenetrates. If the two arms
cannot be made to collide in this model at all (e.g. contype/conaffinity or an
over-broad <exclude> prevents cross-arm contacts), the adversarial-scenario
assertion fails loudly -- which is the correct signal, because in that case no
planner can be said to be avoiding a collision it can't even perceive.
"""

from __future__ import annotations

import numpy as np
import pytest

from openarm.robot import Openarm
from openarm.config import OpenarmConfig


# ---------------------------------------------------------------------------
# Fixtures (mirrors tests/test_openarm_collision_detection.py conventions)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def robot():
    """A real Openarm instance. Skips the module if the model can't be
    resolved in this environment, consistent with the other integration
    tests."""
    try:
        return Openarm(config=OpenarmConfig.default())
    except FileNotFoundError as e:
        pytest.skip(
            f"Openarm model not available in this environment: {e}",
            allow_module_level=True,
        )


@pytest.fixture(autouse=True)
def _reset_robot(robot):
    """Known state before every test, so joint manipulation doesn't leak."""
    robot.reset()
    yield


@pytest.fixture
def bimanual(robot):
    """Skip any test that needs the bimanual planner until it exists.

    Turns this module into a green-pending target: the tests are fully
    written, but don't fail CI on `AttributeError` before the feature lands.
    Once `plan_bimanual` is implemented, this guard simply stops skipping.
    """
    if not hasattr(robot, "plan_bimanual"):
        pytest.skip("robot.plan_bimanual not implemented yet -- bimanual planner pending")
    return robot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _combined_checker(robot):
    """Build a collision checker that owns BOTH arms' joints, so it reports
    left-vs-right contacts as well as each arm vs the environment.

    This is the ground-truth oracle the tests judge trajectories against.
    It is intentionally constructed independently of `plan_bimanual` so the
    test cannot pass just because the planner and the oracle share a bug: we
    build the checker straight from the collision module against a fork of the
    live scene.
    """
    from mj_manipulator.collision import CollisionChecker

    left = robot.left
    right = robot.right

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
    ), (left.dof, right.dof)


def _is_pair_collision_free(robot, q_left, q_right) -> bool:
    """True iff the combined (14-DOF) configuration is collision-free,
    left-vs-right included."""
    checker, _ = _combined_checker(robot)
    q = np.concatenate([np.asarray(q_left), np.asarray(q_right)])
    return checker.is_valid(q)


def _straight_line(q_start, q_goal, n=25):
    """Naive per-arm straight-line joint interpolation -- what independent
    single-arm execution effectively does between two configs."""
    q_start = np.asarray(q_start)
    q_goal = np.asarray(q_goal)
    return [q_start + (q_goal - q_start) * t for t in np.linspace(0.0, 1.0, n)]


def _reach_into_other_workspace(robot):
    """Return (qL_goal, qR_goal): goals that send the left arm toward the
    right side and the right arm toward the left side, so their straight-line
    paths overlap in space and time.

    Strategy that doesn't hard-code joint values for a specific URDF: take
    each arm's current 'ready' config and mirror the base (shoulder-yaw) joint
    toward the opposite side, pushing each arm across the centerline. If your
    joint-0 sign convention differs, flip the signs below; the adversarial-
    scenario assertion in the crossing test will tell you immediately if the
    goals don't actually induce a crossing.
    """
    qL0 = np.asarray(robot.left.get_joint_positions(), dtype=float)
    qR0 = np.asarray(robot.right.get_joint_positions(), dtype=float)

    loL, hiL = robot.left.get_joint_limits()
    loR, hiR = robot.right.get_joint_limits()

    qL_goal = qL0.copy()
    qR_goal = qR0.copy()

    # Push each arm's shoulder-yaw toward the other side (~60% toward the limit
    # that points across the body), driving the two arms into a shared region.
    qL_goal[0] = qL0[0] + 0.6 * (hiL[0] - qL0[0])
    qR_goal[0] = qR0[0] + 0.6 * (loR[0] - qR0[0])

    # Also reach forward/inward with the elbow so the links (not just the base)
    # sweep through the overlap region.
    if len(qL_goal) > 3:
        qL_goal[3] = qL0[3] + 0.5 * (hiL[3] - qL0[3])
        qR_goal[3] = qR0[3] + 0.5 * (hiR[3] - qR0[3])

    return qL_goal, qR_goal


# ---------------------------------------------------------------------------
# API-shape tests (cheap, deterministic)
# ---------------------------------------------------------------------------


def test_plan_bimanual_exists(robot):
    """The bimanual entry point exists and is callable. This is the one test
    that does NOT use the `bimanual` skip fixture -- it is the signal that the
    feature has landed at all."""
    assert hasattr(robot, "plan_bimanual"), (
        "Openarm must expose plan_bimanual(q_left_goal, q_right_goal, ...) as the "
        "single-call bimanual planning entry point."
    )
    assert callable(robot.plan_bimanual)


def test_result_has_two_synchronized_trajectories(bimanual):
    """A successful plan returns a left and a right trajectory that share one
    time parameterization -- equal duration and equal waypoint count -- so the
    two arms execute in lockstep."""
    robot = bimanual
    qL_goal, qR_goal = _reach_into_other_workspace(robot)

    result = robot.plan_bimanual(qL_goal, qR_goal, seed=0)
    assert result is not None, "planner failed on a reachable goal pair (try a different seed/timeout)"

    assert hasattr(result, "left") and hasattr(result, "right"), (
        "BimanualPlanResult must expose .left and .right trajectories"
    )

    left, right = result.left, result.right
    assert left is not None and right is not None

    assert left.num_waypoints == right.num_waypoints, (
        f"arms must share a time parameterization: "
        f"{left.num_waypoints} vs {right.num_waypoints} waypoints"
    )
    assert left.dof == robot.left.dof
    assert right.dof == robot.right.dof
    assert np.isclose(left.duration, right.duration), (
        f"arms must share a time base: {left.duration}s vs {right.duration}s"
    )


def test_trajectories_are_tagged_per_arm(bimanual):
    """Each returned trajectory should identify which arm it belongs to, so
    the executor can route them. Uses Trajectory.entity, which already exists."""
    robot = bimanual
    qL_goal, qR_goal = _reach_into_other_workspace(robot)
    result = robot.plan_bimanual(qL_goal, qR_goal, seed=0)
    assert result is not None

    entities = {result.left.entity, result.right.entity}
    assert None not in entities, "both trajectories must be tagged with an entity"
    assert len(entities) == 2, f"left and right must have distinct entity tags, got {entities}"


# ---------------------------------------------------------------------------
# THE conclusive test: joint collision avoidance between the two arms
# ---------------------------------------------------------------------------


def test_crossing_goal_is_solved_without_arm_arm_collision(bimanual):
    """The core proof of bimanual support.

    1. Build an adversarial goal pair that sends each arm across the
       centerline into the other's workspace.
    2. PROVE it's a real trap: the naive independent straight-line execution
       (what looping single-arm plans effectively produces) passes through at
       least one configuration where the two arms collide. If this assertion
       fails, the scenario isn't adversarial in this model -- adjust the goals
       in `_reach_into_other_workspace` until it bites.
    3. REQUIRE the bimanual planner to return a solution whose EVERY shared
       timestep is collision-free under the combined (both-arms) checker.

    Step 2 is what makes this conclusive. A "move both arms independently"
    implementation cannot satisfy step 3 for a scenario that step 2 has shown
    is a genuine crossing -- it has no mechanism to see the other arm move.
    """
    robot = bimanual
    qL_start = np.asarray(robot.left.get_joint_positions(), dtype=float)
    qR_start = np.asarray(robot.right.get_joint_positions(), dtype=float)
    qL_goal, qR_goal = _reach_into_other_workspace(robot)

    # Sanity: start and both goals are themselves individually valid, so any
    # failure is about the CROSSING, not an unreachable/self-colliding endpoint.
    assert _is_pair_collision_free(robot, qL_start, qR_start), (
        "start configuration already reports an arm-arm collision -- fixture/reset issue"
    )
    assert _is_pair_collision_free(robot, qL_goal, qR_goal), (
        "goal pair itself collides; pick goals whose final pose is clear so the test "
        "isolates the crossing, not the endpoint"
    )

    # --- Step 2: prove the naive independent execution actually collides. ---
    left_line = _straight_line(qL_start, qL_goal)
    right_line = _straight_line(qR_start, qR_goal)
    naive_collides = any(
        not _is_pair_collision_free(robot, qL, qR)
        for qL, qR in zip(left_line, right_line)
    )
    assert naive_collides, (
        "Adversarial scenario is not adversarial in this model: independent "
        "straight-line execution of the two goals never collides, so this test "
        "cannot distinguish a real bimanual planner from an independent one. "
        "Strengthen the goals in _reach_into_other_workspace (or verify the MJCF "
        "actually generates left-vs-right contacts: contype/conaffinity and no "
        "cross-arm <exclude>)."
    )

    # --- Step 3: the bimanual planner must avoid that collision. ---
    result = robot.plan_bimanual(qL_goal, qR_goal, seed=0, timeout=30.0)
    assert result is not None, (
        "bimanual planner returned no path for a goal pair with a known "
        "collision-free solution region -- it should find one"
    )

    # Reconstruct synchronized waypoints and check EVERY shared timestep.
    n = result.left.num_waypoints
    assert result.right.num_waypoints == n

    ts = np.linspace(0.0, result.left.duration, n)
    colliding_steps = []
    for i, t in enumerate(ts):
        qL, _, _ = result.left.sample(t)
        qR, _, _ = result.right.sample(t)
        if not _is_pair_collision_free(robot, qL, qR):
            colliding_steps.append(i)

    assert not colliding_steps, (
        f"bimanual plan collides at {len(colliding_steps)} of {n} synchronized "
        f"timesteps (indices {colliding_steps[:5]}{'...' if len(colliding_steps) > 5 else ''}). "
        "The two arms are not being planned jointly."
    )

    # And it actually reaches both goals.
    qL_end, _, _ = result.left.sample(result.left.duration)
    qR_end, _, _ = result.right.sample(result.right.duration)
    assert np.allclose(qL_end, qL_goal, atol=1e-2), "left arm did not reach its goal"
    assert np.allclose(qR_end, qR_goal, atol=1e-2), "right arm did not reach its goal"


def test_endpoints_are_reached_on_a_clear_goal(bimanual):
    """A non-adversarial pair (both arms move within their own workspace) should
    also plan and reach goals -- guards against a planner that only ever
    succeeds by refusing to move."""
    robot = bimanual
    qL0 = np.asarray(robot.left.get_joint_positions(), dtype=float)
    qR0 = np.asarray(robot.right.get_joint_positions(), dtype=float)

    loL, hiL = robot.left.get_joint_limits()
    loR, hiR = robot.right.get_joint_limits()

    # Small, safe elbow move outward for each arm -- no crossing.
    qL_goal = qL0.copy()
    qR_goal = qR0.copy()
    if len(qL_goal) > 3:
        qL_goal[3] = qL0[3] + 0.2 * (loL[3] - qL0[3])
        qR_goal[3] = qR0[3] + 0.2 * (loR[3] - qR0[3])

    result = robot.plan_bimanual(qL_goal, qR_goal, seed=0)
    assert result is not None, "planner failed on an easy, clear goal pair"

    qL_end, _, _ = result.left.sample(result.left.duration)
    qR_end, _, _ = result.right.sample(result.right.duration)
    assert np.allclose(qL_end, qL_goal, atol=1e-2)
    assert np.allclose(qR_end, qR_goal, atol=1e-2)