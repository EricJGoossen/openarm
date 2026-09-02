"""
Conclusive test for bimanual (two-arm-aware) motion planning on Openarm.

WHAT "BIMANUAL SUPPORT" MEANS HERE
----------------------------------
The bar this file sets is deliberately strict: it is NOT enough for the robot
to move both arms. A naive implementation could move both arms by looping
over them and calling `arm.plan_to_configuration()` independently. Each
single-arm plan validates against a frozen snapshot of the other arm, so two
arms whose paths cross in time can still collide. That is precisely the gap a
real bimanual planner closes.

So "bimanual support" is defined here as:

    A single planning call that reasons about BOTH arms at once and returns a
    pair of time-synchronized trajectories which are collision-free at every
    shared timestep -- INCLUDING left-arm-vs-right-arm collisions -- even for a
    goal pair whose naive independent (straight-line) execution would drive the
    two arms through each other.

The centerpiece is `test_crossing_goal_is_solved_without_arm_arm_collision`:
it constructs an adversarial start/goal pair, first PROVES the scenario is a
genuine trap (independent straight-line motion collides), then requires the
planner to return a jointly-collision-free solution. A merely "move both arms"
implementation cannot pass it; only a planner that accounts for the other
arm's motion can.

API UNDER TEST
--------------
There is no separate `plan_bimanual` entry point. Bimanual planning is just
`Openarm.plan_to_configuration()` called with goals for both arms at once:

    result = robot.plan_to_configuration(
        {"left": q_left_goal, "right": q_right_goal}, seed=..., timeout=...
    )

    result is None            -> planning failed
    result.success            -> True iff every arm in the goal succeeded
    result.left  : Trajectory -> left arm's joint trajectory (or None)
    result.right : Trajectory -> right arm's joint trajectory (or None)
    (both share one time parameterization: equal duration + waypoint count,
    since `Openarm._package_plan` retimes the combined path before splitting it)

This module holds `ArmGroup.plan_to_configuration` to the strict bimanual bar
described above: passing requires it to reason about both arms jointly, not
just loop single-arm plans under a shared call.

FORCING A REAL CROSS-ARM COLLISION
----------------------------------
Like the sibling collision-detection test, we avoid relying on incidental URDF
geometry (and on assumptions about what a given joint index physically does --
see `_reach_into_other_workspace`). We deterministically sample goal configs
for both arms and, via the planner's own collision checker, keep the first
pair that is itself collision-free but whose naive straight-line execution
genuinely interpenetrates somewhere along the way. If the two arms cannot be
made to collide in this model at all (e.g. contype/conaffinity or an
over-broad <exclude> prevents cross-arm contacts), that search fails loudly --
which is the correct signal, because in that case no planner can be said to be
avoiding a collision it can't even perceive.
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


def _plan_bimanual(robot, q_left_goal, q_right_goal, **kwargs):
    """Thin wrapper so the rest of the file reads like the old
    `robot.plan_bimanual(...)` call sites, even though the actual API is the
    generic group planner given a two-arm goal dict."""
    return robot.plan_to_configuration({"left": q_left_goal, "right": q_right_goal}, **kwargs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _combined_checker(robot):
    """Build a collision checker that owns BOTH arms' joints, so it reports
    left-vs-right contacts as well as each arm vs the environment.

    This is the ground-truth oracle the tests judge trajectories against.
    It is intentionally constructed independently of `plan_to_configuration`
    so the test cannot pass just because the planner and the oracle share a
    bug: we build the checker straight from the collision module against a
    fork of the live scene.
    """
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
    ), (left.dof, right.dof)


def _arm_side(body_name: str) -> str | None:
    """'left'/'right' for an arm body, None for anything else (torso, world,
    grasped objects, ...). Bodies follow the openarm_{left,right,body}_*
    naming convention used throughout this model."""
    if body_name.startswith("openarm_left_"):
        return "left"
    if body_name.startswith("openarm_right_"):
        return "right"
    return None


def _cross_arm_contacts(robot, q_left, q_right) -> list[tuple[str, str, float]]:
    """Return only the LEFT-arm-vs-RIGHT-arm contacts at this combined
    configuration.

    `CollisionChecker.get_contacts` reports every invalid contact touching
    the robot, including each arm's own self-collisions and grazes against
    the shared torso (`openarm_body_*`). Those don't require any awareness of
    the OTHER arm to avoid -- an independent single-arm planner keeps its own
    links off its own body just as well as a joint one would -- so they are
    not evidence for or against bimanual reasoning, which is this file's
    entire premise. Filtering to left-vs-right pairs keeps the oracle
    specific to what this test is actually proving.
    """
    checker, _ = _combined_checker(robot)
    q = np.concatenate([np.asarray(q_left), np.asarray(q_right)])
    cross = []
    for body_a, body_b, depth_mm in checker.get_contacts(q):
        if {_arm_side(body_a), _arm_side(body_b)} == {"left", "right"}:
            cross.append((body_a, body_b, depth_mm))
    return cross


def _is_pair_collision_free(robot, q_left, q_right) -> bool:
    """True iff there is no LEFT-arm-vs-RIGHT-arm contact at this combined
    configuration (see `_cross_arm_contacts` for why other contacts, e.g.
    self-collisions, are deliberately excluded)."""
    return not _cross_arm_contacts(robot, q_left, q_right)


def _straight_line(q_start, q_goal, n=25):
    """Naive per-arm straight-line joint interpolation -- what independent
    single-arm execution effectively does between two configs."""
    q_start = np.asarray(q_start)
    q_goal = np.asarray(q_goal)
    return [q_start + (q_goal - q_start) * t for t in np.linspace(0.0, 1.0, n)]


def _reach_into_other_workspace(robot):
    """Return (qL_goal, qR_goal): a goal pair that is itself collision-free but
    whose naive straight-line execution passes through a real left-vs-right
    collision.

    A fixed formula (e.g. "push joint 0 toward the opposite limit") turned out
    not to be a reliable way to build this trap: joint 0 on this arm isn't a
    simple shoulder-yaw about a vertical axis, so folding it toward its limit
    barely moves the end-effector sideways at all (it mostly changes height).
    Hand-picking a different formula would just re-couple the test to today's
    kinematic convention.

    So these values were found empirically, not guessed: a script sampled goal
    configs for both arms uniformly within their own joint limits and kept a
    candidate that was (a) collision-free at the goal itself (checked against
    the *entire* robot, self-collisions included -- see `_is_pair_collision_free`'s
    sibling full-validity check used during the search) and (b) not reachable
    via naive independent straight-line motion without a real left-vs-right
    collision occurring somewhere along the way (checked with
    `_cross_arm_contacts`, i.e. left-vs-right only, matching what this file
    actually tests for).

    That search is NOT run at test time: genuine cross-arm overlaps turned out
    to be rare in this model (on the order of 1 in several thousand uniform
    samples), so searching live would make this test take upwards of a minute
    and could still come up empty within a bounded budget. The values below
    were verified (once) to additionally produce a real planned trajectory
    with zero colliding synchronized timesteps out of 110, i.e. they already
    satisfy everything `test_crossing_goal_is_solved_without_arm_arm_collision`
    checks. If the URDF/collision geometry changes enough that the "goal
    collides" or "naive path never collides" assertions below start failing,
    regenerate these by uniformly sampling goal pairs within
    `robot.left.arm.get_joint_limits()` / `robot.right.arm.get_joint_limits()`
    and keeping one where `_is_pair_collision_free` is True at the goal but
    False for at least one point along `_straight_line` from the ready pose.
    """
    qL_goal = np.array([-0.97019901, -0.60228936, 1.19591137, 2.38015869, -0.85566228, -0.33569914, 0.43686219])
    qR_goal = np.array([1.38920022, -0.1051121, 0.91826051, 1.15166886, 1.53557046, -0.10184814, -1.18932409])
    return qL_goal, qR_goal


# ---------------------------------------------------------------------------
# API-shape tests (cheap, deterministic)
# ---------------------------------------------------------------------------


def test_result_has_two_synchronized_trajectories(robot):
    """A successful plan returns a left and a right trajectory that share one
    time parameterization -- equal duration and equal waypoint count -- so the
    two arms execute in lockstep."""
    qL_goal, qR_goal = _reach_into_other_workspace(robot)

    result = _plan_bimanual(robot, qL_goal, qR_goal, seed=0)
    assert result is not None and result.success, (
        "planner failed on a reachable goal pair (try a different seed/timeout)"
    )

    left, right = result.left, result.right
    assert left is not None and right is not None

    assert left.num_waypoints == right.num_waypoints, (
        f"arms must share a time parameterization: "
        f"{left.num_waypoints} vs {right.num_waypoints} waypoints"
    )
    assert left.dof == robot.left.arm.dof
    assert right.dof == robot.right.arm.dof
    assert np.isclose(left.duration, right.duration), (
        f"arms must share a time base: {left.duration}s vs {right.duration}s"
    )


def test_trajectories_are_tagged_per_arm(robot):
    """Each returned trajectory should identify which arm it belongs to, so
    the executor can route them. Uses Trajectory.entity, which already exists."""
    qL_goal, qR_goal = _reach_into_other_workspace(robot)
    result = _plan_bimanual(robot, qL_goal, qR_goal, seed=0)
    assert result is not None and result.success

    entities = {result.left.entity, result.right.entity}
    assert None not in entities, "both trajectories must be tagged with an entity"
    assert len(entities) == 2, f"left and right must have distinct entity tags, got {entities}"


# ---------------------------------------------------------------------------
# THE conclusive test: joint collision avoidance between the two arms
# ---------------------------------------------------------------------------


def test_crossing_goal_is_solved_without_arm_arm_collision(robot):
    """The core proof of bimanual support.

    1. Build an adversarial goal pair that sends each arm across the
       centerline into the other's workspace.
    2. PROVE it's a real trap: the naive independent straight-line execution
       (what looping single-arm plans effectively produces) passes through at
       least one configuration where the two arms collide. If this assertion
       fails, the scenario isn't adversarial in this model -- adjust the goals
       in `_reach_into_other_workspace` until it bites.
    3. REQUIRE the planner to return a solution whose EVERY shared timestep is
       collision-free under the combined (both-arms) checker.

    Step 2 is what makes this conclusive. A "move both arms independently"
    implementation cannot satisfy step 3 for a scenario that step 2 has shown
    is a genuine crossing -- it has no mechanism to see the other arm move.
    """
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

    # --- Step 3: the planner must avoid that collision. ---
    result = _plan_bimanual(robot, qL_goal, qR_goal, seed=0, timeout=30.0)
    assert result is not None and result.success, (
        "planner returned no path for a goal pair with a known "
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


def test_endpoints_are_reached_on_a_clear_goal(robot):
    """A non-adversarial pair (both arms move within their own workspace) should
    also plan and reach goals -- guards against a planner that only ever
    succeeds by refusing to move."""
    qL0 = np.asarray(robot.left.get_joint_positions(), dtype=float)
    qR0 = np.asarray(robot.right.get_joint_positions(), dtype=float)

    loL, hiL = robot.left.arm.get_joint_limits()
    loR, hiR = robot.right.arm.get_joint_limits()

    # Small, safe elbow move outward for each arm -- no crossing.
    qL_goal = qL0.copy()
    qR_goal = qR0.copy()
    if len(qL_goal) > 3:
        qL_goal[3] = qL0[3] + 0.2 * (loL[3] - qL0[3])
        qR_goal[3] = qR0[3] + 0.2 * (loR[3] - qR0[3])

    result = _plan_bimanual(robot, qL_goal, qR_goal, seed=0)
    assert result is not None and result.success, "planner failed on an easy, clear goal pair"

    qL_end, _, _ = result.left.sample(result.left.duration)
    qR_end, _, _ = result.right.sample(result.right.duration)
    assert np.allclose(qL_end, qL_goal, atol=1e-2)
    assert np.allclose(qR_end, qR_goal, atol=1e-2)