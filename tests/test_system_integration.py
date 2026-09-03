"""
System-level tests for the full bimanual plan -> execute pipeline.

Where test_kinematic_safety.py and test_collision_safety.py check
*planned* trajectories in isolation, these tests drive the same path a real
operation would: plan, then actually execute through `Openarm.sim()` /
`Openarm.execute()`, and check what the robot's live joint state does --
including the one thing planning-only tests can't cover, that an
emergency-stop request actually halts a trajectory already in flight.

These use the (fast, physics-free) kinematic sim context rather than real
ROS hardware, since that's what's available off-robot; the point is to
exercise `Openarm`'s own execution/abort bookkeeping end to end, not the
ROS/hardware transport underneath it.
"""

from __future__ import annotations

import numpy as np
import pytest

from openarm.config import OpenarmConfig
from openarm.robot import Openarm


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


def _small_bimanual_goal(robot, delta=0.4):
    goal_l = robot.left.arm.get_joint_positions().copy()
    goal_l[0] += delta
    goal_r = robot.right.arm.get_joint_positions().copy()
    goal_r[0] += delta
    return goal_l, goal_r


# ---------------------------------------------------------------------------
# Full plan -> execute round trip
# ---------------------------------------------------------------------------


class TestPlanExecuteRoundTrip:
    def test_bimanual_plan_executes_and_reaches_goal(self, robot):
        """The full pipeline a real operation actually uses: plan, then
        execute through a sim context, then check the robot's live joint
        state -- not just the trajectory object -- ends at the goal for
        both arms."""
        goal_l, goal_r = _small_bimanual_goal(robot)
        result = robot.plan_to_configuration({"left": goal_l, "right": goal_r}, seed=0)
        assert result is not None and result.success

        with robot.sim(physics=False, headless=True):
            ok = robot.execute(result)

        assert ok is True
        assert np.allclose(robot.left.arm.get_joint_positions(), goal_l, atol=1e-3)
        assert np.allclose(robot.right.arm.get_joint_positions(), goal_r, atol=1e-3)

    def test_repeated_plan_execute_reset_cycles_stay_consistent(self, robot):
        """Regression net for state leaking between operations -- runs
        several plan/execute/reset cycles back to back and checks each one
        independently reaches its goal from a clean ready pose, the way a
        real session would run many operations in sequence without
        restarting the process."""
        for i in range(5):
            robot.reset()
            delta = 0.1 * (i + 1)
            goal_l, goal_r = _small_bimanual_goal(robot, delta=delta)
            result = robot.plan_to_configuration({"left": goal_l, "right": goal_r}, seed=i)
            assert result is not None and result.success, f"cycle {i}: planning failed"

            with robot.sim(physics=False, headless=True):
                ok = robot.execute(result)
            assert ok is True, f"cycle {i}: execution failed"
            assert np.allclose(robot.left.arm.get_joint_positions(), goal_l, atol=1e-3), f"cycle {i}: left didn't reach goal"
            assert np.allclose(robot.right.arm.get_joint_positions(), goal_r, atol=1e-3), f"cycle {i}: right didn't reach goal"


# ---------------------------------------------------------------------------
# E-stop / abort during execution
# ---------------------------------------------------------------------------


class TestAbortDuringExecution:
    def test_prearmed_abort_prevents_any_motion(self, robot):
        """If the e-stop is already active when an execute call comes in,
        the robot must not move at all -- not partially, not by one
        waypoint."""
        goal_l, goal_r = _small_bimanual_goal(robot, delta=0.8)
        result = robot.plan_to_configuration({"left": goal_l, "right": goal_r}, seed=0)
        assert result is not None and result.success

        q_before = robot.left.arm.get_joint_positions().copy()
        robot.request_abort()
        with robot.sim(physics=False, headless=True):
            ok = robot.execute(result)

        assert ok is False
        assert np.array_equal(robot.left.arm.get_joint_positions(), q_before), (
            "arm moved despite the abort flag being set before execute() was called"
        )

    def test_mid_execution_abort_halts_before_reaching_goal(self, robot, monkeypatch):
        """A trajectory long enough to have several waypoints; abort fires
        partway through via a deterministic fake (returns False for the
        first few checks, then True), simulating an e-stop pressed while
        the robot is already moving. Execution must stop before the goal,
        not run to completion and merely report failure afterward."""
        goal_l, goal_r = _small_bimanual_goal(robot, delta=1.0)
        result = robot.plan_to_configuration({"left": goal_l, "right": goal_r}, seed=0)
        assert result is not None and result.success
        assert result.left.num_waypoints > 10, "trajectory too short to reliably observe a mid-flight abort"

        calls = {"n": 0}

        def fake_abort_after_a_few_ticks():
            calls["n"] += 1
            return calls["n"] > 5

        monkeypatch.setattr(robot, "is_abort_requested", fake_abort_after_a_few_ticks)

        with robot.sim(physics=False, headless=True):
            ok = robot.execute(result)

        assert ok is False, "execute() should report failure when aborted mid-flight"
        q_left_final = robot.left.arm.get_joint_positions()
        assert not np.allclose(q_left_final, goal_l, atol=1e-3), (
            "arm reached the goal despite a mid-execution abort -- abort is not actually "
            "interrupting a trajectory already in flight"
        )
        assert not np.allclose(q_left_final, np.zeros_like(q_left_final), atol=1e-6), (
            "arm never moved at all -- abort fired before the first waypoint, so this isn't "
            "actually testing a mid-flight interruption (tighten the fake or lengthen the trajectory)"
        )

    def test_abort_flag_is_independent_per_operation(self, robot):
        """clear_abort() must actually re-enable motion -- an operator
        clearing a fault should not be met with every subsequent command
        silently refusing to move."""
        goal_l, goal_r = _small_bimanual_goal(robot, delta=0.3)
        result = robot.plan_to_configuration({"left": goal_l, "right": goal_r}, seed=0)
        assert result is not None and result.success

        robot.request_abort()
        robot.clear_abort()
        with robot.sim(physics=False, headless=True):
            ok = robot.execute(result)

        assert ok is True, "execution did not resume after clear_abort()"
        assert np.allclose(robot.left.arm.get_joint_positions(), goal_l, atol=1e-3)