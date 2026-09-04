"""
Regression tests for Openarm._context handling across execution-context
types.

request_abort(), clear_abort(), and reset() all reach into self._context
directly (.ownership, ._event_loop, .reset_to_keyframe(...)) -- outside the
narrow step/execute/sync/is_running/arm/control_dt ExecutionContext protocol
that OpenarmRealContext was built to satisfy. Every existing abort test
(test_system_integration.py) only exercises this via robot.sim(...), where
_context is the raw SimContext and all three attributes exist. In
robot.real(...), _context is OpenarmRealContext itself, which exposes none
of them -- so on real hardware these currently raise AttributeError instead
of doing their job.

These tests drive OpenarmRealContext directly with fakes standing in for
the ROS HardwareContext and the shadow SimContext, so they run without ROS
or hardware but fail for the same reason the real thing would.
"""

from __future__ import annotations

import pytest

from openarm.config import OpenarmConfig
from openarm.robot import Openarm, OpenarmRealContext


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
    robot._active_context = None
    robot.clear_abort()
    yield
    robot._active_context = None
    robot.clear_abort()


class _FakeOwnership:
    """Stands in for mj_manipulator.ownership.OwnershipRegistry."""

    def __init__(self):
        self.abort_all_called = False
        self.clear_all_called = False

    def abort_all(self):
        self.abort_all_called = True

    def clear_all(self):
        self.clear_all_called = True


class _FakeEventLoop:
    def __init__(self):
        self.deactivate_all_teleop_called = False

    def _deactivate_all_teleop(self):
        self.deactivate_all_teleop_called = True


class _FakeShadowContext:
    """Minimal stand-in for the shadow SimContext inside OpenarmRealContext.
    Implements exactly the surface Openarm's abort/reset bookkeeping needs,
    so failures here mean the same thing they'd mean on real hardware."""

    def __init__(self):
        self.ownership = _FakeOwnership()
        self._event_loop = _FakeEventLoop()
        self.reset_to_keyframe_called_with = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def reset_to_keyframe(self, name):
        self.reset_to_keyframe_called_with = name

    def sync(self):
        pass

    def step(self, targets=None):
        pass

    def step_cartesian(self, arm_name, position, velocity=None):
        pass

    def execute(self, item):
        return True

    def arm(self, name):
        return None


class _FakeHardwareContext:
    """Minimal stand-in for mj_manipulator_ros.HardwareContext. Also
    carries a reset_to_keyframe spy -- not because real HardwareContext
    has one, but so a test can prove reset() never reaches for it."""

    def __init__(self):
        self._running_flag = True
        self.reset_to_keyframe_called_with = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def reset_to_keyframe(self, name):
        self.reset_to_keyframe_called_with = name

    def is_running(self):
        return self._running_flag

    def sync(self):
        pass

    def step(self, targets=None):
        pass

    def step_cartesian(self, arm_name, position, velocity=None):
        pass

    def execute(self, item):
        return True

    def arm(self, name):
        return None

    @property
    def control_dt(self):
        return 0.002


def _install_fake_real_context(robot):
    """Bypasses .real() (which needs live ROS action servers) and drops a
    real OpenarmRealContext, wired to fakes, directly into
    robot._active_context -- exactly the object shape request_abort(),
    clear_abort(), and reset() will see on actual hardware."""
    shadow = _FakeShadowContext()
    hw = _FakeHardwareContext()
    ctx = OpenarmRealContext(hw, shadow, robot)
    robot._active_context = ctx
    return ctx, shadow, hw


# ---------------------------------------------------------------------------
# The core regression: these must not raise, in either context type
# ---------------------------------------------------------------------------


class TestAbortAndResetBookkeepingWithRealContext:
    def test_request_abort_does_not_raise(self, robot):
        _install_fake_real_context(robot)
        robot.request_abort()  # currently raises AttributeError

    def test_request_abort_actually_sets_the_flag(self, robot):
        """The AttributeError happens *before* self._abort_event.set(), so
        even if a caller swallowed the exception, is_abort_requested()
        would still (silently) report no abort happened."""
        _install_fake_real_context(robot)
        try:
            robot.request_abort()
        except AttributeError:
            pass
        assert robot.is_abort_requested() is True

    def test_request_abort_calls_ownership_abort_all(self, robot):
        _, shadow, _ = _install_fake_real_context(robot)
        try:
            robot.request_abort()
        except AttributeError:
            pass
        assert shadow.ownership.abort_all_called, (
            "ownership.abort_all() never reached -- arms are not released/"
            "aborted via the ownership registry on real hardware"
        )

    def test_request_abort_deactivates_teleop(self, robot):
        _, shadow, _ = _install_fake_real_context(robot)
        try:
            robot.request_abort()
        except AttributeError:
            pass
        assert shadow._event_loop.deactivate_all_teleop_called, (
            "teleop was never deactivated -- an operator hitting e-stop "
            "mid-teleop would keep sending commands"
        )

    def test_clear_abort_does_not_raise(self, robot):
        _install_fake_real_context(robot)
        robot.request_abort()
        robot.clear_abort()

    def test_clear_abort_calls_ownership_clear_all(self, robot):
        _, shadow, _ = _install_fake_real_context(robot)
        robot.request_abort()
        try:
            robot.clear_abort()
        except AttributeError:
            pass
        assert shadow.ownership.clear_all_called

    def test_reset_does_not_raise(self, robot):
        """reset() calls self._context.reset_to_keyframe('ready'), which is
        also absent from OpenarmRealContext -- a second, independent break."""
        _install_fake_real_context(robot)
        robot.reset()

    def test_reset_only_touches_the_shadow_never_the_hardware(self, robot):
        """There is no such thing as teleporting a real robot. If reset()
        ever reaches HardwareContext instead of the shadow, that's a much
        worse bug than a missing attribute."""
        _, shadow, hw = _install_fake_real_context(robot)
        try:
            robot.reset()
        except AttributeError:
            pass
        assert shadow.reset_to_keyframe_called_with == "ready"
        assert hw.reset_to_keyframe_called_with is None


# ---------------------------------------------------------------------------
# Contract test: catches the *next* one of these before it ships, not just
# the ones we happened to find today
# ---------------------------------------------------------------------------


class TestContextInterfaceContract:
    """Every attribute Openarm accesses on self._context outside the
    standard ExecutionContext protocol, in one place. Add to this tuple the
    moment robot.py grows a new self._context.<something> access -- that's
    the only way this test stays a contract instead of a snapshot of
    today's bugs."""

    REQUIRED_CONTEXT_ATTRS = ("ownership", "_event_loop", "reset_to_keyframe", "sync")

    def test_openarm_real_context_implements_the_full_contract(self, robot):
        ctx = OpenarmRealContext(_FakeHardwareContext(), _FakeShadowContext(), robot)
        missing = [a for a in self.REQUIRED_CONTEXT_ATTRS if not hasattr(ctx, a)]
        assert not missing, (
            f"OpenarmRealContext is missing {missing} -- request_abort()/"
            f"clear_abort()/reset() will raise AttributeError the first "
            f"time this is exercised on real hardware"
        )