"""
GATE FILE -- naming/identity consistency across the planning/execution
boundary, without needing ROS or real hardware.

`OpenarmConfig.to_hardware_config()` is the one place this codebase
translates the planning layer's arm/joint identity (`ArmGroup` keys,
`Trajectory.entity` tags) into whatever `mj_manipulator_ros.HardwareContext`
will use to address real controllers. Nothing else in this test suite
crosses that boundary -- `mj_manipulator_ros`'s own tests build their configs
by hand, self-consistently, and never touch this method's actual output.

If these fail, the arm names / joint names / controller identifiers produced
by `to_hardware_config()` do not match what the rest of this codebase (and
therefore the real robot) will actually be addressed by: a planned
trajectory's entity tag will not resolve to any hardware client, and
`execute()` on real hardware will fail -- or, worse, silently address the
wrong arm -- before a single joint moves.
"""

from __future__ import annotations

import numpy as np
import pytest

from mj_manipulator_ros.config import ArmHardwareConfig, HardwareConfig
from openarm.config import OpenarmConfig
from openarm.robot import Openarm

# The real ROS controller-naming convention this robot is actually deployed
# with (see openarm_impedance_control/config/openarm_impedance_controllers.yaml).
# This is site-specific deployment data, not a code invariant -- update it
# here the same way you'd update a config file if the real controller names
# ever change.
EXPECTED_CONTROLLER_NAME = {"left": "left_controller", "right": "right_controller"}


@pytest.fixture(scope="module")
def robot():
    try:
        return Openarm(config=OpenarmConfig.default())
    except FileNotFoundError as e:
        pytest.skip(f"Openarm model not available in this environment: {e}", allow_module_level=True)


@pytest.fixture(autouse=True)
def _reset_robot(robot):
    robot.reset()
    yield


# ---------------------------------------------------------------------------
# Arm and joint identity
# ---------------------------------------------------------------------------


class TestHardwareConfigArmIdentity:
    def test_hardware_config_has_one_entry_per_planning_arm(self, robot):
        hw = robot.config.to_hardware_config()
        assert isinstance(hw, HardwareConfig)
        assert len(hw.arms) == len(robot.arms), (
            f"to_hardware_config() produced {len(hw.arms)} arm entries, but the planning "
            f"layer has {len(robot.arms)} arms ({set(robot.arms)})"
        )

    def test_every_hardware_arm_name_matches_a_real_planning_entity(self, robot):
        """The core check: every name to_hardware_config() hands to the
        execution layer must be one of the identifiers the PLANNING layer
        will actually tag trajectories with (ArmGroup's keys / Trajectory
        .entity values) -- not some other convention (a URDF prefix like
        'openarm_left', or a decorated name like 'left_arm') that merely
        looks similar.
        """
        hw = robot.config.to_hardware_config()
        hw_names = {arm_cfg.name for arm_cfg in hw.arms}
        planning_entity_names = set(robot.arms)  # ArmGroup keys == Trajectory.entity values

        assert hw_names == planning_entity_names, (
            f"hardware config arm names {hw_names} do not match the planning layer's "
            f"actual entity names {planning_entity_names} -- a trajectory's .entity tag "
            f"will not resolve to a hardware client for the mismatched name(s): "
            f"{hw_names.symmetric_difference(planning_entity_names)}"
        )

    def test_a_real_planned_trajectorys_entity_resolves_to_a_hardware_arm(self, robot):
        """End-to-end version of the check above, using an actual planned
        trajectory's real `.entity` value rather than just comparing key
        sets -- catches the case where the two happen to share the same SET
        of strings but a bug elsewhere attaches the wrong one to a given
        side.
        """
        goal_l = robot.left.arm.get_joint_positions().copy()
        goal_l[0] += 0.2
        result = robot.plan_to_configuration({"left": goal_l}, seed=0)
        assert result is not None and result.success

        hw = robot.config.to_hardware_config()
        hw_names = {arm_cfg.name for arm_cfg in hw.arms}

        assert result.left.entity is not None
        assert result.left.entity in hw_names, (
            f"planned trajectory entity '{result.left.entity}' has no matching entry in "
            f"to_hardware_config()'s arm names {hw_names} -- execute() would raise looking "
            f"for a hardware client for this trajectory"
        )

    @pytest.mark.parametrize("side", ["left", "right"])
    def test_joint_names_match_between_planning_and_hardware_config(self, robot, side):
        """Same identity check, one level down: joint names AND order must
        also agree, or commands will be sent to the right arm but the wrong
        joints (or the right joints in the wrong order -- exactly the class
        of bug openarm_impedance_control's own check_ordering.py exists to
        catch on the ROS side; this is the equivalent check on the planning/
        config side, runnable without ROS).
        """
        hw = robot.config.to_hardware_config()
        planning_joint_names = list(robot.arms[side].config.joint_names)

        matching = [a for a in hw.arms if a.name == side]
        assert matching, f"no hardware config entry named '{side}' (see previous test for why)"
        hw_joint_names = list(matching[0].joint_names)

        assert hw_joint_names == planning_joint_names, (
            f"'{side}': hardware config joint order {hw_joint_names} != planning layer "
            f"joint order {planning_joint_names}"
        )


# ---------------------------------------------------------------------------
# Controller/action naming (deployment-specific, but worth pinning down)
# ---------------------------------------------------------------------------


class TestHardwareConfigControllerNaming:
    """Softer than the identity checks above -- controller names are
    deployment configuration, not a pure code invariant -- but still worth
    asserting explicitly rather than discovering a mismatch live against real
    hardware. Update EXPECTED_CONTROLLER_NAME at the top of this file if the
    real ROS-side controller names ever change.
    """

    @pytest.mark.parametrize("side", ["left", "right"])
    def test_trajectory_action_name_matches_the_real_controller(self, side):
        cfg = OpenarmConfig.default()
        hw = cfg.to_hardware_config()
        matching = [a for a in hw.arms if a.name == side]
        assert matching, f"no hardware config entry named '{side}'"
        arm_cfg: ArmHardwareConfig = matching[0]

        expected_controller = EXPECTED_CONTROLLER_NAME[side]
        expected_action = f"/{expected_controller}/follow_joint_trajectory"

        # Either the trajectory-controller name matches the real controller
        # directly, or an explicit action override points at the real path.
        # Either is fine; silence on both is not.
        controller_ok = arm_cfg.joint_trajectory_controller == expected_controller
        override_ok = arm_cfg.follow_joint_trajectory_action == expected_action
        assert controller_ok or override_ok, (
            f"'{side}': neither joint_trajectory_controller "
            f"('{arm_cfg.joint_trajectory_controller}') nor an explicit "
            f"follow_joint_trajectory_action override points at the real controller "
            f"'{expected_controller}' (expected action path '{expected_action}')"
        )