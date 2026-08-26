"""Bimanual OpenArm real-hardware teleop demo -- WebXR-driven left/right arms.

Requires:
- Your ROS 2 arm/gripper controller nodes running and reachable (topics/
  actions per OpenarmConfig.to_hardware_config()).
- The WebXR bridge server + Quest 3S, same as sim_teleop_demo.py.

Identical to sim_teleop_demo.py except robot.real() instead of robot.sim();
rig construction, viewer, and run loop are shared via webxr_teleop.
"""

from __future__ import annotations

import logging

from mj_manipulator.event_loop import PhysicsEventLoop
from openarm.robot_old import Openarm

from openarm.webxr_teleop.teleop_config import TeleopConfig
from openarm.webxr_teleop.webxr_teleop import TeleopRig, run_teleop

logger = logging.getLogger(__name__)


def main(config: TeleopConfig | None = None) -> None:
    config = config or TeleopConfig.default()

    if config.debug:
        logging.basicConfig(level=logging.DEBUG)

    robot = Openarm()
    loop = PhysicsEventLoop()

    with robot.real(event_loop=loop,
                 physics_config=config.to_mj_physics_config()) as ctx:
        rig = TeleopRig(robot, ctx, loop, config)
        run_teleop(rig, config)


if __name__ == "__main__":
    main()