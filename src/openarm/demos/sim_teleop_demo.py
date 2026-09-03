"""Bimanual OpenArm sim teleop demo -- WebXR-driven left/right arms.

Requires the WebXR bridge server (webxr_pose_bridge) running and the
Quest 3S connected via its own browser (https://<jetson-ip>:8443/, "Enter
AR") before running this script.

The only thing sim-specific here is `robot.sim()` -- rig construction,
the viewer, and the run loop are shared with the real-hardware demo via
webxr_teleop, since both backends run MuJoCo (sim moves it directly,
real uses it for collision checking and viewing alongside real motor
commands).
"""

from __future__ import annotations

import logging

from mj_manipulator.event_loop import PhysicsEventLoop
from openarm.robot import Openarm

from openarm.webxr_teleop.teleop_config import TeleopConfig
from openarm.webxr_teleop.webxr_teleop import TeleopRig, run_teleop

logger = logging.getLogger(__name__)


def main(config: TeleopConfig | None = None) -> None:
    config = config or TeleopConfig.default()

    if config.debug:
        logging.basicConfig(level=logging.DEBUG)

    robot = Openarm()
    loop = PhysicsEventLoop()

    with robot.sim(physics=False, headless=True, event_loop=loop) as ctx:
        rig = TeleopRig(robot, ctx, loop, config)
        run_teleop(rig, config)


if __name__ == "__main__":
    main()