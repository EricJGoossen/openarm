# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Eric Goossen

"""Visual smoke test: plan and execute motions in the MuJoCo viewer.


Opens the sim viewer, plans to the joint-range midpoint, executes it, then
plans back to the HOME configuration and executes that. Useful for eyeballing
that planning + retiming + execution all work together on the real scene.


This is an interactive/visual check, not an automated assertion — it leaves
the viewer open at the end so you can inspect the final state.


Run directly: `uv run mjpython -m openarm.tests.test_visual_plan_execute`
(the native viewer needs mjpython on macOS; plain python is fine on Linux).
"""


from __future__ import annotations

import time

from mj_environment import Environment
from mj_manipulator.sim_context import SimContext

from openarm_assets import get_model_path
from openarm.openarm_left import OPENARM_LEFT_HOME, create_openarm_left_arm

DECORATED_XML = (
    "openarm_assets/src/openarm_assets/models/openarm/"
    "generated/openarm_bimanual_decorated.xml"
)


def _plan_execute(arm, ctx, target_q, label: str) -> None:
    """Plan to target_q, retime, and execute; print status."""
    print(f"\nPlanning to {label}: {target_q}")
    path = arm.plan_to_configuration(target_q)

    if path is None:
        print(f"Plan to {label} returned None.")
        return
    
    traj = arm.retime(path)
    print(f"Retimed: duration={traj.duration:.2f}s, {len(traj.positions)} samples")
    
    ok = ctx.execute(traj)
    print(f"Execute to {label}: {'OK' if ok else 'FAILED'}")


def main() -> None:
    env = Environment(DECORATED_XML)
    arm = create_openarm_left_arm(env)

    with SimContext(env.model, env.data, {"openarm_left": arm},
                    physics=False, headless=False) as ctx:
        lower, upper = arm.get_joint_limits()
        print(f"Start joints: {arm.get_joint_positions()}")
        print(f"Collisions at start: {len(arm.check_collisions())}")

        safe_q = (lower + upper) / 2
        _plan_execute(arm, ctx, safe_q, "safe midpoint")
        time.sleep(1.0)  # settle, so the motion is visible

        _plan_execute(arm, ctx, OPENARM_LEFT_HOME, "HOME")
        print(f"Final joints: {arm.get_joint_positions()}")

        print("\nDone. Viewer stays open — close the window or Ctrl-C to exit.")
        while ctx.is_running():
            ctx.sync()


if __name__ == "__main__":
    main()
