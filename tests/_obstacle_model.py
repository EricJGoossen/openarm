"""
Not a test file (no test_ prefix, not collected by pytest).

Builds a copy of the generated bimanual scene XML with one extra static
sphere obstacle welded into the world, for tests that need a real
arm-vs-environment collision case without depending on mj_environment's
object-loading path (which composes the scene via
`mujoco.MjModel.from_xml_string`, and separately loses the robot's own
relative `meshdir` in that mode -- unrelated to bimanual planning, and not
what these tests are checking).

The obstacle is loaded the same way a robot-only scene normally is (a
plain `from_xml_path` load of a single self-contained MJCF file), which is
also a closer match to how a real cell's fixed obstacles (a mounting table,
a cage bar, a wall) would actually be represented -- baked into the scene
description, not injected as a runtime "object".
"""

from __future__ import annotations

import os

from openarm_assets import get_generated_model_path

# Sphere placed to sit on the straight-line path between the right arm's
# ready pose and a specific large-swing goal, while leaving both endpoints
# clear. Derived empirically (see build_obstacle_model docstring below) --
# regenerate by sampling end-effector positions along
# `_straight_line(q_ready, q_goal)` for a candidate goal and placing the
# sphere near the midpoint, then confirming zero contacts at both endpoints
# and at least one contact along the naive straight-line path.
OBSTACLE_POS = (-0.285, -0.218, 0.181)
OBSTACLE_RADIUS = 0.06

# The specific goal this obstacle was tuned against (right arm, from the
# all-zeros ready pose). See test_collision_safety.py for how it's used.
RIGHT_ARM_OBSTACLE_GOAL_SCALE = 0.7  # fraction of the way to the lower joint limits


def build_obstacle_model_xml(dst_path: str) -> str:
    """Write a copy of the default bimanual model with the test obstacle
    added, and return the path. `dst_path` should be outside the source
    tree (e.g. a pytest tmp_path) since this rewrites `meshdir` to an
    absolute path rather than leaving it relative to the copy's location.
    """
    src_path = str(get_generated_model_path(sides="bimanual"))
    src_dir = os.path.dirname(os.path.abspath(src_path))
    abs_meshdir = os.path.normpath(os.path.join(src_dir, "../meshes")) + "/"

    with open(src_path) as f:
        xml = f.read()

    if 'meshdir="../meshes/"' not in xml:
        raise RuntimeError(
            "Expected meshdir=\"../meshes/\" in the generated model XML; "
            "the model layout may have changed and OBSTACLE_POS/RADIUS "
            "above may need re-deriving."
        )
    xml = xml.replace('meshdir="../meshes/"', f'meshdir="{abs_meshdir}"')

    obstacle = (
        f'\n    <body name="test_obstacle" pos="{OBSTACLE_POS[0]} {OBSTACLE_POS[1]} {OBSTACLE_POS[2]}">\n'
        f'      <geom name="test_obstacle_geom" type="sphere" size="{OBSTACLE_RADIUS}" rgba="1 0 0 0.5"/>\n'
        f"    </body>\n"
    )
    if "</worldbody>" not in xml:
        raise RuntimeError("Expected a </worldbody> closing tag in the generated model XML.")
    xml = xml.replace("</worldbody>", obstacle + "</worldbody>")

    with open(dst_path, "w") as f:
        f.write(xml)
    return dst_path