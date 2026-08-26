"""
Tests for Openarm's collision detection (`_ArmScope.check_collisions()` /
`Arm.check_collisions()`).

Contract under test (from openarm.py):

    def check_collisions(self) -> list[tuple[str, str, float]]:
        Returns a list of (arm_body, other_body, penetration_mm) tuples,
        empty if collision-free. Uses the same collision checker as the
        planner, with grasp-aware filtering (gripper-to-held-object
        contacts are OK). Also prints a human-readable summary.

One thing below is a placeholder you may need to adjust:

1. Forcing a real collision uses `robot.env.registry.activate("can", pos=...)`
   to place the object directly at the arm's current end-effector
   position -- guaranteed overlap by construction, independent of the
   URDF's self-collision geometry (joint limits are typically set
   specifically to *prevent* self-collision, so folding toward a limit
   isn't a reliable way to force one). If your scene uses a different
   object type than "can", swap the name in `_place_object_at_gripper`
   (and the "can_0" literal in `test_held_object_contact_is_filtered`).
"""

from __future__ import annotations

import numpy as np
import pytest

from openarm.robot_old import Openarm
from openarm.config import OpenarmConfig


def _place_object_at_gripper(robot: Openarm, side: str, object_type: str = "can") -> None:
    """Spawn/move a scene object to the given arm's current end-effector
    position, guaranteeing a real arm-vs-object collision. Independent of
    the URDF's self-collision geometry, which is generally already
    constrained by joint limits and may never actually self-intersect."""
    ee_pos = np.asarray(getattr(robot, side).get_ee_pose())[:3, 3]
    robot.env.registry.activate(object_type, pos=list(ee_pos))
    robot.forward()


@pytest.fixture(scope="module")
def robot():
    """A real Openarm instance with one graspable object in the scene.

    Skips the whole module if the default config's model can't be
    resolved in this environment, rather than failing -- consistent with
    how the ssik integration tests skip on missing config."""
    try:
        return Openarm(config=OpenarmConfig.default(), objects={"can": 1})
    except FileNotFoundError as e:
        pytest.skip(f"Openarm model not available in this environment: {e}", allow_module_level=True)


@pytest.fixture(autouse=True)
def _reset_robot(robot):
    """Reset to a known state before every test in this module, so
    joint-position manipulation in one test doesn't leak into the next."""
    robot.reset()
    yield


# ---------------------------------------------------------------------------
# Collision-free configurations
# ---------------------------------------------------------------------------


def test_ready_pose_is_collision_free(robot):
    for side in ("left", "right"):
        contacts = getattr(robot, side).check_collisions()
        assert contacts == [], f"{side} arm: expected collision-free at 'ready' pose, got {contacts}"


def test_returns_empty_list_not_none_when_collision_free(robot):
    """The documented contract is an empty list, not None/False -- callers
    doing `if contacts:` need this to actually be falsy-but-iterable."""
    contacts = robot.left.check_collisions()
    assert contacts == []
    assert isinstance(contacts, list)


# ---------------------------------------------------------------------------
# Detected collisions
# ---------------------------------------------------------------------------


def test_object_in_gripper_space_is_detected(robot):
    """Placing the scene object exactly at the left gripper's current
    position guarantees overlap -- check_collisions() should report it."""
    _place_object_at_gripper(robot, "left")
    contacts = robot.left.check_collisions()
    assert contacts, "expected at least one contact with the object placed at the gripper, got none"


def test_contact_tuple_shape_and_types(robot):
    """Every contact should match the documented
    (arm_body: str, other_body: str, penetration_mm: float) contract."""
    _place_object_at_gripper(robot, "left")
    contacts = robot.left.check_collisions()
    assert contacts, "test needs at least one contact to check tuple shape"

    for entry in contacts:
        assert len(entry) == 3, f"expected a 3-tuple, got {entry!r}"
        arm_body, other_body, depth_mm = entry
        assert isinstance(arm_body, str)
        assert isinstance(other_body, str)
        assert isinstance(depth_mm, (int, float))
        assert depth_mm >= 0, f"penetration depth should be non-negative, got {depth_mm}"


# ---------------------------------------------------------------------------
# Per-arm isolation
# ---------------------------------------------------------------------------


def test_collisions_are_reported_per_arm(robot):
    """Placing the object at only the left gripper shouldn't cause the
    right arm's check_collisions() to report anything."""
    _place_object_at_gripper(robot, "left")
    right_contacts = robot.right.check_collisions()
    assert right_contacts == [], f"right arm reported contacts after only placing an object at the left gripper: {right_contacts}"


def test_object_contacts_reference_that_arms_bodies(robot):
    """The (arm_body, ...) entries for the left arm's contacts should
    plausibly reference left-arm bodies, not right-arm ones -- a coarse
    sanity check that the two arms' collision bodies aren't cross-wired.

    Checks for the arm-side prefix ("openarm_right_") rather than a bare
    "right" substring, since gripper fingers are themselves named
    left_finger/right_finger regardless of which arm they belong to --
    "openarm_left_right_finger" is the LEFT arm's right finger."""
    _place_object_at_gripper(robot, "left")
    contacts = robot.left.check_collisions()
    assert contacts, "need at least one contact to check body naming"
    for arm_body, _, _ in contacts:
        assert "openarm_right_" not in arm_body.lower(), (
            f"left-arm contact unexpectedly references a right-arm body: {arm_body}"
        )


# ---------------------------------------------------------------------------
# Printed summary
# ---------------------------------------------------------------------------


def test_prints_collision_free_message(robot, capsys):
    robot.left.check_collisions()
    out = capsys.readouterr().out
    assert "collision-free" in out.lower()


def test_prints_contact_count_and_lines_when_colliding(robot, capsys):
    _place_object_at_gripper(robot, "left")
    contacts = robot.left.check_collisions()
    out = capsys.readouterr().out

    assert contacts, "need at least one contact for this test to be meaningful"
    assert str(len(contacts)) in out, "printed summary should mention the contact count"
    assert "<->" in out, "printed summary should list per-contact lines"


# ---------------------------------------------------------------------------
# Grasp-aware filtering
# ---------------------------------------------------------------------------


def test_held_object_contact_is_filtered(robot):
    """Once the object is both marked grasped AND kinematically attached,
    gripper-to-held-object contact should be filtered out of
    check_collisions() -- even though the gripper is still geometrically
    touching the object. Compared against the identical physical
    configuration used in test_object_in_gripper_space_is_detected, so
    this is a true before/after on the SAME overlap.

    Both calls are required: `mark_grasped` drives the is_grasped()
    classification used earlier in get_contacts(), but the actual
    gripper-vs-object filtering in _is_gripper_object_contact() reads
    from the attachments dict (via get_attachment_body()), not from
    is_grasped(). Without attach_object(), mark_grasped() alone has no
    effect on filtering -- confirmed by tracing collision.py directly.

    "openarm_left_hand" is passed as the attachment body rather than a
    specific finger: _get_gripper_base_name() expects a "parent/base"
    slash-delimited naming convention this URDF doesn't use (its body
    names are underscore-delimited), so it always returns None here and
    the code falls back to using the passed-in body directly as the
    "base" for a descendant check. Since both fingers are kinematic
    children of the hand body, passing the hand body lets the filter
    recognize contact with either finger.

    "can_0" matches the naming this scene's registry.activate("can", ...)
    produces for a single spawned instance -- confirmed from an actual
    check_collisions() run against this fixture."""
    _place_object_at_gripper(robot, "left")
    contacts_before = robot.left.check_collisions()
    assert contacts_before, "expected contact before grasping -- can't test filtering without one"

    robot.grasp_manager.mark_grasped("can_0", "left")
    robot.grasp_manager.attach_object("can_0", "openarm_left_hand")
    try:
        contacts_after = robot.left.check_collisions()
        assert not any("can_0" in (a, b) for a, b, _ in contacts_after), (
            f"expected can_0 contacts to be filtered once grasped and attached, got {contacts_after}"
        )
    finally:
        robot.grasp_manager.detach_object("can_0")
        robot.grasp_manager.mark_released("can_0")