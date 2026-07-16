"""Robot configuration for Openarm."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from mj_manipulator.config import KinematicLimits

# Try to import openarm_assets for model paths, fall back to None if not installed
try:
    from openarm_assets import get_generated_model_path
except ImportError:
    get_generated_model_path = None

# Fudge factor to scale down the URDF's velocity limits while validating
SAFETY_SCALE = 1

# ---------------------------------------------------------------------------
# Arm specification
# ---------------------------------------------------------------------------


def _default_arm_kinematic_limits() -> KinematicLimits:
    """Default per-joint velocity/acceleration limits for an OpenArm arm.

    Returns a KinematicLimits directly -- no intermediate wrapper class.
    A dataclass field's default_factory must be a single zero-arg callable
    that produces the final value; the previous OpenarmArmKinematicLimits
    wrapper (construct-then-read-a-property) couldn't be expressed that
    way in one step, which is what caused the AttributeError. This
    function is the same information with no indirection.
    """
    return KinematicLimits(
        velocity=np.array(
            [16.754666, 16.754666, 5.445426, 5.445426, 20.943946, 20.943946, 20.943946]
        )
        * SAFETY_SCALE,
        acceleration=np.array(
            [167.54666, 167.54666, 54.45426, 54.45426, 209.43946, 209.43946, 209.43946]
        )
        * SAFETY_SCALE,
    )


@dataclass
class OpenarmArmSpec:
    """Specification for an Openarm arm."""

    prefix: str  # e.g., "openarm_left" or "openarm_right"
    ee_site: str = ""  # MuJoCo site name for the end-effector
    gripper_prefix: str = ""  # e.g., "left/gripper/"
    hand_type: str = "openarm"  # e.g., "openarm", "robotiq", etc.
    ik_solver: Literal["auto", "eaik", "ssik", "mink", "none"] = "auto"
    ssik_module: str = "openarm.IK.openarm_ssik"  # Python module for ssik
    kinematic_limits: KinematicLimits = field(default_factory=_default_arm_kinematic_limits)


# ---------------------------------------------------------------------------
# Gripper specification
# ---------------------------------------------------------------------------


@dataclass
class OpenarmGripperSpec:
    """Gripper travel limits, shared by both left and right arms.

    Values sourced from the vendor URDF's finger_joint1/2 <limit> tags
    (lower="0.0" upper="0.044") -- both fingers use the same range, and
    both arms use the same gripper hardware, so one config instance
    covers both sides.
    """

    finger_open: float = 0.044
    finger_closed: float = 0.0
    body_prefix: str = ""
    actuator_prefix: str = ""

    @classmethod
    def for_arm(cls, arm_name: str, **overrides) -> "OpenarmGripperSpec":
        """Convenience constructor matching today's vendor XML naming.

        Args:
            arm_name: "openarm_left" or "openarm_right".
            **overrides: any other OpenarmGripperSpec field (e.g.
                finger_open) to override from its default.
        """
        side = arm_name.replace("openarm_", "", 1)
        defaults = {
            "body_prefix": f"{arm_name}_",
            "actuator_prefix": f"{side}_",
        }
        defaults.update(overrides)
        return cls(**defaults)


# ---------------------------------------------------------------------------
# Mast Base
# ---------------------------------------------------------------------------


def _default_mast_kinematic_limits() -> KinematicLimits:
    """Default velocity/acceleration limits for the Mast linear base.

    Same fix as _default_arm_kinematic_limits above -- returns
    KinematicLimits directly, no wrapper class, no chained-property bug.
    """
    return KinematicLimits(
        velocity=np.array(0.1),  # m/s -- TODO: Calibrate
        acceleration=np.array(0.2),  # m/s^2 -- TODO: Calibrate
    )


@dataclass
class OpenarmMastConfig:
    """Configuration for a Mast base."""

    name: str
    joint_name: str  # MuJoCo joint name
    actuator_name: str = ""  # MuJoCo actuator name
    height_range: tuple[float, float] = (0.0, 0.5)  # meters (min, max) TODO: Calibrate
    collision_check_resolution: float = 0.01  # meters between collision checks
    kinematic_limits: KinematicLimits = field(default_factory=_default_mast_kinematic_limits)


# ---------------------------------------------------------------------------
# Debug logging
# ---------------------------------------------------------------------------


@dataclass
class DebugConfig:
    """Debug logging configuration.

    Controls which subsystems emit debug-level log messages.
    Use OPENARM_DEBUG=subsystem1,subsystem2 or OPENARM_DEBUG=all.

    ``verbose`` enables behavior tree visualization after each
    primitive (pickup, place, go_home), showing which nodes
    succeeded/failed. Can also be overridden per-call::

        robot.pickup("can_0", verbose=True)  # override for one call
        robot.config.debug.verbose = True    # enable globally
    """

    verbose: bool = False  # show BT tree status after each primitive
    planning: bool = False
    primitives: bool = False

    show_timestamps: bool = True
    show_module: bool = True

    def enable_all(self) -> None:
        self.verbose = True
        self.planning = True
        self.primitives = True

    def get_enabled_subsystems(self) -> list[str]:
        return [s for s in ("planning", "primitives") if getattr(self, s)]

    @classmethod
    def from_env(cls) -> "DebugConfig":
        """Create config from OPENARM_DEBUG environment variable."""
        config = cls()
        debug_env = os.environ.get("OPENARM_DEBUG", "")
        if debug_env:
            if debug_env.lower() == "all":
                config.enable_all()
            else:
                for s in debug_env.split(","):
                    s = s.strip()
                    if s and hasattr(config, s):
                        setattr(config, s, True)
        return config


_SUBSYSTEM_LOGGERS = {
    "planning": "openarm.robot",
    "primitives": "openarm.primitives",
    "affordances": "openarm.affordances",
}


def setup_logging(config: DebugConfig | None = None) -> None:
    """Configure openarm loggers based on debug config."""
    if config is None:
        config = DebugConfig.from_env()

    root_logger = logging.getLogger("openarm")
    root_logger.propagate = False

    if not root_logger.handlers:
        handler = logging.StreamHandler()
        fmt_parts = []
        if config.show_timestamps:
            fmt_parts.append("%(asctime)s")
        fmt_parts.append("%(levelname)s")
        if config.show_module:
            fmt_parts.append("[%(name)s]")
        fmt_parts.append("%(message)s")
        handler.setFormatter(logging.Formatter(" - ".join(fmt_parts)))
        root_logger.addHandler(handler)

    root_logger.setLevel(logging.WARNING)

    for subsystem in config.get_enabled_subsystems():
        logger_name = _SUBSYSTEM_LOGGERS.get(subsystem)
        if logger_name:
            logging.getLogger(logger_name).setLevel(logging.DEBUG)


# ---------------------------------------------------------------------------
# Top-level Openarm configuration
# ---------------------------------------------------------------------------


# Openarm joint name suffixes (combined with arm prefix). NOTE: prefix
# already ends without a trailing underscore ("openarm_left"), and these
# suffixes already start with one ("_joint1") -- f"{prefix}{suffix}"
# below (NOT f"{prefix}_{suffix}") to avoid a double underscore.
_OPENARM_JOINT_SUFFIXES = [f"_joint{i}" for i in range(1, 8)]


@dataclass
class PlanningConfig:
    """Planning parameters -- single source of truth for timeouts etc."""

    timeout: float = 30.0  # seconds per planning attempt
    base_heights: list[float] = field(default_factory=lambda: [0.2, 0.0, 0.4])
    lift_height: float = 0.05  # meters to lift after grasping


@dataclass
class OpenarmConfig:
    """Full robot configuration."""

    model_path: Path
    left_arm: OpenarmArmSpec
    right_arm: OpenarmArmSpec
    left_gripper: OpenarmGripperSpec | None = None
    right_gripper: OpenarmGripperSpec | None = None
    base: OpenarmMastConfig | None = None
    named_poses: dict[str, dict[str, list[float]]] = field(default_factory=dict)
    planning: PlanningConfig = field(default_factory=PlanningConfig)
    debug: DebugConfig = field(default_factory=DebugConfig.from_env)

    def joint_names(self, arm_spec: OpenarmArmSpec) -> list[str]:
        """Get prefixed OpenArm joint names for an arm spec.

        NOTE: fixed double-underscore bug -- arm_spec.prefix is
        "openarm_left" (no trailing underscore) and each suffix already
        starts with "_" (e.g. "_joint1"), so plain concatenation gives
        "openarm_left_joint1", not "openarm_left__joint1".
        """
        return [f"{arm_spec.prefix}{j}" for j in _OPENARM_JOINT_SUFFIXES]

    def gripper_spec_for_arm(self, arm_spec: OpenarmArmSpec) -> OpenarmGripperSpec | None:
        """Get gripper spec for an arm spec."""
        if arm_spec.prefix == "openarm_left":
            return self.left_gripper
        elif arm_spec.prefix == "openarm_right":
            return self.right_gripper
        else:
            return None

    @classmethod
    def default(cls) -> "OpenarmConfig":
        """Create default configuration for Openarm with its native gripper.

        UNVERIFIED: finger_open/finger_closed values below match
        OpenArmGripperConfig.for_arm()'s convention from earlier in this
        project (finger_open=0.044, finger_closed=0.0, per the vendor
        URDF's finger_joint1/2 <limit lower="0.0" upper="0.044"/>) --
        a previous version of this method had these two values swapped.
        Double-check against the real hardware/URDF before trusting this
        if gripper open/close ever look inverted in sim.
        """
        if get_generated_model_path is None:
            raise ImportError("openarm_assets package not found. Install it with:\n  uv add openarm_assets")
        return cls(
            model_path=get_generated_model_path(sides="bimanual"),
            left_arm=OpenarmArmSpec(
                prefix="openarm_left",
                ee_site="openarm_left_ee_site",
                gripper_prefix="openarm_left_gripper",
                ik_solver="auto",
                ssik_module="openarm.IK.openarm_left_ik",
            ),
            right_arm=OpenarmArmSpec(
                prefix="openarm_right",
                ee_site="openarm_right_ee_site",
                gripper_prefix="openarm_right_gripper",
                ik_solver="auto",
                ssik_module="openarm.IK.openarm_right_ik",
            ),
            left_gripper=OpenarmGripperSpec.for_arm("openarm_left"),
            right_gripper=OpenarmGripperSpec.for_arm("openarm_right"),
            # NOTE: no base= here -- unlike Geodude's Vention linear-actuator
            # base, OpenArm's current MJCF mounts the torso rigidly to world
            # via a fixed joint (openarm_body_world_joint); there's no linear
            # mast/base hardware to reference. base defaults to None. If a
            # real height-adjustable base is added to the model later,
            # construct an OpenarmMastConfig here with the actual joint/
            # actuator names from that MJCF -- not copied from Geodude's.
        )

    @classmethod
    def from_yaml(cls, path: Path) -> "OpenarmConfig":
        """Load configuration from YAML file."""
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)

        left_gripper = None
        if "left_gripper" in data:
            left_gripper = OpenarmGripperSpec(**data["left_gripper"])
        right_gripper = None
        if "right_gripper" in data:
            right_gripper = OpenarmGripperSpec(**data["right_gripper"])

        base = None
        if "base" in data:
            base = OpenarmMastConfig(**data["base"])

        return cls(
            model_path=Path(data["model_path"]),
            left_arm=OpenarmArmSpec(**data["left_arm"]),
            right_arm=OpenarmArmSpec(**data["right_arm"]),
            left_gripper=left_gripper,
            right_gripper=right_gripper,
            base=base,
            named_poses=data.get("named_poses", {}),
        )