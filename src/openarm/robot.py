from __future__ import annotations

import logging
import random
from typing import Any

import mujoco
import numpy as np
from mj_environment import Environment
from mj_manipulator import (
    Arm,
    GraspManager,
    PlanGroupResult,
    SimContext,
    ArmGroup,
)

from mj_manipulator.config import ArmConfig, ArmGroupConfig, KinematicLimits, PhysicsConfig
from mj_manipulator.grasp_verifier import GraspVerifier
from openarm.config import OpenarmConfig, OpenarmArmSpec, OpenarmGripperSpec, setup_logging
from openarm.openarm_mast import OpenarmMast
from openarm.openarm_gripper import OpenarmGripper

logger = logging.getLogger(__name__)

class OpenarmSimContext:
    """Wrapper that sets robot._active_context on enter/exit."""

    def __init__(self, inner: SimContext, robot: "Openarm"):
        self._inner = inner
        self._robot = robot

    def __enter__(self):
        ctx = self._inner.__enter__()
        self._robot._active_context = ctx
        return ctx

    def __exit__(self, *args):
        self._robot._active_context = None
        return self._inner.__exit__(*args)

class OpenarmRealContext:
    """Composite context: real ROS 2 hardware for actual motor commands,
    plus a headless kinematic SimContext (same event_loop) purely as a
    collision-checked shadow for the viewer/TeleopPanel.
    """

    def __init__(self, hw, shadow, robot: "Openarm"):
        self._hw = hw
        self._shadow = shadow
        self._robot = robot

    def __enter__(self):
        self._shadow.__enter__()   # registers controller on the event loop
        self._hw.__enter__()       # connects ROS 2, waits for action servers
        self._robot._active_context = self
        return self

    def __exit__(self, *args):
        self._robot._active_context = None
        hw_ok = self._hw.__exit__(*args)
        shadow_ok = self._shadow.__exit__(*args)
        return hw_ok and shadow_ok

    def step_cartesian(self, arm_name, position, velocity=None):
        self._shadow.step_cartesian(arm_name, position, velocity)  # local viewer/collision
        self._hw.step_cartesian(arm_name, position, velocity)      # real motors

    def step(self, targets=None):
        self._shadow.step(targets)
        self._hw.step(targets)

    def execute(self, item):
        self._shadow.execute(item)
        return self._hw.execute(item)

    def sync(self):
        self._shadow.sync()
        self._hw.sync()

    def is_running(self):
        return self._hw.is_running()

    def arm(self, name):
        return _ShadowArmController(self._shadow.arm(name), self._hw.arm(name))

    @property
    def control_dt(self):
        return self._hw.control_dt  # real cadence (500 Hz), not sim's 250 Hz default

class _ShadowArmController:
    """Forwards grasp/release to both the shadow (bookkeeping/visual) and
    the real gripper. Real result wins."""

    def __init__(self, shadow_arm, hw_arm):
        self._shadow = shadow_arm
        self._hw = hw_arm

    def grasp(self, object_name=None, synchronous=True):
        self._shadow.grasp(object_name)
        return self._hw.grasp(object_name, synchronous=synchronous)

    def release(self, object_name=None, synchronous=True):
        self._shadow.release(object_name)
        self._hw.release(object_name, synchronous=synchronous)

    def set_width(self, width, synchronous=True):
        self._shadow.set_width(width)
        return self._hw.set_width(width, synchronous=synchronous)

    def get_width(self):
        return self._shadow.get_width()

class _ArmScope:
    """Provides access to an arm by name, e.g., robot.left or robot.right."""

    def __init__(self, robot: "Openarm", name: str):
        self._robot = robot
        self._name = name
        self._arm = robot.arms[name]

    @property
    def arm(self) -> Arm:
        return self._robot.arms[self._name]

    @property
    def gripper(self) -> OpenarmGripper:
        return self._robot.grasp_manager.grippers[self._name]

    @property
    def named_poses(self) -> dict[str, list[float]]:
        """Named poses from config and keyframes."""
        return self._robot.named_poses[self._name]

    def plan_ee_to_pose(
        self, target: list[np.ndarray],
    ) -> PlanGroupResult | None:
        """Plan a trajectory for the end effectors to reach the target poses."""
        return self._robot.plan_ee_to_pose({self._name: target})

    def plan_to_configuration(
        self, target: list[np.ndarray],
    ) -> PlanGroupResult | None:
        """Plan a trajectory for the arms to reach the target joint configurations."""
        return self._robot.plan_to_configuration({self._name: target})

    def plan_to_tsrs(
        self, target: list[np.ndarray],
    ) -> PlanGroupResult | None:
        """Plan a trajectory for the arms to reach the target TSRs."""
        return self._robot.plan_to_tsrs({self._name: target})

    def set_gripper_width(self, target: float, synchronous: bool = True) -> bool:
        """Set the width of the gripper.

        Args:
            target: Desired gripper width, 0.0 (closed) to 1.0 (open).
            synchronous: On real hardware (inside `with robot.real()`),
                blocks until the physical gripper finishes moving. In
                sim, or with no active context, application is instant
                and this has no effect.
        """
        ctx = self._robot._active_context
        if isinstance(ctx, OpenarmRealContext):
            return ctx.arm(self._name).set_width(target, synchronous=synchronous)
        return self._arm.gripper.set_width(target)

    def get_gripper_width(self) -> float:
        """Get the current width of the grippers."""
        return self._arm.gripper.get_width()

    def check_collisions(self) -> bool:
        """Check for collisions in the current state."""
        return self._robot.arms.check_collisions(self._name)

    def get_ee_pose(self) -> np.ndarray:
        """Current end-effector pose as 4x4 homogeneous transform."""
        return self._arm.get_ee_pose()

    def get_joint_positions(self) -> np.ndarray:
        """Current joint positions (rad)."""
        return self._arm.get_joint_positions()

    def get_ee_velocities(self) -> np.ndarray:
        """Current end-effector velocities (linear and angular, in world frame)."""
        return self._arm.get_ee_velocities()

    def get_joint_velocities(self) -> np.ndarray:
        """Current joint velocities (rad/s)."""
        return self._arm.get_joint_velocities()


class Openarm:
    config: OpenarmConfig
    model: mujoco.MjModel
    data: mujoco.MjData
    grasp_manager: GraspManager

    # -----------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------

    def __init__(
        self,
        config: OpenarmConfig | None = None,
        objects: dict[str, int] | None = None,
    ):
        self.config: OpenarmConfig = config or OpenarmConfig.default()

        # Load MuJoCo model via mj_environment
        if not self.config.model_path.exists():
            raise FileNotFoundError(
                f"MuJoCo model not found: {self.config.model_path}\n make sure openarm_assets is available."
            )
        
        if objects:
            from prl_assets import OBJECTS_DIR

            scene_config = self._create_temp_scene_config(objects)
            self._env = Environment(
                base_scene_xml=str(self.config.model_path),
                objects_dir=str(OBJECTS_DIR),
                scene_config_yaml=scene_config,
            )
        else:
            self._env = Environment(
                base_scene_xml=str(self.config.model_path),
                objects_dir=None,
                scene_config_yaml=None,
            )

        self.model: mujoco.MjModel = self._env.model
        self.data: mujoco.MjData = self._env.data

        # Shared grasp manager
        self.grasp_manager: GraspManager = GraspManager(self.model, self.data)

        # Create arms from mj_manipulator
        self._arm_group = self._create_arm_group(
            {"left": self.config.left_arm, "right": self.config.right_arm}
        )

        # Cache freejoint qpos addresses (for hide-all in reset)
        self._freejoint_qpos_addrs = [
            self.model.jnt_qposadr[i]
            for i in range(self.model.njnt)
            if self.model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE
        ]

        # Create base (if configured).
        self._base: OpenarmMast | None = None
        if self.config.base is not None:
            self._base = OpenarmMast(
                self.model,
                self.data,
                self.config.base,
                self._arm_group
            )

        # Named poses from keyframes
        keyframe_poses = self._load_keyframe_poses()
        self._named_poses = {**self.config.named_poses, **keyframe_poses}

        # Initialize state
        mujoco.mj_forward(self.model, self.data)
        setup_logging(self.config.debug)

        # Active execution context (set by sim() context manager)
        self._context: SimContext | None = None

        # Abort flag (thread-safe, shared between terminal and Viser)
        import threading

        self._abort_event = threading.Event()

    def _create_arm(self, spec: OpenarmArmSpec, name: str) -> Arm:
        """Create an mj_manipulator Arm from a OpenarmArmSpec."""
        joint_names = self.config.joint_names(spec)
        arm_config = ArmConfig(
            name=name,
            entity_type="arm",
            joint_names=joint_names,
            kinematic_limits=spec.kinematic_limits,
            ee_site=spec.ee_site,
        )

        # Create arm first to get joint indices for IK solver
        arm = Arm(self._env, arm_config) 

        from mj_manipulator.arms._ik_factory import resolve_ik_solver
        ik_solver = resolve_ik_solver(
            arm, with_ik=spec.ik_solver, ssik_module=spec.ssik_module
        )

        gripper = OpenarmGripper(
            self.model,
            self.data,
            name,
            grasp_manager=self.grasp_manager,
            config=self.config.gripper_spec_for_arm(spec),
        )

        gripper.grasp_verifier = GraspVerifier(
            gripper=gripper,
            signals=[],
        )

        arm = Arm(
            self._env,
            arm_config,
            ik_solver=ik_solver,
            gripper=gripper,
            grasp_manager=self.grasp_manager,
        )

        return arm

    def _create_arm_group(self, specs: dict[str, OpenarmArmSpec]) -> ArmGroup:
        """Create an ArmGroup from a mapping of side name ("left"/"right") to OpenarmArmSpec."""
        arms = {}
        for name, spec in specs.items():
            arm = self._create_arm(spec, spec.prefix)
            arms[name] = arm
        group_config = ArmGroupConfig(
            name="bimanual",
            entity_type="arm_group",
            joint_names=sum([self.config.joint_names(spec) for spec in specs.values()], []),
        )
        return ArmGroup(arms, group_config)

    def sim(
        self,
        physics: bool = True,
        viewer=None,
        viewer_fps: float = 30.0,
        headless: bool = False,
        event_loop=None,
    ) -> OpenarmSimContext:
        """Create simulation execution context.

        Returns a context manager for executing trajectories in MuJoCo.
        Sets robot._active_context on enter, clears on exit.

        Example::

            with robot.sim(physics=True) as ctx:
                path = robot.left_arm.plan_to_pose(target)
                traj = robot.left_arm.retime(path)
                ctx.execute(traj)
                ctx.arm("left").grasp("can_0")
        """
        entities = {}
        if self._base is not None:
            entities[self._base.config.name] = self._base
        inner = SimContext(
            self.model,
            self.data,
            self._arm_group,
            physics=physics,
            headless=headless,
            viewer=viewer,
            viewer_fps=viewer_fps,
            entities=entities,
            abort_fn=self.is_abort_requested,
            event_loop=event_loop,
            physics_config=self.config.physics_config,
        )
        return OpenarmSimContext(inner, self)

    def real(
        self,
        event_loop=None,
    ) -> OpenarmRealContext:
        """Create real-hardware execution context via ROS 2.
    
        Requires your ROS 2 arm/gripper controller nodes already running and
        advertising the topics/actions in mj_manipulator_ros.interfaces for
        arm names "left"/"right" (see OpenarmConfig.to_hardware_config).

        Example::

            loop = PhysicsEventLoop()
            with robot.real(event_loop=loop) as ctx:
                rig = TeleopRig(robot, ctx, loop, config)
                run_teleop(rig, config)
        """
        from mj_manipulator_ros.hardware_context import HardwareContext
        
        shadow = SimContext(
            self.model,
            self.data,
            self._arm_group,
            physics=False,
            headless=True,
            abort_fn=self.is_abort_requested,
            event_loop=event_loop,
            physics_config=self.config.physics_config,
        )
        hw = HardwareContext(self.config.to_hardware_config(), node_name=self.config.physics_config.node_name)
        return OpenarmRealContext(hw, shadow, self)

    def __getattr__(self, name: str) -> Any:
        """Allow access to arms via robot.left or robot.right."""
        return getattr(self._arm_group, name)

    def __dir__(self) -> list[str]:
        """Include arm names in dir(robot)."""
        return sorted(set(super().__dir__()) | set(dir(self._arm)))

    @property
    def left(self) -> _ArmScope:
        """Left arm scope."""
        return _ArmScope(self, "left")

    @property
    def right(self) -> _ArmScope:
        """Right arm scope."""
        return _ArmScope(self, "right")

    @property
    def arms(self) -> ArmGroup:
        """All arms, keyed by side name."""
        return self._arm_group

    @property
    def env(self) -> Environment:
        """Underlying MuJoCo environment."""
        return self._env

    @property
    def _active_context(self):
        return self._context

    @_active_context.setter
    def _active_context(self, ctx):
        self._context = ctx

    # -----------------------------------------------------------------
    # Path planning
    # -----------------------------------------------------------------

    def _package_plan(self, path: list[np.ndarray] | None):
        """Retime a raw geometric path and split it into per-arm trajectories.
 
        Each plan_* method below only needs to produce the raw combined
        path; this does the retime + split + wrap-into-PlanGroupResult
        that all of them do identically afterward.
        """
        if path is None:
            return None
        combined = self._arm_group.retime(path)
        split = combined.split_trajectory(self._arm_group)
        return PlanGroupResult.from_trajectories(split)

    def plan_to_configuration(self, goal, **kwargs):
        """Plan a trajectory for the arms to reach the target joint configurations."""
        return self._package_plan(self._arm_group.plan_to_configuration(goal, **kwargs))

    def plan_ee_to_pose(self, goal, **kwargs):
        """Plan a trajectory for the end effectors to reach the target poses.
 
        Note: translates to ArmGroup.plan_to_poses at the boundary -- the
        naming differs deliberately (openarm's public API is end-effector-
        pose-focused; ArmGroup's is the generic mj_manipulator name), not
        by oversight. See item C.6.
        """
        return self._package_plan(self._arm_group.plan_to_poses(goal, **kwargs))

    def plan_to_tsrs(self, goal, **kwargs):
        """Plan a trajectory for the arms to reach the target TSRs."""
        return self._package_plan(self._arm_group.plan_to_tsrs(goal, **kwargs))

    def plan_reach_to_pose(
        self,
        goal: list[np.ndarray]
    ) -> PlanGroupResult | None:
        """Plans a trajectory for the arm closest to the first waypoint in the goal list to reach the target poses."""
        if not goal:
            raise ValueError("goal must contain at least one waypoint")

        first_target_pos = np.asarray(goal[0])[:3, 3]

        closest_arm = min(
            self._arm_group.keys(),
            key=lambda name: float(
                np.linalg.norm(self._arm_group[name].get_ee_pose()[:3, 3] - first_target_pos)
            ),
        )

        return self.plan_ee_to_pose({closest_arm: goal})

    def execute(self, plan, synchronous: bool = True) -> bool:
        ctx = self._active_context
        if ctx is None:
            raise RuntimeError(
                "No active execution context. Use 'with robot.sim() as ctx:' or 'with robot.real() as ctx:'"
            )
        return ctx.execute(plan)

    def retime_plan(self, plan: PlanGroupResult) -> PlanGroupResult | None:
        """Retime a planned trajectory to respect velocity and acceleration limits."""
        self.arms.retime(plan.arm_trajectory, self.config.kinematic_limits)

    @property
    def named_poses(self) -> dict[str, dict[str, list[float]]]:
        """Named poses from config and keyframes."""
        return self._named_poses

    # -----------------------------------------------------------------
    # Gripper control
    # -----------------------------------------------------------------

    def set_gripper_width(self, target: dict[str, float], synchronous: bool = True) -> bool:
        """Set the width of the grippers.

        Args:
            target: Map of arm name -> desired gripper width, 0.0 (closed)
                to 1.0 (open).
            synchronous: On real hardware (inside `with robot.real()`),
                blocks until each physical gripper finishes moving. In
                sim, or with no active context, application is instant
                and this has no effect.
        """
        ctx = self._active_context
        for name, width in target.items():
            gripper = self._arm_group[name].gripper
            if gripper is None:
                logger.warning("No gripper found for %s", name)
                return False
            if isinstance(ctx, OpenarmRealContext):
                if not ctx.arm(name).set_width(width, synchronous=synchronous):
                    return False
            else:
                gripper.set_width(width)
        self.forward()
        return True

    def get_gripper_width(self) -> dict[str, float]:
        """Get the current width of the grippers."""
        return {
            name: arm.gripper.get_width()
            for name, arm in self._arm_group.items()
            if arm.gripper is not None
        }

    # -----------------------------------------------------------------
    # Proprioception
    # -----------------------------------------------------------------

    def check_collisions(self) -> bool:
        """Check for collisions in the current state."""
        return self.arms.check_collisions()

    def get_ee_pose(self) -> dict[str, np.ndarray]:
        """Current end-effector pose as 4x4 homogeneous transform."""
        return self.arms.get_ee_pose()

    def get_joint_positions(self) -> dict[str, np.ndarray]:
        """Current joint positions (rad)."""
        return self.arms.get_joint_positions()

    def get_ee_velocities(self) -> dict[str, np.ndarray]:
        """Current end-effector velocities (linear and angular, in world frame)."""
        return self.arms.get_ee_velocities()

    def get_joint_velocities(self) -> dict[str, np.ndarray]:
        """Current joint velocities (rad/s)."""
        return self.arms.get_joint_velocities()

    # -----------------------------------------------------------------
    # Abort
    # -----------------------------------------------------------------

    def request_abort(self) -> None:
        """Global kill switch: stop everything immediately.

        - Sets the abort flag (checked by all primitives and trajectory
          runners on the next tick)
        - Aborts all arms via the ownership registry
        - Deactivates all teleop (operator must re-activate)

        The abort flag stays set until the user starts a new command
        from the REPL, which calls ``clear_abort()``. Primitives
        check ``is_abort_requested()`` at the top and return False
        immediately without clearing it.
        """
        if not self._abort_event.is_set():
            logger.warning("E-Stop activated. All execution halted.")
        if self._context is not None and self._context.ownership is not None:
            self._context.ownership.abort_all()
        # Deactivate all teleop
        if self._context is not None and hasattr(self._context, "_event_loop"):
            loop = self._context._event_loop
            if loop is not None and hasattr(loop, "_deactivate_all_teleop"):
                loop._deactivate_all_teleop()
        self._abort_event.set()

    def clear_abort(self) -> None:
        """Clear the abort flag (call before starting a new operation)."""
        if self._abort_event.is_set():
            logger.warning("E-Stop cleared. Resuming.")
        if self._context is not None and self._context.ownership is not None:
            self._context.ownership.clear_all()
        self._abort_event.clear()

    def is_abort_requested(self) -> bool:
        """Check if a global abort has been requested."""
        return self._abort_event.is_set()

    # -----------------------------------------------------------------
    # Environment sync
    # -----------------------------------------------------------------

    def forward(self) -> None:
        """Run forward kinematics and sync viewer."""
        mujoco.mj_forward(self.model, self.data)
        if self._context is not None:
            self._context.sync()

    def setup_scene(self, fixtures: dict[str, list[list[float]]] | None = None) -> None:
        """Set up the scene: place fixtures and ready the robot."""
        fixtures = fixtures or {}
        self._fixtures = fixtures

        for obj_type, positions in fixtures.items():
            for pos in positions:
                self._env.registry.activate(obj_type, pos=list(pos))

        if "ready" in self._named_poses:
            for side, arm in self._arm_group.items():   # Mapping access, not .arms.items()
                q = np.array(self._named_poses["ready"][side])
                for i, idx in enumerate(arm.joint_qpos_indices):
                    self.data.qpos[idx] = q[i]

        self.forward()

    def reset(self) -> None:
        """Reset the scene to its initial state."""
        if self._context is not None:
            self._context.reset_to_keyframe("ready")
        else:
            key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "ready")
            if key_id != -1:
                mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
            else:
                mujoco.mj_resetData(self.model, self.data)

        for arm in self._arm_group.values():   # Mapping access
            arm._ft_tare_offset = np.zeros(6)

        if self._env.registry is not None:
            hide_pos = self._env.hide_pos
            for qpos_adr in self._freejoint_qpos_addrs:
                self.data.qpos[qpos_adr : qpos_adr + 3] = hide_pos
            for name in list(self._env.registry.active_objects):
                self._env.registry.hide(name)

        self.setup_scene(fixtures=getattr(self, "_fixtures", None))

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    def _load_keyframe_poses(self) -> dict[str, dict[str, list[float]]]:
        """Extract named poses from MuJoCo keyframes."""
        poses: dict[str, dict[str, list[float]]] = {}
        for key_id in range(self.model.nkey):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_KEY, key_id)
            if name is None:
                continue
            key_qpos = self.model.key_qpos[key_id]
            left_qpos = [float(key_qpos[i]) for i in self._arm_group["left"].joint_qpos_indices]
            right_qpos = [float(key_qpos[i]) for i in self._arm_group["right"].joint_qpos_indices]
            poses[name] = {"left": left_qpos, "right": right_qpos}
        return poses

    def _create_temp_scene_config(self, objects: dict[str, int]) -> str:
        """Create temporary scene_config.yaml from objects dict."""
        import tempfile

        import yaml

        config = {"objects": {obj_type: {"count": count} for obj_type, count in objects.items()}}
        temp_file = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".yaml",
            delete=False,
        )
        yaml.dump(config, temp_file)
        temp_file.close()
        return temp_file.name
