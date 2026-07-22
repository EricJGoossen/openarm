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
    PlanResult,
    SimContext,
)

from mj_manipulator.config import ArmConfig, KinematicLimits
from mj_manipulator.grasp_verifier import GraspVerifier
from mj_manipulator.config import PhysicsConfig

from openarm.config import OpenarmConfig, OpenarmArmSpec, OpenarmGripperSpec, setup_logging
# TODO(mast): OpenarmMast/the linear actuator base doesn't exist as a real
# asset in openarm_assets yet -- config.OpenarmConfig.default() leaves
# base=None, and every self._base use below is None-guarded accordingly.
# Once a real mast/base MJCF + joint/actuator names exist, wire it up here
# the same way Geodude's VentionBase is wired (see this file's own
# _get_base_for_arm / setup_scene / _plan_single for the exact spots that
# currently no-op on a None base).
from openarm.openarm_mast import OpenarmMast
from openarm.openarm_gripper import OpenarmGripper

logger = logging.getLogger(__name__)

class _OpenarmSimContext:
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
    
class _OpenarmHardwareContext:
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
        return _DualArmController(self._shadow.arm(name), self._hw.arm(name))

    @property
    def control_dt(self):
        return self._hw.control_dt  # real cadence (500 Hz), not sim's 250 Hz default
    
class _DualArmController:
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

class _ArmScope:
    """Unified arm interface — high-level primitives + low-level Arm access.

    Returned by ``robot.right`` / ``robot.left``. Provides openarm-specific
    methods (pickup, place, close, open) and delegates everything else to
    the underlying mj_manipulator Arm via ``__getattr__``::

        robot.left.pickup("can")        # openarm primitive
        robot.left.close()              # gripper control
        robot.left.get_ee_pose()        # Arm method (delegated)
        robot.left.get_ft_wrench()      # Arm method (delegated)
        robot.left.plan_to_pose(target) # Arm method (delegated)

    Class-level annotations expose delegated Arm methods to IPython/Jedi
    for tab completion (Jedi uses static analysis and can't follow __getattr__).
    """

    def __init__(self, robot: "Openarm", side: str) -> None:
        self._robot = robot
        self._side = side

    @property
    def _arm(self):
        return self._robot._resolve_arm(self._side)
    
    def __getattr__(self, name: str) -> Any:
        return getattr(self._arm, name)
    
    def __dir__(self) -> list[str]:
        return sorted(set(super().__dir__()) | set(dir(self._arm)))
    
    def get_ee_pose(self):
        """Current end-effector pose as 4x4 homogeneous transform."""
        return self._arm.get_ee_pose()
    
    def get_ft_wrench(self):
        """Wrist F/T reading [fx,fy,fz,tx,ty,tz] in sensor local frame."""
        return self._arm.get_ft_wrench()
    
    def get_ft_wrench_world(self):
        """Wrist F/T reading [fx,fy,fz,tx,ty,tz] in world frame."""
        return self._arm.get_ft_wrench_world()
    
    def tare_ft(self):
        """Zero the F/T sensor at the current reading."""
        return self._arm.tare_ft()
    
    def get_joint_positions(self):
        """Current joint positions (rad)."""
        return self._arm.get_joint_positions()
    
    def set_joint_positions(self, q):
        """Set joint positions directly (sim only). Use plan_to_configuration() on hardware."""
        return self._arm.set_joint_positions(q, ctx=self._robot._active_context)
    
    def get_joint_velocities(self):
        """Current joint velocities (rad/s)."""
        return self._arm.get_joint_velocities()
    
    def get_joint_limits(self):
        """Joint position limits as (lower, upper) arrays."""
        return self._arm.get_joint_limits()
    
    def forward_kinematics(self, q):
        """Compute end-effector pose for a given joint configuration."""
        return self._arm.forward_kinematics(q)
    
    def plan_to_pose(self, pose, **kwargs):
        """Plan a collision-free path to an end-effector pose."""
        return self._arm.plan_to_pose(pose, **kwargs)
    
    def plan_to_configuration(self, q_goal, **kwargs):
        """Plan a collision-free path to a joint configuration."""
        return self._arm.plan_to_configuration(q_goal, **kwargs)
    
    def plan_to_tsrs(self, tsrs, **kwargs):
        """Plan a collision-free path to a TSR-defined goal region."""
        return self._arm.plan_to_tsrs(tsrs, **kwargs)
    
    def retime(self, plan, **kwargs):
        """Retime a joint path into a smooth trajectory."""
        return self._arm.retime(plan, **kwargs)
    
    def check_collisions(self):
        """Check current configuration for collisions. Prints a summary."""
        return self._arm.check_collisions()

    @property
    def has_ft_sensor(self) -> bool:
        """Whether this arm has a wrist F/T sensor configured."""
        return self._arm.has_ft_sensor

    @property
    def gripper(self):
        """The arm's gripper."""
        return self._arm.gripper

    @property
    def config(self) -> ArmConfig:
        """Arm configuration (joint names, limits, ee_site, etc.)."""
        return self._arm.config

    @property
    def grasp_manager(self) -> GraspManager | None:
        """GraspManager for grasp state queries."""
        return self._arm.grasp_manager

    @property
    def ee_site_id(self) -> int:
        """MuJoCo site ID for the end-effector."""
        return self._arm.ee_site_id

    # TODO: implement primvitives
    
    def close(self) -> str | None:
        """Close the gripper. Grasps whatever is between the fingers.

        Returns:
            Name of grasped object, or None if nothing detected.
        """
        ctx = self._robot._active_context
        if ctx is None:
            raise RuntimeError("No active execution context. Use 'with robot.sim() as ctx:'")
        return ctx.arm(self._side).grasp()
    
    def open(self) -> None:
        """Open the gripper. Releases whatever is held."""
        ctx = self._robot._active_context
        if ctx is None:
            raise RuntimeError("No active execution context. Use 'with robot.sim() as ctx:'")
        ctx.arm(self._side).release()

class Openarm:
    """High-level interface for the Openarm bimanual robot.

    Provides:
    - Two OpenArm arms with their native parallel-jaw grippers (from mj_manipulator)
    - Optional linear actuator base (TODO(mast): not yet available -- see
      the import comment above and setup_scene/_plan_single below)
    - Bimanual planning with arm/height interleaving
    - Named configurations from MuJoCo keyframes
    - Affordance-driven pickup/place primitives

    Example::

        robot = Openarm(objects={"can": 2, "recycle_bin": 1})
        with robot.sim(physics=True) as ctx:
            robot.pickup("can_0")
            robot.place("recycle_bin_0")
    """  

    config: OpenarmConfig
    model: mujoco.MjModel
    data: mujoco.MjData
    grasp_manager: GraspManager

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
        self._left_arm = self._create_arm(self.config.left_arm, "left")
        self._right_arm = self._create_arm(self.config.right_arm, "right")

        # Cache freejoint qpos addresses (for hide-all in reset)
        self._freejoint_qpos_addrs = [
            self.model.jnt_qposadr[i]
            for i in range(self.model.njnt)
            if self.model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE
        ]

        # Create base (if configured).
        # TODO(mast): self.config.base is always None right now -- the
        # linear actuator base doesn't exist as a real MJCF asset yet in
        # openarm_assets. This branch is correct as written (it'll just
        # work once a real OpenarmMastConfig with valid joint_name/
        # actuator_name is passed in), but currently never executes.
        self._base: OpenarmMast | None = None
        if self.config.base is not None:
            self._base = OpenarmMast(
                self.model,
                self.data,
                self.config.base,
                [self._left_arm, self._right_arm]
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

    # -----------------------------------------------------------------
    # Abort mechanism
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
        # Deactivate all telop
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

    # Properties

    @property
    def left(self) -> _ArmScope:
        """Left arm: ``robot.left.pickup("can")``, ``robot.left.get_ee_pose()``."""
        return _ArmScope(self, "left")
    
    @property
    def right(self) -> _ArmScope:
        """Right arm: ``robot.right.pickup("can")``, ``robot.right.get_ee_pose()``."""
        return _ArmScope(self, "right")
    
    @property
    def base(self) -> OpenarmMast | None:
        """Linear actuator base. TODO(mast): always None until a real
        base MJCF asset exists -- see the TODO(mast) notes in __init__.
        """
        return self._base
    
    @property
    def env(self) -> Environment:
        """Underlying MuJoCo environment."""
        return self._env
    
    @property
    def arms(self) -> dict[str, Arm]:
        """All arms, keyed by side name."""
        return {"left": self._left_arm, "right": self._right_arm}
    
    # TODO: add grasp_source

    @property
    def named_poses(self) -> dict[str, dict[str, list[float]]]:
        """Named poses from config and keyframes."""
        return self._named_poses
    
    @property
    def _active_context(self) -> SimContext | None:
        """Currently active execution context (set by sim())."""
        return self._context
    
    @_active_context.setter
    def _active_context(self, ctx: SimContext | None) -> None:
        self._context = ctx

    def get_arm_spec(self, arm: Arm) -> OpenarmArmSpec:
        """Get the OpenarmArmSpec for an arm (has hand_type, prefix, etc.)."""
        if arm is self._left_arm:
            return self.config.left_arm
        return self.config.right_arm
    
    # Arm resolution

    def _resolve_arms(self, arm: Arm | str | None) -> list[Arm]:
        """Resolve arm specification to list of Arm instances."""
        if arm is None:
            return [self._right_arm, self._left_arm]
        if isinstance(arm, str):
            if arm in ("left", "left_arm"):
                return [self._left_arm]
            if arm in ("right", "right_arm"):
                return [self._right_arm]
            raise ValueError(f"Unknown arm name: {arm}")
        if isinstance(arm, Arm):
            return [arm]
        raise TypeError(f"Invalid arm specification: {arm}")
    
    def _resolve_arm(self, arm: Arm | str) -> Arm:
        """Resolve a single arm specification to an Arm instance."""
        if isinstance(arm, Arm):
            return arm
        if arm in ("left", "left_arm"):
            return self._left_arm
        if arm in ("right", "right_arm"):
            return self._right_arm
        raise ValueError(f"Unknown arm name: {arm}")
    
    def arm_name(self, arm: Arm) -> str:
        """Get the side name for an arm ('left' or 'right')."""
        if arm is self._left_arm:
            return "left"
        return "right"
    
    # Simulation context

    def sim(
        self,
        physics: bool = True,
        viewer=None,
        viewer_fps: float = 30.0,
        headless: bool = False,
        event_loop=None,
        physics_config: "PhysicsConfig | None" = None,
    ) -> _OpenarmSimContext:
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
        arms = {"left": self._left_arm, "right": self._right_arm}
        entities = {}
        # TODO(mast): self._base is always None right now -- see __init__.
        if self._base is not None:
            entities[self._base.config.name] = self._base
        inner = SimContext(
            self.model,
            self.data,
            arms,
            physics=physics,
            headless=headless,
            viewer=viewer,
            viewer_fps=viewer_fps,
            entities=entities,
            abort_fn=self.is_abort_requested,
            event_loop=event_loop,
            physics_config=physics_config,
        )
        return _OpenarmSimContext(inner, self)
    
    def real(
        self,
        node_name: str = "openarm_hardware",
        event_loop=None,
        physics_config: "PhysicsConfig | None" = None,
    ) -> _OpenarmHardwareContext:
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

        arms = {"left": self._left_arm, "right": self._right_arm}
        shadow = SimContext(
            self.model,
            self.data,
            arms,
            physics=False,
            headless=True,
            abort_fn=self.is_abort_requested,
            event_loop=event_loop,
            physics_config=physics_config,
        )
        hw = HardwareContext(self.config.to_hardware_config(), node_name=node_name)
        return _OpenarmHardwareContext(hw, shadow, self)
    
    # Named poses

    def go_to(self, pose_name: str, ctx: SimContext | None = None) -> bool:
        """Move both arms to a named configuration.

        Args:
            pose_name: Name of the configuration (e.g., 'ready').
            ctx: Execution context. If None, uses the active context.

        Returns:
            True if both arms planned and executed successfully.
        """
        if pose_name not in self.named_poses:
            raise ValueError(f"Unknown named pose: {pose_name}")

        ctx = ctx or self._active_context
        if ctx is None:
            raise RuntimeError("No active execution context. Use robot.sim().")

        pose = self.named_poses[pose_name]
        success = True
        for side, arm in [("left", self._left_arm), ("right", self._right_arm)]:
            if side in pose:
                q_goal = np.array(pose[side])
                path = arm.plan_to_configuration(q_goal)
                if path is not None:
                    traj = arm.retime(path)
                    ctx.execute(traj)
                else:
                    logger.warning("Failed to plan %s arm to '%s'", side, pose_name)
                    success = False
        return success
    
    # Planning 

    def plan_to_tsrs(
        self,
        goal_tsrs,
        *,
        arm: Arm | str | None = None,
        base_heights: list[float] | None = None,
        strategy: str = "first",
        timeout: float | None = None,
        seed: int | None = None,
    ) -> PlanResult | None:
        """Plan to TSR goals with arm/height interleaving.

        If no arm is specified, tries both arms. If base_heights are given,
        tries each arm at each height.

        TODO(mast): base_heights is a no-op right now since self._base is
        always None -- _plan_single below already guards on
        `self._base is not None` before touching base height at all, so
        passing base_heights today just costs a wasted planning pass per
        height value with no base movement behind it. Harmless, just
        pointless until a real base exists.

        Returns:
            PlanResult with arm trajectory (and base trajectory if height changed),
            or None if all attempts failed.
        """
        return self._plan_with_sequence(
            goal_tsrs=goal_tsrs,
            arm=arm,
            base_heights=base_heights,
            strategy=strategy,
            timeout=timeout,
            seed=seed,
        )
    
    def plan_to_pose(
        self,
        pose: np.ndarray,
        *,
        arm: Arm | str | None = None,
        base_heights: list[float] | None = None,
        strategy: str = "first",
        timeout: float | None = None,
        seed: int | None = None,
    ) -> PlanResult | None:
        """Plan to an EE pose with arm/height interleaving."""
        return self._plan_with_sequence(
            pose=pose,
            arm=arm,
            base_heights=base_heights,
            strategy=strategy,
            timeout=timeout,
            seed=seed,
        )
    
    def _plan_with_sequence(
        self,
        *,
        goal_tsrs=None,
        pose: np.ndarray | None = None,
        arm: Arm | str | None = None,
        base_heights: list[float] | None = None,
        strategy: str = "first",
        timeout: float | None = None,
        seed: int | None = None,
    ) -> PlanResult | None:
        """Core planning: try arm/height combinations, return first or best."""
        arms = self._resolve_arms(arm)

        # Build (arm, height) sequence
        heights = base_heights if base_heights else [None]
        sequence: list[tuple[Arm, float | None]] = []

        # Randomize arm order for fairness
        arms_shuffled = list(arms)
        random.shuffle(arms_shuffled)

        for h in heights:
            for a in arms_shuffled:
                sequence.append((a, h))

        # Try each combination
        results: list[PlanResult] = []
        for a, h in sequence:
            result = self._plan_single(a, h, goal_tsrs=goal_tsrs, pose=pose, timeout=timeout, seed=seed)
            if result is not None:
                if strategy == "first":
                    return result
                results.append(result)

        if not results:
            return None

        # "best" strategy: pick shortest arm trajectory
        return min(results, key=lambda r: r.arm_trajectory.duration)
    
    def _plan_single(
        self,
        arm: Arm,
        height: float | None,
        *,
        goal_tsrs=None,
        pose: np.ndarray | None = None,
        timeout: float | None = None,
        seed: int | None = None,
    ) -> PlanResult | None:
        """Plan with a single arm at a specific base height.

        Plans in a forked environment with the base at the target height,
        so live state is never mutated. The base trajectory is included
        in the PlanResult so ctx.execute() moves the base properly.

        TODO(mast): self._base is always None right now, so the
        `height is not None and self._base is not None` guards below
        always take the "no base" path -- height is accepted but silently
        ignored. Not broken, just inert until a real base exists.
        """
        base_traj = None

        # Plan base trajectory on live state (read-only collision query)
        if height is not None and self._base is not None:
            current_height = self._base.get_height()
            if abs(current_height - height) > 0.001:
                base_traj = self._base.plan_to(height, check_collisions=True)
                if base_traj is None:
                    return None  # path blocked by collision

        # Fork env and set base to target height for arm planning.
        # No live state mutation — the fork is discarded after planning.
        planning_env = self._env.fork()
        if height is not None and self._base is not None:
            planning_env.data.qpos[self._base._qpos_idx] = height
            mujoco.mj_forward(planning_env.model, planning_env.data)

        if timeout is None:
            timeout = self.config.planning.timeout

        # Plan arm in the fork (base at target height, everything else live)
        try:
            config = arm._make_planner_config(timeout, None)
            planner = arm.create_planner(config, planning_env=planning_env)
            start = arm.get_joint_positions()

            if goal_tsrs is not None:
                tsrs = goal_tsrs if isinstance(goal_tsrs, list) else [goal_tsrs]
                path = planner.plan(start=start, goal_tsrs=tsrs, seed=seed)
            elif pose is not None:
                path = planner.plan(start=start, goal_tsrs=[arm._make_pose_tsr(pose)], seed=seed)
            else:
                raise ValueError("Must provide goal_tsrs or pose")
        except Exception as e:
            logger.info("Planning failed: %s", e)
            path = None

        if path is None:
            return None

        arm_traj = arm.retime(path)

        return PlanResult(
            arm_name=arm.config.name,
            arm_trajectory=arm_traj,
            base_trajectory=base_traj,
            base_height=height,
        )
    
    # Scene setup

    def setup_scene(
        self,
        fixtures: dict[str, list[list[float]]] | None = None,
    ) -> None:
        """Set up the scene: place fixtures and ready the robot.

        Activates fixture objects at specified positions, sets bases to
        midpoint height, and arms to "ready" keyframe.

        Args:
            fixtures: Stationary objects and their positions, e.g.
                ``{"recycle_bin": [[0.85, -0.35, 0.01], [-0.85, -0.35, 0.01]]}``
        """
        fixtures = fixtures or {}
        self._fixtures = fixtures

        # Tell the perception service which types are fixtures so
        # refresh() preserves them (they're not detected by perception).
        # self._perception._fixture_types = set(fixtures.keys()) # TODO: add back in perception

        # 1. Activate fixtures at specified positions
        for obj_type, positions in fixtures.items():
            for pos in positions:
                self._env.registry.activate(obj_type, pos=list(pos))

        # 2. Set bases to midpoint
        # TODO(mast): no-op until a real base exists -- self._base is
        # always None right now (see __init__). Once a base is added,
        # implement OpenarmMast.set_midpoint_height() and uncomment.
        # if self._base is not None:
        #     self._base.set_midpoint_height()

        # 3. Set arms to ready keyframe
        if "ready" in self._named_poses:
            for side, arm in [("left", self._left_arm), ("right", self._right_arm)]:
                q = np.array(self._named_poses["ready"][side])
                for i, idx in enumerate(arm.joint_qpos_indices):
                    self.data.qpos[idx] = q[i]

        self.forward()

    def holding(self) -> tuple[str, str] | None:
        """Return which arm is holding an object.

        Returns:
            ``(side, object_name)`` if either arm is holding, else ``None``.

        Example::

            result = robot.holding()
            if result:
                side, obj = result
                print(f"{side} arm is holding {obj}")
        """
        for side in ("left", "right"):
            held = list(self.grasp_manager.get_grasped_by(side))
            if held:
                return (side, held[0])
        return None
    
    # Scene queries

    # TODO: add grasp source
    # def find_objects(self, target: str | None = None) -> list[str]:
    #     """Find objects in the scene.

    #     Args:
    #         target: "can_0" (specific), "can" (type), None (all graspable).

    #     Returns:
    #         List of body names on the table (active, not grasped, not hidden).

    #     Example::

    #         robot.find_objects()         # ['can_0', 'can_1', 'spam_can_0']
    #         robot.find_objects("can")    # ['can_0', 'can_1']
    #     """
    #     objects = self.grasp_source.get_graspable_objects()
    #     if target is not None:
    #         # Filter: exact match (instance) or prefix match (type)
    #         objects = [o for o in objects if o == target or o.startswith(target + "_")]
    #     return objects

    # Primitives TODO: implement primitives

    # State management

    def forward(self) -> None:
        """Run forward kinematics and sync viewer."""
        mujoco.mj_forward(self.model, self.data)
        if self._context is not None:
            self._context.sync()

    def reset(self) -> None:
        """Reset the scene to its initial state.

        Hides all objects, releases grasps, restores fixtures to their
        original positions, and sets the robot to the ready keyframe.
        Call ``_spawn_manipulable_objects`` after to re-scatter objects.

        For just returning the robot to home, use ``robot.go_home()``
        which plans and executes through the context.
        """
        # Reset MuJoCo state + deactivate teleop, release grasps
        if self._context is not None:
            self._context.reset_to_keyframe("ready")
        else:
            key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "ready")
            if key_id != -1:
                mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
            else:
                mujoco.mj_resetData(self.model, self.data)

        # Clear F/T tare
        for arm in [self._left_arm, self._right_arm]:
            arm._ft_tare_offset = np.zeros(6)

        # Move ALL freejoint bodies to hide position. The keyframe reset put
        # them at the origin; setup_scene and _spawn_manipulable_objects will
        # activate the ones that should be visible.
        if self._env.registry is not None:
            hide_pos = self._env.hide_pos
            for qpos_adr in self._freejoint_qpos_addrs:
                self.data.qpos[qpos_adr : qpos_adr + 3] = hide_pos
            # Clear the registry's active state
            for name in list(self._env.registry.active_objects):
                self._env.registry.hide(name)

        # Re-setup scene (fixtures + robot pose — calls forward() internally).
        # setup_scene may modify qpos (base heights, arm positions).
        # reset_to_keyframe deferred the hold, so the next tick captures
        # whatever qpos exists after setup_scene — no manual hold needed.
        self.setup_scene(fixtures=getattr(self, "_fixtures", None))

    def reset_to_keyframe(self, name: str) -> None:
        """Reset robot to a MuJoCo keyframe by name."""
        if self._context is not None:
            self._context.reset_to_keyframe(name)
        else:
            key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, name)
            if key_id == -1:
                raise ValueError(f"Keyframe '{name}' not found in model")
            mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
            self.forward()

    # TODO: implement perception

    # Internal helpers

    def _load_keyframe_poses(self) -> dict[str, dict[str, list[float]]]:
        """Extract named poses from MuJoCo keyframes."""
        poses: dict[str, dict[str, list[float]]] = {}
        for key_id in range(self.model.nkey):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_KEY, key_id)
            if name is None:
                continue
            key_qpos = self.model.key_qpos[key_id]
            left_qpos = [float(key_qpos[i]) for i in self._left_arm.joint_qpos_indices]
            right_qpos = [float(key_qpos[i]) for i in self._right_arm.joint_qpos_indices]
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