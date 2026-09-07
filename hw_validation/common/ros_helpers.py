"""Thin ROS 2 helpers shared by the stage scripts that talk to hardware at
the raw controller level (Stages 2-3, and the physical-e-stop path of
Stage 5). Stages that use the Python planning/execution API (Stage 4+) don't
need most of this -- they go through `openarm.robot.Openarm` directly.

Kept deliberately generic: controller and joint names are always parameters,
never hardcoded, since this file has no business knowing what a given
session's actual topology is.
"""

from __future__ import annotations

import threading
import time
import subprocess
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rclpy.task import Future


class RosSession:
    """Context manager: init rclpy, create one node, spin it in a background
    thread for the duration of the `with` block, clean shutdown after."""

    def __init__(self, node_name: str):
        self._node_name = node_name
        self.node = None
        self._executor = None
        self._thread = None

    def __enter__(self):
        import rclpy
        from rclpy.executors import MultiThreadedExecutor

        if not rclpy.ok():
            rclpy.init()
        self.node = rclpy.create_node(self._node_name)
        self._executor = MultiThreadedExecutor()
        self._executor.add_node(self.node)
        self._thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        import rclpy

        if self._executor is not None:
            self._executor.shutdown()
        if self.node is not None:
            self.node.destroy_node()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        return None


def make_streaming_publisher(node, controller_name: str):
    """Publisher for a controller's streaming joint-command topic
    (`/<controller_name>/joint_commands`, a single `JointTrajectoryPoint`
    per message -- see the impedance controller's streaming interface)."""
    from trajectory_msgs.msg import JointTrajectoryPoint

    return node.create_publisher(JointTrajectoryPoint, f"/{controller_name}/joint_commands", 10)


def send_streaming_point(pub, joint_names: list[str], positions: list[float]) -> None:
    from trajectory_msgs.msg import JointTrajectoryPoint

    msg = JointTrajectoryPoint()
    msg.positions = [float(p) for p in positions]
    pub.publish(msg)


def ramp_stream(
    node,
    pub,
    joint_names: list[str],
    q_start: list[float],
    q_end: list[float],
    duration_s: float,
    rate_hz: float = 50.0,
    stop_flag: threading.Event | None = None,
) -> None:
    """Publish a smooth linear ramp from q_start to q_end over duration_s.
    Blocks for the duration. Pass `stop_flag` (an Event) to allow external
    early termination (used by the abort-latency test)."""
    n_steps = max(2, int(duration_s * rate_hz))
    period = 1.0 / rate_hz
    q_start_arr = list(q_start)
    q_end_arr = list(q_end)
    for i in range(n_steps + 1):
        if stop_flag is not None and stop_flag.is_set():
            return
        alpha = i / n_steps
        q = [s + alpha * (e - s) for s, e in zip(q_start_arr, q_end_arr)]
        send_streaming_point(pub, joint_names, q)
        time.sleep(period)


def make_trajectory_client(node, controller_name: str):
    from control_msgs.action import FollowJointTrajectory
    from rclpy.action import ActionClient

    return ActionClient(node, FollowJointTrajectory, f"/{controller_name}/follow_joint_trajectory")


@dataclass
class TrajectoryDispatchHandle:
    """Returned by `send_trajectory_async` -- lets a caller poll status or
    request cancellation of a goal genuinely in flight, which is exactly
    the capability this test suite needs to exercise for the abort-latency
    checks (and which application code may or may not currently provide --
    that's exactly the kind of thing Stage 5 exists to find out)."""

    goal_future: Future
    result_future: Future | None = None
    goal_handle: Any = None

    def wait_for_result(self, node, timeout_s: float = 60.0):
        import rclpy

        if self.goal_handle is None:
            rclpy.spin_until_future_complete(node, self.goal_future, timeout_sec=10.0)
            self.goal_handle = self.goal_future.result()
            if self.goal_handle is None or not self.goal_handle.accepted:
                return None
            self.result_future = self.goal_handle.get_result_async()
        result_future = self.result_future
        if result_future is None:
            return None
        rclpy.spin_until_future_complete(node, result_future, timeout_sec=timeout_s)
        return result_future.result()

    def cancel(self, node, timeout_s: float = 5.0) -> bool:
        import rclpy

        if self.goal_handle is None:
            return False
        cancel_future = self.goal_handle.cancel_goal_async()
        rclpy.spin_until_future_complete(node, cancel_future, timeout_sec=timeout_s)
        return cancel_future.result() is not None


def send_trajectory(node, client, joint_names: list[str], waypoints: list[dict]) -> TrajectoryDispatchHandle:
    """waypoints: list of {'positions': [...], 'time_from_start': seconds}."""
    from builtin_interfaces.msg import Duration
    from control_msgs.action import FollowJointTrajectory
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

    goal = FollowJointTrajectory.Goal()
    goal.trajectory = JointTrajectory()
    goal.trajectory.joint_names = list(joint_names)
    for wp in waypoints:
        pt = JointTrajectoryPoint()
        pt.positions = [float(x) for x in wp["positions"]]
        t = wp["time_from_start"]
        pt.time_from_start = Duration(sec=int(t), nanosec=int((t - int(t)) * 1e9))
        goal.trajectory.points.append(pt)  # pyright: ignore[reportAttributeAccessIssue]

    goal_future = client.send_goal_async(goal)
    handle = TrajectoryDispatchHandle(goal_future=goal_future)
    # Resolve the goal handle right away so `.cancel()` can be called by a
    # concurrent watcher thread without waiting on `wait_for_result` first.
    import rclpy

    rclpy.spin_until_future_complete(node, goal_future, timeout_sec=10.0)
    handle.goal_handle = goal_future.result()
    if handle.goal_handle is not None and handle.goal_handle.accepted:
        handle.result_future = handle.goal_handle.get_result_async()
    return handle


def get_hardware_component_states(node) -> dict:
    """Returns {component_name: state_label} via
    controller_manager_msgs/ListHardwareComponents. Service name/type follow
    the standard ros2_control controller_manager; if a given distro/setup
    names it differently, override via the CLI args on the calling script
    rather than editing this function.
    """
    from controller_manager_msgs.srv import ListHardwareComponents
    import rclpy

    client = node.create_client(ListHardwareComponents, "/controller_manager/list_hardware_components")
    if not client.wait_for_service(timeout_sec=5.0):
        raise RuntimeError(
            "/controller_manager/list_hardware_components not available -- is ros2_control_node running?"
        )
    future = client.call_async(ListHardwareComponents.Request())
    rclpy.spin_until_future_complete(node, future, timeout_sec=5.0)
    result = future.result()
    return {c.name: c.state.label for c in result.component}

def spawn_controller(controller_name: str, timeout_s: float = 15.0) -> bool:
    """Spawn one controller via the standard `controller_manager spawner`
    CLI, e.g. `joint_state_broadcaster` -- and only that one. This is how
    Stage 1's sensor-liveness check gets live /joint_states data while the
    underlying hardware component stays 'inactive' (motors never enabled):
    joint_state_broadcaster is a *controller* with its own lifecycle,
    separate from the hardware component's, and reads whatever the hardware
    interface's read() populates regardless of whether the hardware
    component itself has been activated.
 
    Assumes a global (not per-arm-namespaced) controller_manager, matching
    this robot's actual topology -- adjust the `--controller-manager` arg
    below if that's ever not the case.
    """
    result = subprocess.run(
        ["ros2", "run", "controller_manager", "spawner", controller_name,
         "--controller-manager", "/controller_manager"],
        capture_output=True, text=True, timeout=timeout_s,
    )
    return result.returncode == 0

def list_controllers(node) -> dict:
    """Returns {controller_name: state_label} via
    controller_manager_msgs/ListControllers against the global
    /controller_manager (not per-arm-namespaced -- see spawn_controller)."""
    from controller_manager_msgs.srv import ListControllers
    import rclpy
 
    client = node.create_client(ListControllers, "/controller_manager/list_controllers")
    if not client.wait_for_service(timeout_sec=5.0):
        raise RuntimeError("/controller_manager/list_controllers not available")
    future = client.call_async(ListControllers.Request())
    rclpy.spin_until_future_complete(node, future, timeout_sec=5.0)
    result = future.result()
    return {c.name: c.state for c in result.controller}


def set_hardware_component_state(node, component_name: str, target_label: str) -> bool:
    """Explicitly transition one hardware component's lifecycle state
    (e.g. 'inactive' -> 'active'). This is the operator-gated activation
    mechanism Stage 2 uses instead of relying on whatever a launch file's
    default auto-activation behavior happens to be.
    """
    from controller_manager_msgs.srv import SetHardwareComponentState
    from lifecycle_msgs.msg import State
    import rclpy

    client = node.create_client(SetHardwareComponentState, "/controller_manager/set_hardware_component_state")
    if not client.wait_for_service(timeout_sec=5.0):
        raise RuntimeError("/controller_manager/set_hardware_component_state not available")
    req = SetHardwareComponentState.Request()
    req.name = component_name
    req.target_state = State(label=target_label)
    future = client.call_async(req)
    rclpy.spin_until_future_complete(node, future, timeout_sec=10.0)
    result = future.result()
    return bool(result and result.ok)