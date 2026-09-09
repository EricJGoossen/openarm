"""Sync an Openarm's internal planning shadow to real hardware feedback.

Shared by any stage script that plans through `openarm.robot.Openarm`
(`plan_to_configuration`/`real()`/`execute()`): the shadow MuJoCo model that
planning reads and writes is never otherwise synced from the real arm's
measured position. Left unsynced, planning starts from whatever the shadow
happens to hold -- a fixed assumed pose, or wherever execute() nominally
commanded it, which can differ from where the arm's impedance controller
actually settled (especially after an abort/e-stop, which can leave the arm
partway through a trajectory rather than at its planned endpoint). That
drift is what produced "goal rejected"/"path tolerance violated" from the
real controller during Stage 4 bring-up: the planned trajectory's first
waypoint didn't match the controller's own measured starting state closely
enough. Call this immediately before computing a goal and before every
plan, not just once per session -- and note that `robot.reset()` alone does
NOT re-sync from hardware, it only resets the local shadow to a fixed named
pose, so calling reset() per-trial without this actively re-introduces the
same drift.
"""

from __future__ import annotations

import mujoco

from common.telemetry import JointStateRecorder


def sync_shadow_from_hardware(robot, arm: str, recorder: JointStateRecorder, timeout_s: float = 5.0) -> dict:
    """Overwrite the shadow model's qpos for `arm` with a fresh real-hardware
    reading from `/joint_states`."""
    arm_scope = getattr(robot, arm).arm
    joint_names = list(arm_scope.config.joint_names)
    real = recorder.wait_for_joints(joint_names, timeout_s=timeout_s)
    if real is None:
        raise RuntimeError(
            f"Could not read real joint state for {joint_names} within {timeout_s}s -- "
            "refusing to plan against a stale/unsynced shadow position."
        )
    for name, idx in zip(joint_names, arm_scope.joint_qpos_indices):
        robot.data.qpos[idx] = real[name]
    mujoco.mj_forward(robot.model, robot.data)
    return real
