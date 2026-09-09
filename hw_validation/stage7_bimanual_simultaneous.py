#!/usr/bin/env python3
"""Stage 7 -- Bimanual simultaneous, collision-safe trajectory execution.

The target capability. What this proves, per the Definition of Done:
  - a jointly-planned bimanual trajectory actually executes with both arms
    moving over genuinely overlapping time, not sequentially -- checked
    numerically from telemetry, not eyeballed;
  - the recorded, *executed* joint states of both arms are independently
    verified collision-free by a fresh collision checker at every sampled
    instant -- the planner's guarantee is re-proven against reality, not
    assumed to transfer;
  - a goal pair specifically constructed to collide under naive/independent
    execution is executed jointly and confirmed NOT to collide for real;
  - if one arm faults mid-trajectory, the other's response matches a
    behavior you declare in advance -- not whatever happens to occur.

Three phases, selectable via --phase (default: all, run in order):

    basic        a general jointly-planned bimanual round trip, checked for
                 simultaneity, endpoint accuracy, and collision-freedom.
    adversarial  searches for (or loads) a goal pair that would collide
                 under naive independent straight-line motion, executes it
                 jointly on real hardware, and confirms no collision.
    fault        aborts a joint execution mid-flight and checks the other
                 arm's response against a behavior you declare with
                 --expected-behavior.

A note on "confirmed to have actually been at risk of collision, not just
in simulation": there is no safe way to literally run the dangerous naive
version on real hardware -- doing so would defeat the point. What this
script does instead is find the adversarial pair using the *same* geometric
collision model the planner itself uses, built from the real robot's
as-built geometry -- the best available proxy for "physically" without
actually colliding the arms. That caveat is worth keeping in mind when
reading a pass here.

Usage:
    python3 stage7_bimanual_simultaneous.py --phase all --repeats 10 \\
        --expected-behavior both-stop
"""

from __future__ import annotations

import argparse
import json
import threading
import time

import mujoco
import numpy as np

from common.checks import first_motion_time, position_reached, settle_time_after, simultaneity_overlap_fraction
from common.collision_check import check_pair_collision_free, default_openarm_factory
from common.hardware_sync import sync_shadow_from_hardware
from common.operator_io import banner, confirm, confirm_phrase, confirm_step, safety_banner, StepResult
from common.telemetry import JointStateRecorder, TelemetrySession
from common.trial_runner import TrialOutcome, run_repeated_trials


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--phase", choices=["basic", "adversarial", "fault", "all"], default="all")
    p.add_argument("--repeats", type=int, default=10, help="spec minimum for 'basic': 10")
    p.add_argument("--goal-delta-rad", type=float, default=0.3)
    p.add_argument(
        "--position-tolerance-rad", type=float, default=0.06,
        help="Stage 4 bring-up found the real onboard motor control (P+D, no integral term) has a "
        "genuine steady-state error floor from imperfect gravity compensation -- 0.01 (this arg's "
        "original default) is tighter than what pure P control can actually achieve at some poses.",
    )
    p.add_argument("--simultaneity-start-tolerance-s", type=float, default=0.5)
    p.add_argument("--simultaneity-min-overlap-fraction", type=float, default=0.6)
    p.add_argument(
        "--velocity-threshold", type=float, default=0.08,
        help="reported /joint_states velocity is quantized/noisy even at true zero motion -- Stage 5 "
        "measured up to ~0.055 rad/s noise on a bit-exact-frozen joint. Used both to detect motion "
        "onset (first_motion_time/simultaneity) and motion end (settle_time_after in the fault phase); "
        "0.02 is tighter than the noise itself and can misfire either detection.",
    )
    p.add_argument(
        "--settle-time-s", type=float, default=2.0,
        help="extra wait after execute() returns before checking convergence -- execute() returning "
        "only means the goal's nominal duration elapsed, not that the arm has actually converged.",
    )
    p.add_argument(
        "--post-abort-settle-s", type=float, default=1.5,
        help="extra recording time after the fault-phase abort's execute() thread joins, before "
        "checking settle_time_after() -- it needs sustain_s of post-stop data to confirm a settle "
        "at all (see stage5_abort_and_estop.py's identical flag).",
    )
    p.add_argument("--collision-check-dt", type=float, default=0.02)
    p.add_argument("--adversarial-goals-file", default=None, help="pre-found {\"left\": [...], \"right\": [...]}")
    p.add_argument(
        "--adversarial-max-search-attempts", type=int, default=5000,
        help="TODO(review): raised from 300 -- search_adversarial_pair() now IK-constructs candidates "
        "aimed at the same shared point (see its docstring) instead of drawing two independent "
        "full-range-random joint vectors, so a much larger fraction of attempts are worth trying; a "
        "bigger budget costs comparatively little in exchange for reliably finding a genuine cross-arm "
        "case instead of giving up.",
    )
    p.add_argument("--abort-is-global", action="store_true", default=True,
                   help="whether this system's abort mechanism affects both arms at once (see Stage 6's observation)")
    p.add_argument("--expected-behavior", choices=["both-stop", "left-stops-right-continues", "right-stops-left-continues"],
                   default=None, help="required for --phase fault/all: the documented, declared design behavior")
    p.add_argument("--log-dir", default="bringup_logs")
    return p.parse_args()


def build_robot():
    from openarm.config import OpenarmConfig
    from openarm.robot import Openarm

    return Openarm(config=OpenarmConfig.default())


def joint_names_for(robot, arm: str) -> list[str]:
    return list(getattr(robot, arm).arm.config.joint_names)


def _signed_delta_for_joint(arm: str, joint_name: str, magnitude: float) -> float:
    """Same directional convention as stage4_trajectory_execution.py's
    get_signed_delta: left arm joints move positive, right arm joints move
    negative, except joint6 (always opposite the arm's default sign) and
    joint4 (always positive -- its real resting position sits right at its
    own 0.0 lower limit, so a negative draw there is dead-on-arrival)."""
    sign = 1.0 if arm == "left" else -1.0
    if "joint6" in joint_name:
        sign = -sign
    if "joint4" in joint_name:
        sign = 1.0
    return sign * abs(magnitude)


def make_goal(robot, arm: str, rng: np.random.Generator, delta: float, max_attempts: int = 20) -> np.ndarray:
    """Draw a random per-joint goal for `arm`, resampling away from
    self-collision or collision with the OTHER arm's CURRENT pose --
    see stage4_trajectory_execution.py's make_goal() for why independent
    per-joint draws need this.

    TODO(review): this alone is NOT enough for basic_trial()/
    run_fault_phase() -- see make_joint_goals() below, which wraps two
    calls to this and additionally resamples on a colliding COMBINED
    target. An earlier version of this docstring claimed
    plan_to_configuration({"left":.., "right":..}) "independently
    re-checks the COMBINED target for cross-arm collision at plan time",
    which is true but was incomplete: that check has no resample of its
    own, so a colliding combined pair just fails outright (confirmed on
    real hardware -- see make_joint_goals()'s docstring for the failure
    signature). Call make_joint_goals() for basic_trial()/run_fault_phase(),
    not this function directly.
    """
    arm_scope = getattr(robot, arm).arm
    current = arm_scope.get_joint_positions().copy()
    joint_names = list(arm_scope.config.joint_names)
    lower, upper = arm_scope.get_joint_limits()

    saved_qpos = robot.data.qpos.copy()
    goal = current.copy()
    try:
        for attempt in range(max_attempts):
            magnitudes = rng.uniform(0.0, delta, size=current.shape)
            offset = np.array(
                [_signed_delta_for_joint(arm, name, mag) for name, mag in zip(joint_names, magnitudes)]
            )
            candidate = np.clip(current + offset, lower, upper)
            for val, idx in zip(candidate, arm_scope.joint_qpos_indices):
                robot.data.qpos[idx] = val
            mujoco.mj_forward(robot.model, robot.data)
            goal = candidate
            if not robot.arms.check_collisions(arm, verbose=False):
                break
        else:
            print(f"  (make_goal: every draw self-collided for '{arm}' after {max_attempts} attempts -- using the last one)")
        # TODO(review): one summary line for the final candidate, not every
        # retry -- a 20-attempt resample was flooding the console.
        robot.arms.check_collisions(arm)
    finally:
        robot.data.qpos[:] = saved_qpos
        mujoco.mj_forward(robot.model, robot.data)
    return goal


def make_joint_goals(
    robot, rng: np.random.Generator, delta: float, max_attempts: int = 20
) -> tuple[np.ndarray, np.ndarray]:
    """Draw independent left/right goals for a single joint bimanual plan,
    resampling both together if the COMBINED target collides.

    TODO(review): added after a real-hardware finding -- make_goal()'s own
    docstring claimed plan_to_configuration({"left":.., "right":..})
    "independently re-checks the COMBINED target for cross-arm collision at
    plan time", which is true, but incomplete: that check has no resample
    step of its own. When two individually-fine goals collide with each
    other, _plan_frame_sequence() logs "no collision-free combined goal"
    and gives up -- and plan_to_configuration()'s up-to-10 seed retries
    can't fix it, since a seed only varies the RRT search, not the goal
    itself. Confirmed on real hardware (Stage 7 bring-up): all 10 seed
    retries failed identically and near-instantly with that exact message,
    for a goal pair make_goal() had already blessed individually. This is
    the same gap Stage 6 hit and fixed with make_concurrent_goals() --
    ported here almost verbatim.
    """
    left_scope = robot.left.arm
    right_scope = robot.right.arm
    saved_qpos = robot.data.qpos.copy()
    goal_l = goal_r = None
    try:
        for attempt in range(max_attempts):
            goal_l = make_goal(robot, "left", rng, delta)
            goal_r = make_goal(robot, "right", rng, delta)
            for val, idx in zip(goal_l, left_scope.joint_qpos_indices):
                robot.data.qpos[idx] = val
            for val, idx in zip(goal_r, right_scope.joint_qpos_indices):
                robot.data.qpos[idx] = val
            mujoco.mj_forward(robot.model, robot.data)
            if not robot.arms.check_collisions(verbose=False):
                robot.arms.check_collisions()  # one summary line, not every retry
                return goal_l, goal_r
        print(f"  (make_joint_goals: no jointly collision-free pair found after {max_attempts} attempts -- using the last one)")
        robot.arms.check_collisions()
        return goal_l, goal_r
    finally:
        robot.data.qpos[:] = saved_qpos
        mujoco.mj_forward(robot.model, robot.data)


# ---------------------------------------------------------------------------
# Shared: run one joint bimanual plan+execute, record, and analyze.
# ---------------------------------------------------------------------------


def run_joint_execution(robot, goal_l, goal_r, args, label: str, seed: int, sync_recorder: JointStateRecorder):
    """Returns (ok, samples, result) for one plan+execute round trip, or
    (False, [], None) if planning itself failed."""
    # robot.reset() alone does NOT sync from real hardware -- see
    # common/hardware_sync.py. Both arms' trajectories must start from
    # where they actually are.
    sync_shadow_from_hardware(robot, "left", sync_recorder)
    sync_shadow_from_hardware(robot, "right", sync_recorder)
    robot.clear_abort()
    result = robot.plan_to_configuration({"left": goal_l, "right": goal_r}, seed=seed)
    if result is None or not getattr(result, "success", False):
        return False, [], None

    left_joints = joint_names_for(robot, "left")
    right_joints = joint_names_for(robot, "right")
    recorder = JointStateRecorder(_shared_node, joint_filter=left_joints + right_joints)
    with TelemetrySession(args.log_dir, "stage7_bimanual_simultaneous", label, recorder) as tel:
        with robot.real() as ctx:  # noqa: F841
            ok = robot.execute(result)
            # execute() returning only means the goal's nominal duration
            # elapsed, not that either arm has actually converged yet --
            # see stage4_trajectory_execution.py's identical wait.
            if ok and args.settle_time_s > 0:
                time.sleep(args.settle_time_s)
        samples = recorder.samples()
        tel.note("goal_left", np.asarray(goal_l).tolist())
        tel.note("goal_right", np.asarray(goal_r).tolist())
    return ok, samples, result


def return_both_to_zero(robot, sync_recorder: JointStateRecorder) -> None:
    """Return both arms to zero between trials -- consistent starting pose,
    prevents open-loop random goals from drifting either arm over many
    repeats. See stage4/5/6's versions of this for why robot.reset() alone
    doesn't achieve this."""
    sync_shadow_from_hardware(robot, "left", sync_recorder)
    sync_shadow_from_hardware(robot, "right", sync_recorder)
    confirm_phrase("About to return BOTH arms to zero before the next step. Confirm workspace clear.")
    result = robot.plan_to_configuration({"left": np.zeros(7), "right": np.zeros(7)}, seed=0)
    if result is None or not getattr(result, "success", False):
        print("  (couldn't plan a return-to-zero for both arms -- skipping; next step syncs from wherever they are)")
        return
    with robot.real() as ctx:  # noqa: F841
        robot.execute(result)


def analyze_simultaneity(samples, left_joints, right_joints, args) -> StepResult:
    t_left = first_motion_time(samples, left_joints, args.velocity_threshold)
    t_right = first_motion_time(samples, right_joints, args.velocity_threshold)
    if t_left is None or t_right is None:
        return StepResult("simultaneity", False, "one or both arms were never observed moving")
    start_gap = abs(t_left - t_right)
    overlap = simultaneity_overlap_fraction(
        samples, left_joints, samples, right_joints, args.velocity_threshold
    )
    ok = (start_gap <= args.simultaneity_start_tolerance_s) and overlap.ok and (
        overlap.value >= args.simultaneity_min_overlap_fraction
    )
    msg = (
        f"start-time gap = {start_gap:.3f}s (tolerance {args.simultaneity_start_tolerance_s:.2f}s); "
        f"{overlap.message} (required >= {args.simultaneity_min_overlap_fraction:.2f})"
    )
    return StepResult("simultaneity (numeric definition)", ok, msg)


def analyze_collision_free(samples, left_joints, right_joints, args) -> StepResult:
    result = check_pair_collision_free(
        default_openarm_factory(),
        left_joints,
        samples,
        right_joints,
        samples,
        dt=args.collision_check_dt,
    )
    return StepResult("independent collision check on executed telemetry", result.ok, result.summary)


def analyze_endpoints(samples, left_joints, goal_l, right_joints, goal_r, args) -> StepResult:
    left_ok = all(position_reached(samples, n, float(t), args.position_tolerance_rad).ok for n, t in zip(left_joints, goal_l))
    right_ok = all(position_reached(samples, n, float(t), args.position_tolerance_rad).ok for n, t in zip(right_joints, goal_r))
    ok = left_ok and right_ok
    return StepResult("both arms reached their goals", ok, f"left={left_ok} right={right_ok}")


# ---------------------------------------------------------------------------
# Phase: basic
# ---------------------------------------------------------------------------


def basic_trial(robot, args, rng, trial_index: int, sync_recorder: JointStateRecorder) -> TrialOutcome:
    label = f"basic-{trial_index}"
    left_joints = joint_names_for(robot, "left")
    right_joints = joint_names_for(robot, "right")

    # Sync before drawing goals, not just before planning in
    # run_joint_execution() -- make_goal() reads "current position" too,
    # and that needs to be real, not stale.
    sync_shadow_from_hardware(robot, "left", sync_recorder)
    sync_shadow_from_hardware(robot, "right", sync_recorder)
    goal_l, goal_r = make_joint_goals(robot, rng, args.goal_delta_rad)

    confirm_phrase(
        "About to plan and execute a REAL JOINT bimanual trajectory -- both arms moving at once. "
        "Confirm workspace clear around BOTH arms and e-stop hand-ready."
    )

    ok, samples, result = run_joint_execution(robot, goal_l, goal_r, args, label, trial_index, sync_recorder)
    if not ok or not samples:
        return TrialOutcome(False, label, "planning or execute() failed")

    steps = [
        confirm_step(
            analyze_simultaneity(samples, left_joints, right_joints, args),
            question="Did both arms visibly move at the same time, not one after the other?",
        ),
        confirm_step(
            analyze_collision_free(samples, left_joints, right_joints, args),
            question="Did the arms stay clearly clear of each other throughout (no near-miss)?",
        ),
        confirm_step(analyze_endpoints(samples, left_joints, goal_l, right_joints, goal_r, args)),
    ]
    return_both_to_zero(robot, sync_recorder)
    passed = all(s.passed for s in steps)
    notes = "; ".join(f"{s.name}={'ok' if s.passed else 'FAIL'}" for s in steps)
    return TrialOutcome(passed, label, notes)


def run_basic_phase(robot, args, rng, sync_recorder: JointStateRecorder):
    return run_repeated_trials(
        stage="Stage 7 / basic",
        trial_fn=lambda i: basic_trial(robot, args, rng, i, sync_recorder),
        required_consecutive=args.repeats,
    )


# ---------------------------------------------------------------------------
# Phase: adversarial
# ---------------------------------------------------------------------------


def _is_cross_arm_contact(body_a: str, body_b: str) -> bool:
    """True only for a contact strictly between a left-arm body and a
    right-arm body -- excludes self-collision (an arm hitting its own
    links) and single-arm-vs-torso contact (openarm_body_link0 etc.),
    neither of which is the thing this phase is testing for."""
    names = (body_a, body_b)
    return any(n.startswith("openarm_left") for n in names) and any(
        n.startswith("openarm_right") for n in names
    )


def search_adversarial_pair(
    rng, max_attempts: int, n_interp: int = 15, target_radius: float = 0.3, jitter: float = 0.05
):
    """Search, using a fresh offline ArmGroup, for a goal pair whose
    straight-line, independently-interpolated motion collides -- i.e. would
    be at risk under naive (non-jointly-planned) execution. See the module
    docstring's caveat about what "physically" means here.

    TODO(review): targets are now CONSTRUCTED via IK around a shared anchor
    point, not drawn as two independent full-range-random joint vectors and
    then filtered/rejected. Confirmed on real hardware: a forward-kinematics
    proximity REJECTION filter on top of full-range random sampling
    (an earlier version of this fix) still found nothing in 5000 attempts --
    requiring a fully random 7-DOF pose to be simultaneously (a) entirely
    self/torso-collision-free AND (b) within a small Cartesian distance of
    an equally-random OTHER pose is the intersection of two already-rare
    events under blind uniform sampling (this codebase has shown all
    through Stage 4/6/7 bring-up how often even small perturbations from a
    known-good pose self-collide; full-range-from-scratch is far worse).
    Constructing candidates directly instead: pick a random point near the
    midpoint of the arms' CURRENT hand positions (a point known to be in a
    physically plausible shared region, not a hand-picked frame-specific
    guess), then IK-solve each arm to a small random jitter around that
    same point, keeping each arm's own current orientation (keeps the IK
    problem well-posed rather than also randomizing orientation). This
    guarantees Cartesian proximity by construction, so the search only
    needs to additionally confirm collision-freedom and find a genuine
    cross-arm intersection along the path -- both far more likely once
    proximity itself isn't the bottleneck.

    TODO(review): must check for a genuine CROSS-arm contact specifically
    (_is_cross_arm_contact), not just "any collision" -- ag.check_collisions()
    with no filter also returns each arm's OWN self-collisions and
    collisions against the shared torso (openarm_body_link0), which have
    nothing to do with "would the arms hit each other". Confirmed on real
    hardware bring-up: the unfiltered version found a "pair" after 20
    attempts that was purely the right arm colliding with its own base link
    and the torso -- a self-colliding random draw, not a cross-arm
    intersection case -- so the resulting trial failure told us nothing
    about actual cross-arm collision avoidance.

    TODO(review): the ENDPOINTS (q_l1, q_r1) must themselves be jointly
    collision-free, with only the straight-line path BETWEEN current and
    goal allowed to collide -- otherwise this isn't testing "naive motion
    is risky but a joint plan finds a safe path to the same targets", it's
    testing "can you reach an inherently invalid combined configuration",
    which no planner should be able to do. Confirmed on real hardware:
    without this, the search accepted a pair whose collision was first
    detected at alpha=0.93 (near the very end of the interpolation, one of
    the last sampled points before the endpoint itself), and
    plan_to_configuration correctly refused it every time with "no
    collision-free combined goal" -- the endpoint itself was invalid, not
    just the naive path to it, so there was no real task for the planner
    to solve. Now the endpoint is checked first and rejected outright if
    it collides, before the path is scanned for a genuine adversarial
    midpoint.
    """
    ag = default_openarm_factory()()
    left, right = ag["left"], ag["right"]
    q_l0, q_r0 = left.get_joint_positions().copy(), right.get_joint_positions().copy()

    if left.ik_solver is None or right.ik_solver is None:
        raise RuntimeError("search_adversarial_pair needs both arms to have an IK solver configured")

    # Shared anchor: the midpoint of the arms' CURRENT hand positions. This
    # is guaranteed to be in a physically plausible region (derived from the
    # robot's own valid current state), not a hand-picked, frame-specific
    # guess at "somewhere in front of the torso".
    pose_l0 = left.forward_kinematics(q_l0)
    pose_r0 = right.forward_kinematics(q_r0)
    anchor = 0.5 * (pose_l0[:3, 3] + pose_r0[:3, 3])

    for attempt in range(1, max_attempts + 1):
        target_point = anchor + rng.uniform(-target_radius, target_radius, size=3)

        # Each arm aims for the same target point plus a small independent
        # jitter (so the two targets are close but not identical, which
        # numerical IK tends to handle better) -- keep each arm's own
        # current orientation rather than also randomizing it, so the IK
        # problem stays well-posed.
        pose_l1 = pose_l0.copy()
        pose_l1[:3, 3] = target_point + rng.uniform(-jitter, jitter, size=3)
        pose_r1 = pose_r0.copy()
        pose_r1[:3, 3] = target_point + rng.uniform(-jitter, jitter, size=3)

        sols_l = left.ik_solver.solve_valid(pose_l1, q_init=q_l0)
        sols_r = right.ik_solver.solve_valid(pose_r1, q_init=q_r0)
        if not sols_l or not sols_r:
            continue  # IK couldn't reach this jittered target at all -- try another
        q_l1, q_r1 = sols_l[0], sols_r[0]

        left.set_joint_positions(q_l1)
        right.set_joint_positions(q_r1)
        if ag.check_collisions(verbose=False):
            continue  # endpoint itself invalid (any collision, not just cross-arm) -- not a fair test

        for alpha in np.linspace(0.0, 1.0, n_interp):
            q_l = q_l0 + alpha * (q_l1 - q_l0)
            q_r = q_r0 + alpha * (q_r1 - q_r0)
            left.set_joint_positions(q_l)
            right.set_joint_positions(q_r)
            contacts = ag.check_collisions(verbose=False)
            if any(_is_cross_arm_contact(b1, b2) for b1, b2, _ in contacts):
                return q_l1, q_r1, attempt, float(alpha)
    raise RuntimeError(f"no adversarial pair found after {max_attempts} attempts")


def run_adversarial_phase(robot, args, rng, sync_recorder: JointStateRecorder):
    banner("Stage 7 / adversarial: finding or loading a goal pair that collides under naive execution")

    if args.adversarial_goals_file:
        with open(args.adversarial_goals_file) as f:
            data = json.load(f)
        goal_l, goal_r = np.asarray(data["left"]), np.asarray(data["right"])
        print(f"Loaded adversarial goal pair from {args.adversarial_goals_file}.")
    else:
        print(f"Searching for an adversarial pair (up to {args.adversarial_max_search_attempts} attempts)...")
        goal_l, goal_r, attempts, alpha = search_adversarial_pair(rng, args.adversarial_max_search_attempts)
        if goal_l is None:
            print(
                "\nNo adversarial pair found within the search budget. This could mean the arms' "
                "workspaces genuinely can't be made to collide in this model (contype/conaffinity or "
                "an <exclude> may be preventing cross-arm contact detection entirely) -- that itself "
                "is worth checking before concluding this phase can't be run."
            )
            return None
        print(f"Found an adversarial pair after {attempts} attempts (collision at interpolation alpha={alpha:.2f}).")
        save_path = f"{args.log_dir}/stage7_bimanual_simultaneous/adversarial_pair_found.json"
        import os

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump({"left": goal_l.tolist(), "right": goal_r.tolist()}, f, indent=1)
        print(f"Saved for reuse: {save_path}")

    left_joints = joint_names_for(robot, "left")
    right_joints = joint_names_for(robot, "right")

    def trial(i: int) -> TrialOutcome:
        label = f"adversarial-{i}"
        confirm_phrase(
            "About to execute the ADVERSARIAL bimanual goal pair jointly on real hardware. This pair "
            "was specifically chosen because moving independently would collide -- the joint planner "
            "is expected to avoid that. Confirm clear/ready and watch closely."
        )
        ok, samples, result = run_joint_execution(robot, goal_l, goal_r, args, label, i, sync_recorder)
        if not ok or not samples:
            return TrialOutcome(False, label, "planning or execute() failed")

        steps = [
            confirm_step(analyze_simultaneity(samples, left_joints, right_joints, args)),
            confirm_step(
                analyze_collision_free(samples, left_joints, right_joints, args),
                question="Did the arms clearly avoid each other despite this being an adversarial goal pair?",
            ),
            confirm_step(analyze_endpoints(samples, left_joints, goal_l, right_joints, goal_r, args)),
        ]
        return_both_to_zero(robot, sync_recorder)
        passed = all(s.passed for s in steps)
        return TrialOutcome(passed, label, "; ".join(f"{s.name}={'ok' if s.passed else 'FAIL'}" for s in steps))

    return run_repeated_trials(stage="Stage 7 / adversarial", trial_fn=trial, required_consecutive=1, max_attempts=5)


# ---------------------------------------------------------------------------
# Phase: fault
# ---------------------------------------------------------------------------


def run_fault_phase(robot, args, rng, sync_recorder: JointStateRecorder):
    if args.expected_behavior is None:
        print(
            "\nStage 7 / fault SKIPPED -- pass --expected-behavior "
            "{both-stop,left-stops-right-continues,right-stops-left-continues} to declare the "
            "documented design behavior this phase should verify against."
        )
        return None

    left_joints = joint_names_for(robot, "left")
    right_joints = joint_names_for(robot, "right")

    def trial(i: int) -> TrialOutcome:
        label = f"fault-{i}"
        sync_shadow_from_hardware(robot, "left", sync_recorder)
        sync_shadow_from_hardware(robot, "right", sync_recorder)
        goal_l, goal_r = make_joint_goals(robot, rng, args.goal_delta_rad)

        confirm_phrase(
            f"About to execute a joint bimanual trajectory and abort mid-flight. Declared expected "
            f"behavior: '{args.expected_behavior}'. Confirm clear/ready on both arms."
        )

        robot.clear_abort()
        result = robot.plan_to_configuration({"left": goal_l, "right": goal_r}, seed=i)
        if result is None or not result.success:
            return TrialOutcome(False, label, "planning failed")

        recorder = JointStateRecorder(_shared_node, joint_filter=left_joints + right_joints)
        exec_result = {}

        def run_exec():
            with robot.real() as ctx:  # noqa: F841
                exec_result["ok"] = robot.execute(result)

        with TelemetrySession(args.log_dir, "stage7_bimanual_simultaneous", label, recorder) as tel:
            thread = threading.Thread(target=run_exec, daemon=True)
            thread.start()
            time.sleep(1.0)  # crude fixed mid-flight point; good enough to exercise the fault path
            t_abort = time.time()
            robot.request_abort()
            thread.join(timeout=30.0)
            # settle_time_after() needs sustain_s of post-stop data to
            # confirm a settle at all -- execute()/thread.join() returning
            # only means the goal ended, not that either arm has physically
            # finished decelerating (see stage5_abort_and_estop.py's
            # identical wait and the investigation behind it).
            if args.post_abort_settle_s > 0:
                time.sleep(args.post_abort_settle_s)
            samples = recorder.samples()
            tel.note("t_abort", t_abort)
            tel.note("expected_behavior", args.expected_behavior)

        t_left_settled = settle_time_after(samples, left_joints, t_abort, args.velocity_threshold)
        t_right_settled = settle_time_after(samples, right_joints, t_abort, args.velocity_threshold)
        left_reached_goal = all(position_reached(samples, n, float(t), args.position_tolerance_rad).ok for n, t in zip(left_joints, goal_l))
        right_reached_goal = all(position_reached(samples, n, float(t), args.position_tolerance_rad).ok for n, t in zip(right_joints, goal_r))

        left_stopped = t_left_settled is not None
        right_stopped = t_right_settled is not None

        expectation_map = {
            "both-stop": left_stopped and right_stopped and not left_reached_goal and not right_reached_goal,
            "left-stops-right-continues": left_stopped and right_reached_goal,
            "right-stops-left-continues": right_stopped and left_reached_goal,
        }
        matched = expectation_map[args.expected_behavior]
        msg = (
            f"declared='{args.expected_behavior}'; observed: left_stopped={left_stopped} "
            f"left_reached_goal={left_reached_goal} right_stopped={right_stopped} "
            f"right_reached_goal={right_reached_goal}"
        )
        if not matched:
            print(f"\n  MISMATCH: observed behavior does not match the declared expectation.\n  {msg}")

        step = confirm_step(
            StepResult("observed behavior matches declared expectation", matched, msg),
            question="Did what you saw on the robot match the declared expected behavior?",
        )
        robot.clear_abort()  # must clear before return_both_to_zero can command a new motion
        return_both_to_zero(robot, sync_recorder)
        return TrialOutcome(step.passed, label, msg)

    return run_repeated_trials(stage="Stage 7 / fault", trial_fn=trial, required_consecutive=3)


_shared_node = None


def main():
    args = parse_args()
    safety_banner("Stage 7", "Bimanual simultaneous, collision-safe joint trajectory execution.")
    if not confirm(
        "Confirm Stage 6 has already passed, and everyone in the room understands that BOTH arms "
        "will move together, at once, in this stage.",
        default=False,
    ):
        raise SystemExit("Stopping: prerequisites not confirmed.")

    import rclpy
    from rclpy.executors import MultiThreadedExecutor
    import threading as _threading

    global _shared_node
    if not rclpy.ok():
        rclpy.init()
    _shared_node = rclpy.create_node("stage7_telemetry_listener")
    executor = MultiThreadedExecutor()
    executor.add_node(_shared_node)
    spin_thread = _threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    robot = build_robot()
    rng = np.random.default_rng(0)

    all_joint_names = joint_names_for(robot, "left") + joint_names_for(robot, "right")
    sync_recorder = JointStateRecorder(_shared_node, joint_filter=all_joint_names)
    sync_recorder.start()

    # TODO(review): phases now run inside try/finally -- ported from the
    # identical fix in Stage 6/4/5. A declined confirm_phrase deep in any
    # phase used to skip executor.shutdown()/_shared_node.destroy_node()
    # entirely, leaving the daemon spin_thread (running the blocking
    # executor.spin()) orphaned and still touching rclpy's C bindings while
    # the interpreter tore them down during shutdown -- CONFIRMED TO
    # SEGFAULT ON REAL HARDWARE in Stage 6's version of this exact pattern.
    results = {}
    try:
        phases = ["basic", "adversarial", "fault"] if args.phase == "all" else [args.phase]
        for phase in phases:
            if phase == "basic":
                results["basic"] = run_basic_phase(robot, args, rng, sync_recorder)
            elif phase == "adversarial":
                results["adversarial"] = run_adversarial_phase(robot, args, rng, sync_recorder)
            elif phase == "fault":
                results["fault"] = run_fault_phase(robot, args, rng, sync_recorder)
    finally:
        executor.shutdown()
        spin_thread.join(timeout=2.0)
        _shared_node.destroy_node()

    banner("STAGE 7 OVERALL RESULT", char="#")
    all_ok = True
    for name, summary in results.items():
        ok = summary is not None and summary.reached_target
        all_ok = all_ok and ok
        print(f"  {name}: {'PASS' if ok else 'NOT MET / SKIPPED'}")

    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()