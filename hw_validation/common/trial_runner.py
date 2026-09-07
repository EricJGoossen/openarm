"""The repeatability engine used by every stage script.

The spec's repeatability invariant (a capability isn't "working" after one
success) and the operator's explicit requirement (confirm before moving on
to the next real-world check) are the same mechanism here: after every
trial, we stop, show the result, and require an explicit human decision
before continuing -- and a single failed or disputed trial resets the
consecutive-pass counter, it does not just get skipped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .operator_io import banner, confirm, pause_between_trials


@dataclass
class TrialOutcome:
    passed: bool
    label: str
    notes: str = ""
    data: dict = field(default_factory=dict)


@dataclass
class TrialSessionSummary:
    stage: str
    required_consecutive: int
    outcomes: list[TrialOutcome] = field(default_factory=list)
    reached_target: bool = False

    def print_summary(self) -> None:
        banner(f"STAGE '{self.stage}' SESSION SUMMARY", char="-")
        for i, o in enumerate(self.outcomes, 1):
            verdict = "PASS" if o.passed else "FAIL"
            print(f"  trial {i:>3} [{verdict}]  {o.label}   {o.notes}")
        n_pass = sum(1 for o in self.outcomes if o.passed)
        print(f"\n  {n_pass}/{len(self.outcomes)} trials passed overall.")
        if self.reached_target:
            print(f"  Reached the required {self.required_consecutive} consecutive passes. Stage exit criteria MET.")
        else:
            print(
                f"  Did NOT reach {self.required_consecutive} consecutive passes. "
                f"Stage exit criteria NOT met -- do not advance to the next stage."
            )


def run_repeated_trials(
    stage: str,
    trial_fn: Callable[[int], TrialOutcome],
    required_consecutive: int,
    max_attempts: int | None = None,
) -> TrialSessionSummary:
    """Call `trial_fn(trial_index)` repeatedly. `trial_fn` is responsible for
    all real-world confirmation *within* a trial (see the stage scripts --
    each individual check inside a trial should already be using
    `operator_io.confirm_step`); this function's job is the *between-trial*
    gate and the consecutive-pass bookkeeping.

    Stops early if the operator types 'stop' at the between-trial prompt, or
    if `max_attempts` is reached without hitting the target.
    """
    summary = TrialSessionSummary(stage=stage, required_consecutive=required_consecutive)
    consecutive = 0
    attempt = 0

    while True:
        attempt += 1
        if max_attempts is not None and attempt > max_attempts:
            print(f"\nReached max_attempts={max_attempts} without hitting the target. Stopping.")
            break

        banner(f"{stage}: trial {attempt} (consecutive passes so far: {consecutive}/{required_consecutive})")
        outcome = trial_fn(attempt)
        summary.outcomes.append(outcome)

        if outcome.passed:
            consecutive += 1
            print(f"\nTrial {attempt} PASSED. Consecutive passes: {consecutive}/{required_consecutive}.")
        else:
            consecutive = 0
            print(f"\nTrial {attempt} FAILED ({outcome.notes}). Consecutive pass count reset to 0.")
            if not confirm("A trial just failed. Continue attempting this stage?", default=False):
                break

        if consecutive >= required_consecutive:
            summary.reached_target = True
            break

        if not pause_between_trials(f"{stage} trial {attempt + 1}"):
            break

    summary.print_summary()
    return summary