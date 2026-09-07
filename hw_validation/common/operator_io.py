"""Operator interaction helpers shared by every stage script.

None of these talk to ROS or hardware. They exist because the single most
important rule across every stage script is: **a human confirms every real
motion before the script moves on to the next check.** These helpers make
that rule easy to follow consistently and hard to accidentally skip.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from typing import Any


def banner(text: str, char: str = "=") -> None:
    line = char * max(60, len(text) + 4)
    print(f"\n{line}\n{text}\n{line}\n")


def safety_banner(stage_name: str, objective: str) -> None:
    banner(f"STAGE: {stage_name}\nOBJECTIVE: {objective}", char="#")


def wait_for_enter(prompt: str = "Press Enter to continue...") -> None:
    input(f"\n{prompt}")


def confirm(prompt: str, default: bool = False) -> bool:
    """Simple y/n prompt. Use for low-stakes confirmations only -- anything
    that gates real motion should use `confirm_phrase` instead, since a
    mis-typed 'y' is too easy to produce by accident.
    """
    suffix = " [Y/n] " if default else " [y/N] "
    while True:
        resp = input(f"{prompt}{suffix}").strip().lower()
        if resp == "":
            return default
        if resp in ("y", "yes"):
            return True
        if resp in ("n", "no"):
            return False
        print("Please answer 'y' or 'n'.")


def confirm_phrase(prompt: str, phrase: str = "READY") -> None:
    """Gate a genuinely dangerous step behind typing an exact phrase, not a
    single keystroke. Raises SystemExit if the operator does not confirm.

    Use this immediately before anything that can move real hardware for
    the first time in a sequence (arm activation, first motion of a new
    kind, resuming after a fault) -- not for routine "next trial?" prompts,
    where `confirm()` is enough.
    """
    print(f"\n{prompt}")
    resp = input(f"Type '{phrase}' to proceed, or anything else to stop: ").strip()
    if resp != phrase:
        print("Not confirmed. Stopping here -- nothing was sent to hardware.")
        raise SystemExit(1)


def stop_session(reason: str) -> None:
    banner(f"SESSION STOPPED: {reason}", char="!")
    raise SystemExit(1)


@dataclass
class StepResult:
    """Outcome of one real-world check within a trial.

    `automated` is what the script measured. `operator_confirmed` is what a
    human watching the arm says. A step only counts as passed if both agree
    -- an automated PASS the operator disputes is treated as a FAIL, since
    the whole point of the human-in-the-loop requirement is that the person
    watching the robot gets the final word.
    """

    name: str
    automated_pass: bool
    automated_message: str = ""
    operator_confirmed: bool | None = None
    data: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        if self.operator_confirmed is None:
            return False  # never leave a step un-confirmed and call it passed
        return self.automated_pass and self.operator_confirmed


def report_step(result: StepResult) -> None:
    verdict = "PASS" if result.automated_pass else "FAIL"
    print(f"\n  [{verdict}] {result.name}")
    if result.automated_message:
        print(f"        {result.automated_message}")


def confirm_step(result: StepResult, question: str | None = None) -> StepResult:
    """Show the automated verdict, then ask the operator to confirm it
    against what they actually observed on the real robot. Always call this
    right after the real-world action the step describes, before moving on
    to anything else.
    """
    report_step(result)
    q = question or f"Did you observe '{result.name}' behave correctly on the robot?"
    result.operator_confirmed = confirm(q, default=False)
    if result.automated_pass and not result.operator_confirmed:
        print("  Recorded as FAIL: automated check passed but operator did not confirm it.")
    elif not result.automated_pass and result.operator_confirmed:
        print(
            "  Recorded as FAIL: operator confirmed, but the automated check disagreed -- "
            "investigate the discrepancy before trusting either signal next time."
        )
    return result


def pause_between_trials(next_trial_label: str) -> bool:
    """Ask whether to continue after a trial completes. Returns False if the
    operator wants to stop the session."""
    print()
    resp = input(
        f"Reset the arm/workspace as needed for '{next_trial_label}'. "
        f"Press Enter when ready, or type 'stop' to end the session: "
    ).strip().lower()
    return resp != "stop"


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")