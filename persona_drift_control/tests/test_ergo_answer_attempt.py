"""docs/experiments/ergo_fidelity_restoration_plan.md section 4.1: the rule
layer classifier is pre-registered as "strip the `Current answer: X` line,
then check whether at least 80 non-whitespace characters remain". Tested
here directly (`classify_attempt`) and through
`analyze_ergo_dual_metric.per_item_final_attempt_success`'s 规格补丁 D1
reading (a) fallback-to-earlier-attempt behavior on synthetic two-arm data.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

from persona_drift.ergo_answer_attempt import classify_attempt  # noqa: E402

from analyze_ergo_dual_metric import per_item_final_attempt_success  # noqa: E402


def test_long_derivation_ending_in_current_answer_is_an_attempt():
    derivation = "x" * 90  # 90 non-whitespace chars of derivation alone
    message = f"{derivation}\nCurrent answer: 42"
    assert classify_attempt(message) is True


def test_bare_current_answer_is_not_an_attempt():
    assert classify_attempt("Current answer: 42") is False


def test_boundary_79_chars_after_stripping_is_not_an_attempt():
    derivation = "a" * 79
    message = f"{derivation}\nCurrent answer: 42"
    assert classify_attempt(message) is False


def test_boundary_80_chars_after_stripping_is_an_attempt():
    derivation = "a" * 80
    message = f"{derivation}\nCurrent answer: 42"
    assert classify_attempt(message) is True


def test_no_current_answer_line_counts_whole_message():
    # No "Current answer" line at all -- the full message is what gets
    # counted, and here it's short enough not to be an attempt.
    assert classify_attempt("I'm not sure yet.") is False
    long_message = "b" * 80
    assert classify_attempt(long_message) is True


def test_empty_and_missing_reply_is_not_an_attempt():
    assert classify_attempt("") is False
    assert classify_attempt(None) is False


def test_multiple_current_answer_lines_and_case_variations_are_all_stripped():
    derivation = "z" * 90  # 90 non-whitespace chars
    message = f"CURRENT ANSWER: 10\n{derivation}\nCurrent Answer is: 42\ncurrent answer: 42"
    assert classify_attempt(message) is True
    # Same derivation length, but with the derivation itself removed (only
    # answer lines remain) -- must fall back to not-an-attempt.
    assert classify_attempt("CURRENT ANSWER: 10\nCurrent Answer is: 42\ncurrent answer: 42") is False


def _row(item_id, trajectory_id, turn, num_shards, y, agent_message, seed=0):
    return {
        "item_id": item_id,
        "trajectory_id": trajectory_id,
        "turn": turn,
        "num_shards": num_shards,
        "y_task_success": y,
        "agent_message": agent_message,
        "seed": seed,
    }


def test_final_attempt_success_falls_back_to_earlier_attempt_and_zero_when_none():
    long_derivation = "d" * 90  # 90 non-whitespace chars, is an attempt

    # Trajectory 1: turn 1 is a real attempt (scored 1.0), turn 2 (final)
    # regresses to a bare answer-only reply -- must fall back to turn 1's
    # score, not score 0.0 and not use the legacy final-turn score.
    traj1_rows = [
        _row("item1", "item1__seed0", 1, 2, 1.0, f"{long_derivation}\nCurrent answer: 7"),
        _row("item1", "item1__seed0", 2, 2, 0.0, "Current answer: 7"),
    ]

    # Trajectory 2: no turn is ever an attempt -- must score 0.0 regardless
    # of the legacy y_task_success values.
    traj2_rows = [
        _row("item2", "item2__seed0", 1, 1, 1.0, "Current answer: 3"),
    ]

    rows = traj1_rows + traj2_rows
    per_item = per_item_final_attempt_success(rows)
    assert per_item["item1"] == 1.0
    assert per_item["item2"] == 0.0


def test_prose_line_mentioning_the_instruction_is_not_stripped_as_an_answer_line():
    # Opus ruling 2026-09-08: only a line that actually delivers a number is
    # the answer line. A discursive line that merely names the instruction
    # must survive the strip, or its characters stop counting toward the
    # 80-char attempt threshold and a genuine derivation gets misclassified.
    # R1's upstream-profile replies are expected to be longer and more
    # discursive, which is where this bites.
    prose = (
        "I should note that the current answer format asks for a single number, "
        "so I will give the total rather than the per-year figure below.\n"
        "Current answer: 42"
    )
    assert classify_attempt(prose) is True
    # And the delivering line itself still goes, so a bare one is not an attempt.
    assert classify_attempt("Current answer: 42") is False
