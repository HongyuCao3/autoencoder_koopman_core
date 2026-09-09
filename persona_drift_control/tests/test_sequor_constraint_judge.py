"""Tests for persona_drift.sequor_constraint_judge.

The parser is a port of upstream's (deep-spin/SEQUOR
pipeline/best_judge/evaluate_judges.py extract_verdict), so the cases below are
written against upstream's documented behaviour -- last occurrence wins, strict
bracket form preferred, flexible form as fallback -- not against whatever this
port happens to do.

graded_readout's parse-failure contract gets its own tests because the failure
it guards is silent: scoring the surviving constraints when one judgment is
missing would let k vary from turn to turn, and `y` would stop being on a fixed
scale without anything in the output looking wrong.
"""

from __future__ import annotations

import pathlib
import pytest

from persona_drift.sequor_constraint_judge import (
    CONSTRAINT_JUDGE_PROMPT_TEMPLATE,
    binary_turn_success,
    extract_verdict,
    graded_readout,
)


@pytest.mark.parametrize("text,expected", [
    ("Final Verdict: [[Yes]]", True),
    ("Final Verdict: [[No]]", False),
    ("blah blah\n\nFinal Verdict: [[yes]]", True),
    ("Final Verdict:[[No]]", False),
    ("Final Verdict:   [[Yes]]", True),
])
def test_strict_verdicts(text, expected):
    assert extract_verdict(text) is expected


@pytest.mark.parametrize("text,expected", [
    ("Final Verdict: Yes", True),
    ("Final Verdict No", False),
    ("Final Verdict: [Yes]", True),
])
def test_flexible_fallback(text, expected):
    assert extract_verdict(text) is expected


def test_last_occurrence_wins():
    assert extract_verdict("Final Verdict: [[No]]\nOn reflection, Final Verdict: [[Yes]]") is True


def test_strict_form_is_preferred_over_an_earlier_flexible_one():
    # A reasoning block that says "Final Verdict Yes" mid-thought, then commits
    # to [[No]] in the required form: the strict pass runs over the whole text
    # first, so the bracketed verdict wins regardless of position.
    assert extract_verdict("Final Verdict Yes ... actually no.\nFinal Verdict: [[No]]") is False


@pytest.mark.parametrize("text", ["", "I cannot tell.", "The answer is good.", "Verdict: maybe"])
def test_parse_failure_is_none_not_false(text):
    assert extract_verdict(text) is None


def test_prompt_carries_answer_and_constraint_and_no_history():
    p = CONSTRAINT_JUDGE_PROMPT_TEMPLATE.format(answer="AAA", constraint="BBB")
    assert "AAA" in p and "BBB" in p
    assert "Final Verdict" in p
    # upstream's curly quotes, byte-for-byte -- the judge was calibrated on these
    assert "“Final Verdict: [[Yes]]”" in p


@pytest.mark.parametrize("flags,expected", [
    ([True, True, True], 1.0),
    ([True, False, True], pytest.approx(2 / 3)),
    ([False, False, False], 0.0),
])
def test_graded_readout_counts(flags, expected):
    y, n_failed = graded_readout(flags)
    assert y == expected and n_failed == 0


def test_graded_readout_refuses_to_score_a_turn_with_a_parse_failure():
    y, n_failed = graded_readout([True, None, True])
    assert y is None and n_failed == 1


def test_graded_readout_on_empty_is_none():
    assert graded_readout([]) == (None, 0)


@pytest.mark.parametrize("flags,expected", [
    ([True, True, True], True),
    ([True, True, False], False),
    ([True, None, True], None),
])
def test_binary_matches_upstreams_all_aggregation(flags, expected):
    assert binary_turn_success(flags) is expected


def test_prompt_and_parser_import_without_torch():
    """The `constraint` line's vLLM env (plan section 8) deliberately does not
    have this project installed, and its judge runner needs only the prompt
    string and the verdict parser. A top-level `from .chat_model import ...`
    would pull transformers+torch in just to format a string -- and the import
    would fail in that env, not merely be slow. Locked here because a future
    edit reinstating the top-level import would break the vLLM path with an
    error that points at chat_model, not at this file."""

    import subprocess
    import sys

    code = (
        "import sys; "
        "sys.path.insert(0, 'src'); "
        "import persona_drift.sequor_constraint_judge as m; "
        "assert m.extract_verdict('Final Verdict: [[Yes]]') is True; "
        "print('torch' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=str(pathlib.Path(__file__).resolve().parents[1]))
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False", f"torch was imported: {out.stdout!r}"
