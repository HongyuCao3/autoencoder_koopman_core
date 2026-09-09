"""Tests for scripts/score_sequor_trajectories.py's CPU half.

`refuse_self_judge_mislabel` is the load-bearing one. A self-judged score
labelled `independent` is the failure that hid a 4x effect size on the
`defense` line, and once it is written into an artifact nothing downstream can
tell -- the label is the only record of which model produced the verdicts.
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

_PATH = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "score_sequor_trajectories.py"
_SPEC = importlib.util.spec_from_file_location("score_sequor_trajectories", _PATH)
mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mod)

AGENT = "Qwen/Qwen3-4B-Instruct-2507"
REPORTER = "Qwen/Qwen3-14B"
ARM = {"model_id": AGENT, "mode": "canonical"}


def test_the_agents_own_model_cannot_be_labelled_independent():
    with pytest.raises(SystemExit, match="self-judged score"):
        mod.refuse_self_judge_mislabel(ARM, AGENT, "independent")


def test_the_agents_own_model_labelled_self_is_fine():
    mod.refuse_self_judge_mislabel(ARM, AGENT, "self")


def test_a_different_model_labelled_independent_is_fine():
    mod.refuse_self_judge_mislabel(ARM, REPORTER, "independent")


def test_an_arm_config_without_a_model_id_does_not_silently_pass_as_independent():
    """A malformed arm config must not become a free pass: with no model_id
    there is nothing to compare, so the check cannot certify independence and
    the caller's label stands -- which is why the arm config is written by the
    runner and not by hand. Documented here so the hole is deliberate."""

    mod.refuse_self_judge_mislabel({}, AGENT, "independent")


def _row(branch: str, turn: int, y: float | None) -> dict:
    return {"branch": branch, "turn": turn, "y_graded": y}


def test_range_by_turn_reports_spread_per_branch_and_turn():
    scored = [_row("base", 1, 1.0), _row("base", 1, 2 / 3), _row("base", 1, 1 / 3),
              _row("reminded", 2, 1.0), _row("reminded", 2, 1.0)]
    ranges = mod.readout_range_by_turn(scored)
    assert ranges["base_t1"]["n"] == 3 and ranges["base_t1"]["distinct"] == 3
    assert ranges["base_t1"]["sd"] > 0
    assert ranges["reminded_t2"]["sd"] == 0.0 and ranges["reminded_t2"]["distinct"] == 1


def test_range_by_turn_excludes_undefined_rows_rather_than_scoring_them_zero():
    """A turn with an unparsed constraint has NO readout. Counting it as 0
    would invent drift that the judge merely failed to measure."""

    scored = [_row("base", 1, 1.0), _row("base", 1, None), _row("base", 1, 2 / 3)]
    ranges = mod.readout_range_by_turn(scored)
    assert ranges["base_t1"]["n"] == 2
    assert ranges["base_t1"]["mean"] == pytest.approx((1.0 + 2 / 3) / 2)


def test_range_by_turn_is_empty_when_nothing_parsed():
    assert mod.readout_range_by_turn([_row("base", 1, None)]) == {}
