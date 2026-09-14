"""Tests for scripts/analyze_tsar_cefr_gt1.py.

Every criterion gets a planted defect it must catch, and each is built from its
OWN constructed inputs -- level deltas, FKGL deltas and meaning scores are
never derived from one shared fixture. That is the S3 lesson written into the
plan (section 6.3): both of that script's bugs came from an end-to-end fixture
that wrote the same dicts into two files, so the join it was supposed to test
could not fail.
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "analyze_tsar_cefr_gt1.py"


def load_script():
    spec = importlib.util.spec_from_file_location("gt1", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gt1 = load_script()


def test_distinct_levels_passes_at_the_threshold():
    assert gt1.criterion_distinct_levels([1.0 + 0.1 * i for i in range(10)])["passed"]


def test_distinct_levels_catches_a_readout_with_no_range():
    """Planted defect: the `defense` failure -- one value for every text."""
    got = gt1.criterion_distinct_levels([2.5] * 60)
    assert not got["passed"] and got["value"] == 1


def test_distinct_levels_catches_nine_values():
    assert not gt1.criterion_distinct_levels([1.0 + 0.1 * i for i in range(9)])["passed"]


def test_reference_simpler_requires_every_pair():
    assert gt1.criterion_reference_simpler([-0.4] * 40)["passed"]


def test_reference_simpler_catches_one_wrong_sign():
    """Planted defect: 39 of 40 ordered correctly. 97.5% is not the threshold."""
    got = gt1.criterion_reference_simpler([-0.4] * 39 + [0.2])
    assert not got["passed"] and got["value"] == pytest.approx(39 / 40)


def test_reference_simpler_counts_a_zero_delta_as_not_simpler():
    assert not gt1.criterion_reference_simpler([-0.4] * 39 + [0.0])["passed"]


def test_fkgl_agreement_passes_at_seventy_percent():
    levels = [-1.0] * 7 + [-1.0] * 3
    fkgls = [-2.0] * 7 + [+2.0] * 3
    assert gt1.criterion_fkgl_agreement(levels, fkgls)["passed"]


def test_fkgl_agreement_catches_a_classifier_measuring_something_else():
    """Planted defect: the expected level moves down on every pair while FKGL
    moves up on most of them -- range without the right range."""
    got = gt1.criterion_fkgl_agreement([-1.0] * 10, [+2.0] * 7 + [-2.0] * 3)
    assert not got["passed"] and got["value"] == pytest.approx(0.3)


def test_meaning_range_catches_a_collapsed_distribution():
    """Planted defect: MeaningBERT gives every pair the same score, so the
    admission threshold in the signed primary would admit everything."""
    got = gt1.criterion_meaning_range([0.81, 0.81, 0.8100001, 0.81])
    assert not got["passed"] and got["value"] < gt1.MIN_MEANING_SD


def test_meaning_range_passes_on_a_spread_distribution():
    assert gt1.criterion_meaning_range([0.70, 0.78, 0.85, 0.92])["passed"]


def test_thresholds_are_the_signed_ones():
    """The gate's constants are the plan's, not the command line's; a threshold
    that can be passed as an argument is not a gate."""
    assert gt1.MIN_DISTINCT_EXPECTED_LEVELS == 10
    assert gt1.REQUIRED_SIMPLER_SHARE == 1.0
    assert gt1.MIN_FKGL_AGREEMENT == 0.70
    assert gt1.MIN_MEANING_SD == 0.03


def test_report_carries_the_three_gate_rows():
    """.claude/experiments.md: a gate without all three rows is not allowed to
    open. They go in the artifact, not only in the docstring, so a reader of
    the JSON can see what this verdict was supposed to decide."""
    source = SCRIPT.read_text(encoding="utf-8")
    assert '"on_pass": "submit GPU-1 (open-loop random-excitation arm)"' in source
    assert '"on_fail": "swap the instrument once; a second failure closes tsar_cefr"' in source
    assert '"fills": "appendix A10 block 1"' in source


def test_distinct_levels_catches_the_first_runs_fake_range():
    """Planted defect: the 2026-09-13 run itself -- 60 values, all hugging four
    integers, which the old 1e-6 rounding counted as 56 distinct."""
    levels = [lvl + 0.0001 * i for i, lvl in enumerate([2, 3, 4, 5] * 15)]
    got = gt1.criterion_distinct_levels(levels)
    assert not got["passed"] and got["value"] == 4
    assert got["near_integer_share"] == 1.0
    assert got["effective_integer_levels"] == [2, 3, 4, 5]


def test_distinct_levels_passes_on_values_a_controller_could_separate():
    levels = [2.0 + 0.07 * i for i in range(12)]
    got = gt1.criterion_distinct_levels(levels)
    assert got["passed"] and got["near_integer_share"] < 0.2


def test_resolution_is_the_signed_one():
    assert gt1.LEVEL_RESOLUTION == 0.05
