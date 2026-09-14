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


def test_reference_simpler_passes_at_the_revised_threshold():
    """1b\': 34 of 40 is exactly the room F1 0.89 leaves. It passes."""
    got = gt1.criterion_reference_simpler([-0.4] * 34 + [0.2] * 6)
    assert got["passed"] and got["value"] == pytest.approx(0.85)


def test_reference_simpler_catches_an_instrument_worse_than_its_datasheet():
    """Planted defect: 33 of 40 -- a 17.5% inversion rate is not what an F1 0.89
    evaluator does, so this is the instrument being broken, not being finite."""
    got = gt1.criterion_reference_simpler([-0.4] * 33 + [0.2] * 7)
    assert not got["passed"] and got["value"] == pytest.approx(0.825)


def test_reference_simpler_counts_a_zero_delta_as_not_simpler():
    got = gt1.criterion_reference_simpler([-0.4] * 33 + [0.0] * 7)
    assert not got["passed"] and got["n_wrong_direction"] == 7


def test_reference_simpler_catches_a_fully_inverted_readout():
    """Planted defect: the classifier reads every reference as harder."""
    assert not gt1.criterion_reference_simpler([0.4] * 40)["passed"]


def _rows(deltas, fkgl=None, meaning=None, sources=None):
    """Rows for the two row-level checks. Built here, not shared with the
    delta-list fixtures above -- see this module's docstring."""
    n = len(deltas)
    fkgl = fkgl if fkgl is not None else [-1.0] * n
    meaning = meaning if meaning is not None else [0.7 + 0.01 * i for i in range(n)]
    sources = sources if sources is not None else [f"s{i:02d}" for i in range(n)]
    return [{"text_id": f"{sources[i]}-{i}", "source_id": sources[i],
             "source_level_expected": 4.0 + 0.03 * i,
             "reference_level_expected": 4.0 + 0.03 * i + deltas[i],
             "level_delta": deltas[i], "fkgl_delta": fkgl[i],
             "meaning_to_source": meaning[i]} for i in range(n)]


def test_mean_ci_passes_when_the_references_really_are_simpler():
    got = gt1.criterion_reference_simpler_ci(_rows([-0.8] * 38 + [0.3, 0.05]))
    assert got["passed"] and got["ci95"][1] < 0


def test_mean_ci_catches_what_the_direction_share_cannot():
    """Planted defect: 1b\' passes at 0.85 while the batch is not simpler at
    all -- 34 pairs move down by a hair and 6 move up by a lot. The count says
    fine, the mean says the opposite. This is why 1b\'\' exists."""
    rows = _rows([-0.01] * 34 + [2.0] * 6)
    assert gt1.criterion_reference_simpler([r["level_delta"] for r in rows])["passed"]
    got = gt1.criterion_reference_simpler_ci(rows)
    assert not got["passed"] and got["ci95"][1] > 0


def test_mean_ci_catches_a_readout_that_does_not_move():
    got = gt1.criterion_reference_simpler_ci(_rows([0.0] * 40))
    assert not got["passed"] and got["mean"] == pytest.approx(0.0)


def test_mean_ci_resamples_by_source_not_by_row():
    """The 40 pairs are 20 paragraphs at two target levels each. Resampling
    rows would treat two readings of one paragraph as two independent draws."""
    sources = [f"s{i // 2:02d}" for i in range(40)]
    # Both rows of a source carry the same delta: perfectly correlated within a
    # cluster, which is the case row-resampling would mis-state as 40 draws.
    deltas = [-2.0 + 0.1 * (i // 2) for i in range(40)]
    got = gt1.criterion_reference_simpler_ci(_rows(deltas, sources=sources))
    assert got["resample_unit"] == "source_id"
    assert got["n_sources"] == 20 and got["n_pairs"] == 40
    width = got["ci95"][1] - got["ci95"][0]
    width_by_row = got["ci95_by_row"][1] - got["ci95_by_row"][0]
    assert width > width_by_row, "clustering must not narrow the interval"


def test_mean_ci_is_reproducible_from_the_artifact():
    """A signed interval that moves between runs cannot be checked by a reader."""
    rows = _rows([-0.8] * 38 + [0.3, 0.05])
    assert (gt1.criterion_reference_simpler_ci(rows)["ci95"]
            == gt1.criterion_reference_simpler_ci(rows)["ci95"])


def test_data_property_pairs_are_reported_and_still_counted():
    """Planted defect the OTHER way: a pair both readouts call unchanged must
    not quietly leave the denominator. `12-b1` is real; dropping it would be
    choosing the sample by the result."""
    rows = _rows([-0.8] * 39 + [0.03], fkgl=[-2.0] * 39 + [0.05])
    got = gt1.data_property_pairs(rows)
    assert got["n"] == 1 and got["counted_in_denominator"] is True
    assert gt1.criterion_reference_simpler(
        [r["level_delta"] for r in rows])["n_pairs"] == 40


def test_a_genuine_misread_is_not_a_data_property_pair():
    """`03-b1`: level says +0.39, FKGL says -0.92. The readouts disagree, so it
    is the instrument being wrong, not the text being unchanged."""
    rows = _rows([0.388], fkgl=[-0.92])
    assert gt1.data_property_pairs(rows)["n"] == 0


def test_score_is_the_only_place_the_verdict_is_formed():
    """A re-score must not be able to drift from a fresh run, so both go
    through score(). If the criteria were combined twice they could disagree."""
    source = SCRIPT.read_text(encoding="utf-8")
    assert source.count('"PASS" if all(') == 1
    assert "criteria, verdict = score(rows)" in source


def test_score_fails_if_any_single_criterion_fails():
    rows = _rows([-0.8] * 40, meaning=[0.81] * 40)
    criteria, verdict = gt1.score(rows)
    assert verdict == "FAIL"
    assert [c["name"] for c in criteria if not c["passed"]] == ["meaning_sd"]


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
    assert gt1.MIN_SIMPLER_SHARE == 0.85
    assert gt1.MAX_SIMPLER_MEAN_CI_UPPER == 0.0
    assert gt1.MIN_FKGL_AGREEMENT == 0.70
    assert gt1.MIN_MEANING_SD == 0.03
    assert not hasattr(gt1, "REQUIRED_SIMPLER_SHARE"), \
        "the 100% criterion is revised, not kept alongside its replacement"


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
