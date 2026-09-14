"""Tests for src/persona_drift/tsar_cefr_readout.py.

The invariant worth a test: both layers come from ONE classifier -- the one the
shared task's rule selected. Averaging the three, or letting the argmax and the
expectation disagree about which model they came from, would make the control
readout answer a question the reported number never asks
(.claude/global.md -> selection vs reporting).
"""

from __future__ import annotations

import pytest

from persona_drift import tsar_cefr_readout as readout

CONFIDENT_A2 = [0.02, 0.90, 0.04, 0.02, 0.01, 0.01]
FLAT_B1 = [0.15, 0.15, 0.30, 0.20, 0.10, 0.10]
CONFIDENT_C1 = [0.01, 0.01, 0.02, 0.04, 0.91, 0.01]


def test_official_pick_takes_the_highest_top1_confidence():
    assert readout.official_pick([FLAT_B1, CONFIDENT_A2, FLAT_B1]) == 1


def test_expected_level_is_the_probability_weighted_level():
    assert readout.expected_level([1.0, 0, 0, 0, 0, 0]) == pytest.approx(1.0)
    assert readout.expected_level([0, 0, 0, 0, 0, 1.0]) == pytest.approx(6.0)
    assert readout.expected_level([0.5, 0, 0, 0, 0, 0.5]) == pytest.approx(3.5)


def test_both_layers_come_from_the_selected_classifier():
    """Planted disagreement: the flat model would say B1 and 3.25, the selected
    one says A2 and ~2.1. Reading the argmax from one and the expectation from
    another -- or averaging -- fails this."""
    got = readout.read_level([FLAT_B1, CONFIDENT_A2, CONFIDENT_C1])
    assert got.level_official == "C1"
    assert got.level_expected == pytest.approx(readout.expected_level(CONFIDENT_C1))
    assert got.official_model == readout.CEFR_MODELS[2]


def test_expected_level_has_range_where_the_argmax_has_none():
    """Why the control readout is the expectation: three texts the official
    label cannot separate, separated."""
    nudged = [[0.02, 0.90 - d, 0.04 + d, 0.02, 0.01, 0.01] for d in (0.0, 0.05, 0.10)]
    labels = {readout.read_level([p]).level_official for p in nudged}
    values = {round(readout.read_level([p]).level_expected, 6) for p in nudged}
    assert labels == {"A2"} and len(values) == 3


def test_wrong_length_probability_vector_is_rejected():
    with pytest.raises(ValueError, match="probabilities"):
        readout.expected_level([0.5, 0.5])
    with pytest.raises(ValueError, match="CEFR probabilities"):
        readout.official_pick([[0.5, 0.5]])


def test_fkgl_moves_down_when_text_is_simplified():
    hard = ("The committee's deliberations, notwithstanding considerable "
            "procedural complexity, culminated in unanimous ratification.")
    easy = "The group talked for a long time. In the end they all said yes."
    assert readout.fkgl(easy) < readout.fkgl(hard)


def test_syllable_heuristic_drops_the_silent_final_e():
    assert readout.count_syllables("make") == 1
    assert readout.count_syllables("little") == 2
    assert readout.count_syllables("a") == 1


def test_fkgl_refuses_a_text_with_no_words():
    with pytest.raises(ValueError, match="no words"):
        readout.fkgl("... !!!")


def test_temperature_one_is_the_identity():
    assert readout.temperature_scale(FLAT_B1, 1.0) == FLAT_B1


def test_temperature_above_one_softens_and_below_one_sharpens():
    soft = readout.temperature_scale(CONFIDENT_A2, 4.0)
    sharp = readout.temperature_scale(CONFIDENT_A2, 0.5)
    assert max(soft) < max(CONFIDENT_A2) < max(sharp)
    assert sum(soft) == pytest.approx(1.0) and sum(sharp) == pytest.approx(1.0)


def test_softening_moves_the_expected_level_off_the_integer():
    """Why the swap was made: at T=1 a near-one-hot vector reads as its argmax."""
    near_one_hot = [0.002, 0.990, 0.004, 0.002, 0.001, 0.001]
    at_one = readout.expected_level(readout.temperature_scale(near_one_hot, 1.0))
    at_four = readout.expected_level(readout.temperature_scale(near_one_hot, 4.0))
    assert abs(at_one - round(at_one)) < 0.02
    assert abs(at_four - round(at_four)) > 10 * abs(at_one - round(at_one))


def test_temperature_never_touches_the_report_layer():
    """`level_official` is the shared task's evaluator; softening is a control-side
    choice and must not move a reported label."""
    for temperature in (0.5, 1.0, 4.0):
        assert readout.read_level([CONFIDENT_A2], temperature).level_official == "A2"


def test_fit_temperature_sharpens_when_the_classifier_is_already_right():
    """NLL calibrates accuracy, not range. Planted: confident and correct -> T < 1.
    This is why the fitted T is reported rather than assumed to buy spread."""
    assert readout.fit_temperature([CONFIDENT_A2] * 5, ["A2"] * 5) < 1.0


def test_fit_temperature_softens_when_the_classifier_is_confidently_wrong():
    assert readout.fit_temperature([CONFIDENT_A2] * 5, ["C1"] * 5) > 1.0


def test_fit_temperature_requires_one_gold_level_per_text():
    with pytest.raises(ValueError, match="one gold level per text"):
        readout.fit_temperature([CONFIDENT_A2, FLAT_B1], ["A2"])


def test_non_positive_temperature_is_rejected():
    with pytest.raises(ValueError, match="temperature must be positive"):
        readout.temperature_scale(FLAT_B1, 0.0)
