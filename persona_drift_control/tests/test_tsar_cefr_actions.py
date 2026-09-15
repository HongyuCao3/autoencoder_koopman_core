"""Tests for src/persona_drift/tsar_cefr_actions.py.

These exist because of ONE change: on 2026-09-14 the level-naming template was
widened from the two task targets to all six CEFR levels, so that `dp_path`
(plan 5.4) could exist as a distinct arm at all -- with only adjacent
transitions the strictly-decreasing path is unique and `dp_path` collapses into
`fixed_ladder`.

Widening a template that a COMPLETED gate already ran through is exactly the
move .claude/code.md asks for a byte-identical regression test on: D-0's
saturation probe (one-shot hit rate 0.350, reported in tsar_cefr_results.md)
was produced by `saturation_message`, and that string must not have moved
because a baseline needed a wider prompt. The expected strings below are
frozen literals, not re-derived from the module.
"""

from __future__ import annotations

import pytest

from persona_drift import tsar_cefr_actions as actions

FROZEN_A2 = (
    "Rewrite the text so that a reader at CEFR level A2 can read it, "
    "and so that it is not simpler than CEFR level A2.\n\n"
    "Text:\nHELLO.\n\n"
    "Rewritten text:"
)
FROZEN_B1 = (
    "Rewrite the text so that a reader at CEFR level B1 can read it, "
    "and so that it is not simpler than CEFR level B1.\n\n"
    "Text:\nHELLO.\n\n"
    "Rewritten text:"
)


def test_saturation_message_is_byte_identical_to_what_d0_ran():
    assert actions.saturation_message("HELLO.", "A2") == FROZEN_A2
    assert actions.saturation_message("HELLO.", "B1") == FROZEN_B1


def test_saturation_message_keeps_its_own_two_level_contract():
    """The probe's contract must not widen just because the template did."""
    for level in ("A1", "B2", "C1", "C2"):
        with pytest.raises(ValueError, match="target_cefr"):
            actions.saturation_message("HELLO.", level)


def test_level_message_accepts_all_six_and_agrees_on_the_overlap():
    for level in actions.CEFR_LEVELS:
        assert f"CEFR level {level}" in actions.level_message("HELLO.", level)
    assert actions.level_message("HELLO.", "A2") == FROZEN_A2
    assert actions.level_message("HELLO.", "B1") == FROZEN_B1


def test_level_message_rejects_a_non_level():
    with pytest.raises(ValueError, match="CEFR level"):
        actions.level_message("HELLO.", "B1.5")


def test_the_four_control_actions_are_unchanged():
    """dp_path is an opponent, not a control policy: the action set must not grow."""
    assert actions.ACTION_NAMES == ("step_down", "half_step_down", "paraphrase", "copy")
    assert actions.ZERO_INPUT_ACTION == "copy"


def test_step_template_still_refuses_a_non_target_level():
    with pytest.raises(ValueError, match="target_cefr"):
        actions.user_message("HELLO.", "step_down", "B2")
