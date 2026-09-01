import pytest

from persona_drift.safety_reminder import build_safety_reminder_text


def test_level_zero_is_no_insertion():
    assert build_safety_reminder_text(0) is None


def test_level_one_returns_a_fixed_safety_reminder():
    text = build_safety_reminder_text(1)
    assert text is not None
    assert "harm" in text.lower()


def test_reminder_text_does_not_reference_the_attack_or_conversation():
    text = build_safety_reminder_text(1)
    # The reminder must stay a fixed, content-agnostic sentence -- it must
    # not template in anything attack-specific (there's no per-attack
    # variable to template in, but this guards against a future edit
    # accidentally parameterizing it).
    assert build_safety_reminder_text(1) == text


def test_rejects_unimplemented_levels():
    with pytest.raises(ValueError):
        build_safety_reminder_text(0.5)
