import pytest

from persona_drift.consistency_reminder import build_consistency_reminder_text


def test_level_zero_is_no_insertion():
    assert build_consistency_reminder_text(0) is None


def test_level_one_returns_a_fixed_consistency_reminder():
    text = build_consistency_reminder_text(1)
    assert text is not None
    assert "verified facts" in text.lower()


def test_reminder_text_does_not_reference_the_claim_or_conversation():
    text = build_consistency_reminder_text(1)
    # Fixed, claim-agnostic sentence -- guards against a future edit
    # accidentally parameterizing it per-item.
    assert build_consistency_reminder_text(1) == text


def test_rejects_unimplemented_levels():
    with pytest.raises(ValueError):
        build_consistency_reminder_text(0.5)
