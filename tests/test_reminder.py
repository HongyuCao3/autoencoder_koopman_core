import pytest

from persona_drift.reminder import build_reminder_text


def test_level_zero_is_no_insertion():
    assert build_reminder_text("Always speak like a pirate.", 0) is None


def test_level_one_reinserts_the_full_system_prompt():
    text = build_reminder_text("Always speak like a pirate.", 1)
    assert text is not None
    assert "Always speak like a pirate." in text


def test_rejects_unimplemented_levels():
    with pytest.raises(ValueError):
        build_reminder_text("x", 0.5)
