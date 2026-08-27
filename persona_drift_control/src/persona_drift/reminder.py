"""Channel A (reminder intensity, u_remind) reminder-text construction.

Only levels 0 and 1 are implemented: that is all the pre-experiment signal
screening (DATA_COLLECTION_PROTOCOL.md section 7, test 2) needs. The
intermediate levels (0.25/0.5/0.75) require pre-authored fixed summaries per
the full protocol (section 3) and are out of scope for the screening pilot.
"""

from __future__ import annotations

REMINDER_LEVELS = (0, 1)


def build_reminder_text(system_prompt: str, level: int) -> str | None:
    if level not in REMINDER_LEVELS:
        raise ValueError(f"level must be one of {REMINDER_LEVELS} for the screening pilot, got {level}")
    if level == 0:
        return None
    return f"[Reminder of your instructions: {system_prompt}]"


def count_inserted_tokens(tokenizer, reminder_text: str | None) -> int:
    if reminder_text is None:
        return 0
    return len(tokenizer.encode(reminder_text, add_special_tokens=False))
