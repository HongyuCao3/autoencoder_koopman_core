"""Rule-layer classifier for whether an ERGO math-trajectory reply is an
"answer attempt" (docs/experiments/ergo_fidelity_restoration_plan.md
section 4.1, step 1).

Upstream's 7-way strategy classifier scores only the "answer attempt"
class and treats a bare `Current answer: X` line -- with no derivation --
as one of the other six (non-scored) classes. This project's harness has
no such classifier; section 0.1's diagnosis is that append-mode's
"reset restores authority" reading was actually measuring whether the
model bothered to write out its derivation at all, because a short,
answer-only reply is trivially easy under `y_task_success`'s per-turn
regex scoring.

Rule, pre-registered and NOT to be tuned after seeing results (section
4.1): strip the `Current answer: X` line(s) from the reply, then count
the remaining non-whitespace characters. Fewer than 80 -> not an answer
attempt. This is the same 80-character threshold and the same "strip the
answer line, count what's left" operation the section-0.1 diagnosis used
to compute "末轮 ≤80 字符占比" (median reply length correlated +0.984 with
success across the six arms).

An LLM classification layer is section 4.1's second tier, enabled only if
gate G-R2-1 (agreement with a 60-row blind human label) fails for this
rule layer. It is deliberately not implemented here.
"""

from __future__ import annotations

import re

# Matches a line that actually DELIVERS the answer: the "Current answer"
# phrase (any case) followed by something number-shaped at end of line.
# Stripped in its entirety, not just the matched substring -- an answer line
# does not carry independent derivation content worth keeping.
#
# Opus ruling 2026-09-08 (was `re.compile(r"current\s+answer", re.I)`, i.e.
# any line merely MENTIONING the phrase): on all existing output directories
# the two forms are indistinguishable -- the loose form strips 84 additional
# lines out of 10467 (0.80%) and changes none of the six arms'
# final_attempt_success or attempt_rate values, verified by independent
# recomputation. The anchored form is preferred going forward because R1's
# upstream-profile replies are expected to be LONGER and more discursive
# (that is what G-R1-1 measures), which makes a prose line that mentions the
# instruction without answering -- "the current answer format wants a single
# number" -- more likely there than in the legacy data. Stripping such a line
# would shrink the surviving character count and could misclassify a genuine
# derivation as a non-attempt, biasing attempt_rate down exactly where R1
# expects it to rise. The pre-registered wording is "the `Current answer: X`
# line", which is this form.
_CURRENT_ANSWER_LINE = re.compile(
    r"current\s+answer\s*(?::|is)?\s*\$?-?[\d.,]+\s*$", re.IGNORECASE
)

# Pre-registered, section 4.1. Do not tune.
_MIN_NON_WHITESPACE_CHARS = 80


def _strip_current_answer_lines(agent_message: str) -> str:
    """Removes every line containing a "Current answer" mention (case-
    insensitive), handling zero, one, or multiple such lines. Lines are
    split on universal newlines so trailing/leading blank lines and
    Windows-style line endings don't affect the character count."""

    lines = agent_message.splitlines()
    kept = [line for line in lines if not _CURRENT_ANSWER_LINE.search(line)]
    return "\n".join(kept)


def classify_attempt(agent_message: str | None) -> bool:
    """True if `agent_message` is an "answer attempt" under the
    pre-registered rule: after stripping any `Current answer: X` line(s),
    at least 80 non-whitespace characters remain.

    An empty or missing (`None`) reply has zero remaining characters and
    is therefore never an attempt.
    """

    if not agent_message:
        return False

    remaining = _strip_current_answer_lines(agent_message)
    non_whitespace_count = len(re.sub(r"\s+", "", remaining))
    return non_whitespace_count >= _MIN_NON_WHITESPACE_CHARS
