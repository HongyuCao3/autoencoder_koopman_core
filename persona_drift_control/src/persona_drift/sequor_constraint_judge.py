"""Per-constraint judging for the `constraint` line (SEQUOR multi-turn
constraint retention, docs/experiments/constraint_retention_plan.md).

The prompt and the verdict parser are VENDORED VERBATIM from upstream
(deep-spin/SEQUOR @60bcdaca: `multi_if/evaluation.py` SYSTEM_PROMPT_CONSTRAINT,
`pipeline/best_judge/evaluate_judges.py` extract_verdict). This is deliberate
and is the whole reason this task was chosen: the per-constraint decision is an
instrument upstream already built and calibrated, and **the only thing this
project changes is the aggregation** -- upstream collapses the k decisions with
`all()` into a binary turn-success (`multi_if/evaluation.py:539`), while the
`constraint` line keeps the count, so the readout takes k+1 values instead of 2.
Changing the prompt would forfeit that, and would make our numbers
non-comparable with upstream's own judge calibration.

The curly quotes in the prompt are upstream's; do not "fix" them -- the judge
was calibrated with those bytes.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # the prompt and the parser must be importable WITHOUT torch:
    # the `constraint` line's vLLM environment (plan section 8) deliberately
    # does not have this project installed, and a top-level ChatModel import
    # would drag transformers+torch into it just to format a string.
    from .chat_model import ChatModel, GenerationConfig

# multi_if/evaluation.py SYSTEM_PROMPT_CONSTRAINT, verbatim.
CONSTRAINT_JUDGE_PROMPT_TEMPLATE = (
    "An assistant has been asked to perform a task. Your job is to assess whether the provided "
    "answer satisfies a given constraint. You may first reason about both the constraint and the "
    "answer. At the end, present your final verdict as either “Final Verdict: [[Yes]]” if "
    "the answer satisfies the constraint, or “Final Verdict: [[No]]” if it does not.\n"
    "\n"
    "Does the following answer satisfy the constraint?\n"
    "\n"
    "Answer: \n"
    '"{answer}"\n'
    "\n"
    "Constraint:\n"
    '"{constraint}"\n'
)

_STRICT = re.compile(r"Final Verdict:\s*\[\[(Yes|No)\]\]", re.IGNORECASE)
_FLEXIBLE = re.compile(r"Final Verdict:?\s*\[*\[*(Yes|No)\]*\]*", re.IGNORECASE)


def extract_verdict(judge_text: str) -> bool | None:
    """Port of upstream's extract_verdict: last strict match wins, else last
    flexible match, else None (a parse failure, never silently a False --
    "the judge did not answer" and "the judge said No" are different events
    and the calibration reports them separately)."""

    for pattern in (_STRICT, _FLEXIBLE):
        matches = pattern.findall(judge_text)
        if matches:
            return matches[-1].capitalize() == "Yes"
    return None


def judge_constraint_followed(
    judge: "ChatModel",
    constraint_text: str,
    answer_text: str,
    seed: int,
    config: "GenerationConfig | None" = None,
    enable_thinking: bool = False,
) -> tuple[bool | None, str]:
    """Returns (followed, judge_raw_output). `followed` is None on a parse
    failure, matching upstream's `ConstraintEvaluation.followed: bool | None`.

    Note what is NOT passed in: the conversation history. Upstream's judge sees
    only this turn's answer and one constraint, which is what makes the k
    decisions independent of each other and of turn index -- the property the
    graded readout needs."""

    from .chat_model import GenerationConfig

    config = config or GenerationConfig(max_new_tokens=512, temperature=0.0, do_sample=False)
    prompt = CONSTRAINT_JUDGE_PROMPT_TEMPLATE.format(answer=answer_text, constraint=constraint_text)
    output = judge.generate(
        [{"role": "user", "content": prompt}], seed=seed, config=config, enable_thinking=enable_thinking
    )
    return extract_verdict(output), output


def graded_readout(followed_flags: list[bool | None]) -> tuple[float | None, int]:
    """`y_t` = (satisfied active constraints) / k, and the number of parse
    failures that went into it.

    Upstream's own aggregation is `all(judged_flags)` over the non-None flags
    (multi_if/evaluation.py:539), which is this readout thresholded at 1.0.
    Both are reported by the `constraint` line: the graded one is the state the
    controller feeds back on, the binary one is what compares to upstream.

    A turn with ANY parse failure returns None rather than scoring the survivors
    -- otherwise k would silently vary across turns and `y` would not be on a
    fixed scale, which is precisely the kind of quiet readout drift that cost
    this project two lines."""

    n_failed = sum(1 for f in followed_flags if f is None)
    if n_failed or not followed_flags:
        return None, n_failed
    return sum(1 for f in followed_flags if f) / len(followed_flags), 0


def binary_turn_success(followed_flags: list[bool | None]) -> bool | None:
    """Upstream-comparable readout: all k constraints satisfied."""
    y, n_failed = graded_readout(followed_flags)
    return None if y is None else y == 1.0
