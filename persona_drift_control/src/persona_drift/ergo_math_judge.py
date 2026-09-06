"""Scores an agent's per-turn "current best answer" attempt against a
GSM8KShardedItem's gold numeric answer, in plain Python -- no LLM call, no
factual judgment. Same reasoning as mc_answer_judge.py's data-source switch:
GSM8K's answer is a single number from an established answer key, so
correctness is a mechanical extraction-and-compare, not something a judge
model should be arbitrating.

Regex-first extraction, anchored on the "Current answer: X" format
ergo_math_bank.py's per-turn prompt asks for (mirrors
mc_answer_judge.extract_letter_by_regex's anchored-pattern-first design);
falls back to a permissive whole-response number search -- adapted from
upstream's own flexible extraction (`tasks/math/task_math.py`'s
`evaluator_function`, arXiv:2505.06120 / microsoft/lost_in_conversation,
MIT license) -- for the rare case a reply skips the requested format.
Unlike mc_answer_judge.py, there is deliberately no third LLM-extraction
fallback here (docs/feasibility/ERGO_MULTITURN_RELIABILITY_FEASIBILITY.md's
scope-reduction for the minimal pilot): a parse failure just gets scored 0.0,
not retried with a model call. Revisit if the parse-failure rate turns out
to matter once real data is in hand.
"""

from __future__ import annotations

import re

from .chat_model import ChatModel, GenerationConfig
from .ergo_math_bank import GSM8KShardedItem

_ANCHORED_PATTERN = re.compile(r"current answer\s*(?:is)?\s*:?\s*\**(-?[$0-9.,]+)\**", re.IGNORECASE)
# Adapted from upstream's evaluator_function: strip currency/thousands separators and a
# trailing ".00", then require the two normalized strings to match exactly.
_FALLBACK_PATTERN = re.compile(r"(-?[$0-9.,]{2,})|(-?[0-9]+)")
_STRIP_REGEXES = (re.compile(r","), re.compile(r"\$"), re.compile(r"\.$"))


def _normalize(number_str: str) -> str:
    value = number_str.strip()
    if value.endswith(".00"):
        value = value[: -len(".00")]
    for pattern in _STRIP_REGEXES:
        value = pattern.sub("", value)
    return value


def extract_answer_by_regex(response: str) -> str | None:
    match = _ANCHORED_PATTERN.search(response)
    if match:
        return match.group(1)
    fallback_matches = _FALLBACK_PATTERN.findall(response)
    if not fallback_matches:
        return None
    last = fallback_matches[-1]
    return next(group for group in last if group)


def judge_math_answer(
    judge: ChatModel,
    entry: GSM8KShardedItem,
    rows: list[dict],
    turn: int,
    stimulus: str,
    response: str,
    seed: int,
    config: GenerationConfig,
) -> tuple[float, bool, str]:
    """Returns (y_task_success, parse_failure, raw_output). `judge`/`seed`/
    `config` are unused (no model call in this domain, see module docstring)
    but kept for JudgeCall's fixed call signature; `rows`/`stimulus` unused
    for the same reason mc_answer_judge.py's are."""

    del judge, rows, stimulus, seed, config  # unused, see docstring

    extracted = extract_answer_by_regex(response)
    if extracted is None:
        return 0.0, True, ""

    score = 1.0 if _normalize(extracted) == _normalize(entry.gold_answer) else 0.0
    return score, False, extracted
