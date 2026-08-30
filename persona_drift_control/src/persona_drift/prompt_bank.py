"""Loads the vendored 100-system-prompt benchmark and selects the two
categories the pre-experiment signal screening protocol calls "character
traits" and "language constraints".

Naming note: the upstream dataset's own taxonomy is
`pattern_system_prompts` / `multiple_choice_system_prompts` /
`persona_system_prompts` / `memorization_system_prompts` /
`language_system_prompts` (the last one meaning "speak French", only 3
prompts). DATA_COLLECTION_PROTOCOL.md's "character traits" and "language
constraints" labels do not match that taxonomy 1:1; the closest fit by the
protocol's own selection criterion ("probe scores closer to continuous") is
`persona_system_prompts` (character traits/tone) and `pattern_system_prompts`
(grammatical/formatting constraints - word/letter/case/sentence patterns),
not the 3-prompt French category. That mapping is what CATEGORY_LABELS below
encodes.
"""

from __future__ import annotations

import importlib.util
import pathlib
import random
from dataclasses import dataclass
from typing import Callable

RESOURCES_DIR = pathlib.Path(__file__).resolve().parents[2] / "resources"

CATEGORY_LABELS = {
    "persona_system_prompts": "character_traits",
    "pattern_system_prompts": "language_constraints",
}

# Confirmed via the 2026-08-29/30 signal-screening pilot (see
# docs/experiments/signal_screening_pilot.md): these two entries' score_fn
# returned exactly 1.0 for every one of 64 rows each (16 turns x 2 seeds x 2
# conditions), y_probe_sd == 0 throughout, regardless of turn or u_remind --
# Qwen3-4B satisfies "thank me" / "one sentence per reply" trivially every
# time, so the probe carries zero information about drift or control effects.
# Excluded from the pool (rather than fixed in place) because
# hundred_system_prompts.py is vendored, not ours to edit.
KNOWN_SATURATED_PROMPT_IDS = {"character_traits_013", "language_constraints_004"}

# Zero-GPU pre-filter: a diverse battery of canned "typical assistant reply"
# strings, used to flag score_fn's that are structurally unsafe for this
# protocol *before* spending GPU time on them, rather than discovering it
# after the fact via analysis.saturated_prompt_ids (see
# docs/experiments/signal_screening_pilot.md 2026-08-30 investigation).
# Two independent risks this catches:
#   - "binary_across_battery": score_fn is a literal boolean predicate cast
#     to float (both KNOWN_SATURATED_PROMPT_IDS entries were this -- `"thank"
#     in x.lower()` and `len(sentences) == 1`). Not every boolean scorer
#     saturates in practice (a genuinely hard instruction can still show real
#     turn-to-turn variance), but it's the exact failure class already
#     observed, so it's worth deprioritizing when a continuous alternative
#     exists.
#   - "out_of_unit_range": score_fn returns something other than a bounded
#     [0, 1] fraction/probability (e.g. a raw word count) -- pooling that
#     with [0, 1]-scaled scores from other prompts in the same Q1/Q2/Q3
#     average would silently mix incompatible scales.
_SATURATION_PROBE_BATTERY = (
    "",
    "Sure, happy to help with that.",
    "I cannot help with that request.",
    "THIS IS ALL CAPS AND VERY LOUD ABOUT THINGS.",
    "banana banana banana repeat repeat repeat words words.",
    "One. Two three four five six seven eight nine ten eleven twelve.",
    "Thank you so much, I really appreciate it! This is wonderful news today.",
    "no. stop. leave. now. bad. sad. angry. mad. hate. ugly.",
)


def classify_scorer_screening_safety(
    score_fn: Callable[[str], float], battery: tuple[str, ...] = _SATURATION_PROBE_BATTERY
) -> tuple[bool, str]:
    """Returns (is_safe, reason). `reason` is one of "ok",
    "all_battery_evals_failed" (inconclusive, not treated as unsafe),
    "out_of_unit_range", or "binary_across_battery"."""

    scores: list[float] = []
    for text in battery:
        try:
            scores.append(float(score_fn(text)))
        except Exception:
            continue
    if not scores:
        return False, "all_battery_evals_failed"
    if any(s < 0.0 or s > 1.0 for s in scores):
        return False, "out_of_unit_range"
    if all(s in (0.0, 1.0) for s in scores):
        return False, "binary_across_battery"
    return True, "ok"


@dataclass(frozen=True)
class PromptEntry:
    prompt_id: str
    prompt_category: str  # "character_traits" | "language_constraints"
    system_prompt: str
    probe_question: str
    score_fn: Callable[[str], float]


def _load_module(resources_dir: pathlib.Path):
    module_path = resources_dir / "hundred_system_prompts.py"
    spec = importlib.util.spec_from_file_location("hundred_system_prompts", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_prompt_bank(resources_dir: pathlib.Path | None = None) -> dict[str, list[PromptEntry]]:
    """Returns {"character_traits": [...], "language_constraints": [...]},
    with "random" probes resolved to a fixed question per entry (seeded on
    the entry's position, so the resolution is deterministic across runs)."""

    resources_dir = pathlib.Path(resources_dir) if resources_dir is not None else RESOURCES_DIR
    module = _load_module(resources_dir)
    random_probes = module.random_probes

    bank: dict[str, list[PromptEntry]] = {label: [] for label in CATEGORY_LABELS.values()}
    for raw_category, label in CATEGORY_LABELS.items():
        raw_entries = getattr(module, raw_category)
        for index, (system_prompt, probe_question, score_fn) in enumerate(raw_entries):
            if probe_question == "random":
                rng = random.Random(f"{raw_category}:{index}")
                probe_question = rng.choice(random_probes)
            prompt_id = f"{label}_{index:03d}"
            if prompt_id in KNOWN_SATURATED_PROMPT_IDS:
                continue
            bank[label].append(
                PromptEntry(
                    prompt_id=prompt_id,
                    prompt_category=label,
                    system_prompt=system_prompt,
                    probe_question=probe_question,
                    score_fn=score_fn,
                )
            )
    return bank


def select_screening_prompts(
    bank: dict[str, list[PromptEntry]], num_prompts: int, rng_seed: int, avoid_unsafe_scorers: bool = True
) -> list[PromptEntry]:
    """Stratified sample across the two categories, split as evenly as
    possible, deterministic given rng_seed.

    When `avoid_unsafe_scorers` (default), each category's pool is narrowed
    to entries `classify_scorer_screening_safety` calls "ok" before sampling
    -- but only when that narrowed pool still has enough entries for the
    requested `take`; otherwise falls back to the full pool rather than
    raising, so a category thin on continuous scorers doesn't break
    selection."""

    labels = list(bank.keys())
    per_label = num_prompts // len(labels)
    remainder = num_prompts % len(labels)
    selected: list[PromptEntry] = []
    for i, label in enumerate(labels):
        take = per_label + (1 if i < remainder else 0)
        rng = random.Random(rng_seed + i)
        pool = bank[label]
        if avoid_unsafe_scorers:
            safe_pool = [e for e in pool if classify_scorer_screening_safety(e.score_fn)[0]]
            if len(safe_pool) >= take:
                pool = safe_pool
        selected.extend(rng.sample(pool, k=min(take, len(pool))))
    return selected


def score_response(entry: PromptEntry, text: str) -> tuple[float, bool]:
    """Returns (score, scorer_failure). NaN score with scorer_failure=True
    on any exception from the (third-party, sometimes fragile) scoring
    lambda, e.g. division by zero on an empty response."""

    try:
        return float(entry.score_fn(text)), False
    except Exception:
        return float("nan"), True
