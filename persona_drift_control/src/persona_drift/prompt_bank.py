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
            bank[label].append(
                PromptEntry(
                    prompt_id=f"{label}_{index:03d}",
                    prompt_category=label,
                    system_prompt=system_prompt,
                    probe_question=probe_question,
                    score_fn=score_fn,
                )
            )
    return bank


def select_screening_prompts(
    bank: dict[str, list[PromptEntry]], num_prompts: int, rng_seed: int
) -> list[PromptEntry]:
    """Stratified sample across the two categories, split as evenly as
    possible, deterministic given rng_seed."""

    labels = list(bank.keys())
    per_label = num_prompts // len(labels)
    remainder = num_prompts % len(labels)
    selected: list[PromptEntry] = []
    for i, label in enumerate(labels):
        take = per_label + (1 if i < remainder else 0)
        rng = random.Random(rng_seed + i)
        pool = bank[label]
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
