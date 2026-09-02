"""Loads the vendored SYCON-Bench false-presupposition items
(resources/sycon_false_presuppositions.jsonl, see resources/PROVENANCE.md)
used as fixed, pre-generated escalating-pushback content for the
sycophancy-drift screening task (docs/task/SYCOPHANCY_DRIFT_TASK_FEASIBILITY.md).
Mirrors attack_bank.py's load/select shape (same reasoning: fixed,
pre-generated, deterministic disturbance sequences, not adaptive
generation -- see attack_bank.py's own docstring) so screening.py-style
orchestration code looks familiar across every domain.

Only one category exists today ("false_presupposition"); the {category:
[...]} return shape matches attack_bank.py/benign_bank.py so a second
category (e.g. SYCON-Bench's Debate items, see
SYCOPHANCY_DRIFT_TASK_FEASIBILITY.md section 6) can be added later without
changing any caller.
"""

from __future__ import annotations

import json
import pathlib
import random
from dataclasses import dataclass

RESOURCES_DIR = pathlib.Path(__file__).resolve().parents[2] / "resources"


@dataclass(frozen=True)
class SycophancyItem:
    item_id: str
    category: str
    question: str  # turn 1: neutral, no pushback yet
    presupposition: str  # the false claim the pushback tries to get the agent to accept
    correction: str  # the correct fact -- ground truth the judge scores against
    pushback_turns: tuple[str, ...]  # fixed replay: turns 2..N, escalating pushback text

    @property
    def multi_turn_queries(self) -> tuple[str, ...]:
        """Same shape as AttackEntry.multi_turn_queries/BenignEntry.multi_turn_queries
        -- turn 1 is the neutral question, turns 2..N are the pushback script --
        so trajectory_runner.run_reminder_gated_trajectory can replay this
        entry the same way it replays the other two domains'."""
        return (self.question, *self.pushback_turns)


def load_sycophancy_bank(resources_dir: pathlib.Path | None = None) -> dict[str, list[SycophancyItem]]:
    """Returns {category: [SycophancyItem, ...]}. Every item in the current
    vendored file has all four fields populated and >=1 pushback turn (this
    was verified at merge time, see resources/PROVENANCE.md), so unlike
    attack_bank.load_attack_bank there is no length filter here."""

    resources_dir = pathlib.Path(resources_dir) if resources_dir is not None else RESOURCES_DIR
    rows = [
        json.loads(line)
        for line in (resources_dir / "sycon_false_presuppositions.jsonl").read_text().splitlines()
        if line.strip()
    ]

    bank: dict[str, list[SycophancyItem]] = {}
    for row in rows:
        item = SycophancyItem(
            item_id=row["item_id"],
            category=row["category"],
            question=row["question"],
            presupposition=row["presupposition"],
            correction=row["correction"],
            pushback_turns=tuple(row["pushback_turns"]),
        )
        bank.setdefault(item.category, []).append(item)
    return bank


def select_screening_items(
    bank: dict[str, list[SycophancyItem]], num_items: int, rng_seed: int
) -> list[SycophancyItem]:
    """Stratified sample across categories, split as evenly as possible,
    deterministic given rng_seed. Same shape as
    attack_bank.select_screening_attacks/prompt_bank.select_screening_prompts."""

    labels = sorted(bank.keys())
    per_label = num_items // len(labels)
    remainder = num_items % len(labels)
    selected: list[SycophancyItem] = []
    for i, label in enumerate(labels):
        take = per_label + (1 if i < remainder else 0)
        rng = random.Random(rng_seed + i)
        pool = bank[label]
        selected.extend(rng.sample(pool, k=min(take, len(pool))))
    return selected


def select_items_by_id(bank: dict[str, list[SycophancyItem]], item_ids: list[str]) -> list[SycophancyItem]:
    """Look up specific items by id across all categories, preserving
    `item_ids`'s order -- for replaying an exact, externally-determined item
    set (e.g. a held-out split), mirrors attack_bank.select_attacks_by_id.
    Raises KeyError if any id isn't found."""

    by_id = {item.item_id: item for items in bank.values() for item in items}
    return [by_id[item_id] for item_id in item_ids]
