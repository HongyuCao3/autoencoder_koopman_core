"""Loads the vendored ActorAttack multi-turn jailbreak sequences (SafeMTData
Attack_600, see resources/PROVENANCE.md) used as fixed, pre-generated
attacker input for the adversarial-defense screening task
(docs/ADVERSARIAL_DEFENSE_TASK_FEASIBILITY.md section 7 step 1). Mirrors
prompt_bank.py's load/select shape so screening.py-style orchestration code
looks familiar.
"""

from __future__ import annotations

import json
import pathlib
import random
from dataclasses import dataclass

RESOURCES_DIR = pathlib.Path(__file__).resolve().parents[2] / "resources"


@dataclass(frozen=True)
class AttackEntry:
    attack_id: str
    category: str
    actor_name: str
    plain_query: str  # the underlying harmful goal the multi-turn sequence works toward
    multi_turn_queries: tuple[str, ...]  # fixed replay: query text per turn, oldest first


def load_attack_bank(resources_dir: pathlib.Path | None = None) -> dict[str, list[AttackEntry]]:
    """Returns {category: [AttackEntry, ...]}, grouped by the upstream
    dataset's own `category` field (e.g. "chemical_biological",
    "cybercrime_intrusion"). Entries with fewer than 3 queries are dropped
    (too short for a turn-level OLS slope)."""

    resources_dir = pathlib.Path(resources_dir) if resources_dir is not None else RESOURCES_DIR
    raw = json.loads((resources_dir / "safemtdata_attack_600.json").read_text())

    bank: dict[str, list[AttackEntry]] = {}
    for row in raw:
        queries = tuple(row["multi_turn_queries"])
        if len(queries) < 3:
            continue
        entry = AttackEntry(
            attack_id=f"safemtdata_{row['id']:04d}",
            category=row["category"],
            actor_name=row["actor_name"],
            plain_query=row["plain_query"],
            multi_turn_queries=queries,
        )
        bank.setdefault(entry.category, []).append(entry)
    return bank


def select_screening_attacks(
    bank: dict[str, list[AttackEntry]], num_attacks: int, rng_seed: int
) -> list[AttackEntry]:
    """Stratified sample across categories, split as evenly as possible,
    deterministic given rng_seed. Same shape as
    prompt_bank.select_screening_prompts."""

    labels = sorted(bank.keys())
    per_label = num_attacks // len(labels)
    remainder = num_attacks % len(labels)
    selected: list[AttackEntry] = []
    for i, label in enumerate(labels):
        take = per_label + (1 if i < remainder else 0)
        rng = random.Random(rng_seed + i)
        pool = bank[label]
        selected.extend(rng.sample(pool, k=min(take, len(pool))))
    return selected
