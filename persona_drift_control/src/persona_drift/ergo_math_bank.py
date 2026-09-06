"""Loads the vendored ERGO/Laban sharded-GSM8K item bank
(resources/ergo_gsm8k_sharded.jsonl, see resources/PROVENANCE.md) for the
minimal executor-authority check docs/feasibility/
ERGO_MULTITURN_RELIABILITY_FEASIBILITY.md proposes as step 1 of the
multi-turn-reliability candidate line: does a "reset" actuator move a
per-turn readout on this project's own model, before any bank/judge/
trajectory-runner engineering is scaled up.

Single-task bank (unlike mc_sycophancy_bank.py's 57 MMLU categories), so no
per-category stratification -- `select_screening_items` is a flat random
sample.
"""

from __future__ import annotations

import json
import pathlib
import random
from dataclasses import dataclass

RESOURCES_DIR = pathlib.Path(__file__).resolve().parents[2] / "resources"


@dataclass(frozen=True)
class GSM8KShardedItem:
    item_id: str
    gold_answer: str  # numeric string, from upstream's "####"-delimited GSM8K answer
    shards: tuple[str, ...]  # revealed one per turn, in this stored order


def load_ergo_math_bank(resources_dir: pathlib.Path | None = None) -> list[GSM8KShardedItem]:
    resources_dir = pathlib.Path(resources_dir) if resources_dir is not None else RESOURCES_DIR
    rows = [
        json.loads(line)
        for line in (resources_dir / "ergo_gsm8k_sharded.jsonl").read_text().splitlines()
        if line.strip()
    ]
    return [
        GSM8KShardedItem(item_id=row["item_id"], gold_answer=row["gold_answer"], shards=tuple(row["shards"]))
        for row in rows
    ]


def select_screening_items(bank: list[GSM8KShardedItem], num_items: int, rng_seed: int) -> list[GSM8KShardedItem]:
    rng = random.Random(rng_seed)
    return rng.sample(bank, k=min(num_items, len(bank)))


def select_items_by_id(bank: list[GSM8KShardedItem], item_ids: list[str]) -> list[GSM8KShardedItem]:
    """Preserves `item_ids`'s order. Raises KeyError if any id isn't found."""

    by_id = {item.item_id: item for item in bank}
    return [by_id[item_id] for item_id in item_ids]
