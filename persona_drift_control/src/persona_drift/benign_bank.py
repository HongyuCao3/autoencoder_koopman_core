"""Loads the vendored MT-Bench question set (resources/mtbench_questions.jsonl,
see PROVENANCE.md) as fixed, pre-generated benign multi-turn session content --
the Phase F counterpart of attack_bank.py, used by the helpfulness-cost check
(docs/experiments/koopman_defense_pilot.md).

MT-Bench entries are only 2 turns each, but
control.KoopmanMPCController._current_state needs
len(history) >= max(nu-1, mu) + 1 = 3 (for the nu=1, mu=2 config Phase E
fitted) before it ever produces a non-default action -- a single MT-Bench
entry can't exercise the controller at all. Each session here chains 3
same-category entries (fixed order: lowest question_id first, deterministic,
no sampling) into one 6-turn session, so the controller gets turns 4-6 to
actually act on, not just a single opportunity at the very end.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass

RESOURCES_DIR = pathlib.Path(__file__).resolve().parents[2] / "resources"

ENTRIES_PER_SESSION = 3


@dataclass(frozen=True)
class BenignEntry:
    benign_id: str
    category: str
    multi_turn_queries: tuple[str, ...]


def load_benign_bank(resources_dir: pathlib.Path | None = None) -> dict[str, list[BenignEntry]]:
    """Returns {category: [BenignEntry]}, one session per category, built
    from that category's first ENTRIES_PER_SESSION MT-Bench questions
    (sorted by question_id). Session content is fixed regardless of seed --
    same convention as attack_bank.AttackEntry.multi_turn_queries, where seed
    only varies generation/judge sampling, never what's asked."""

    resources_dir = pathlib.Path(resources_dir) if resources_dir is not None else RESOURCES_DIR
    rows = [json.loads(line) for line in (resources_dir / "mtbench_questions.jsonl").read_text().splitlines() if line.strip()]

    by_category: dict[str, list[dict]] = {}
    for row in rows:
        by_category.setdefault(row["category"], []).append(row)

    bank: dict[str, list[BenignEntry]] = {}
    for category, category_rows in by_category.items():
        category_rows = sorted(category_rows, key=lambda r: r["question_id"])[:ENTRIES_PER_SESSION]
        queries = tuple(turn for row in category_rows for turn in row["turns"])
        bank[category] = [
            BenignEntry(benign_id=f"mtbench_{category}", category=category, multi_turn_queries=queries)
        ]
    return bank


def all_benign_sessions(bank: dict[str, list[BenignEntry]]) -> list[BenignEntry]:
    """Flattens the bank into a deterministically-ordered list (sorted by
    category name) -- this pilot always uses every category's session, no
    sub-sampling, unlike attack_bank's select_screening_attacks."""

    return [entry for category in sorted(bank) for entry in bank[category]]
