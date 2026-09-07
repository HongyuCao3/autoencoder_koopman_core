#!/usr/bin/env python3
"""Build blind-labelling samples for the judge-validity study
(docs/experiments/measurement_validity_plan.md).

`--pilot` builds task P's 70-item disagreement-layer pilot
(`outputs/judge_validation/pilot_blind_items.json` +
`pilot_key.json`). Field extraction, blinding rules, and uid numbering
follow measurement_validity_plan.md section 2 exactly: `harmful_goal =
row["plain_query"]`, `question = row["attacker_query"]`, `reply =
row["agent_message"]` (src/persona_drift/rejudge.py:73-76 is the judge
call site these are copied from). Only `--pilot` is implemented -- the
full G0 265-item sample (section two of the plan) is a separate,
larger draw that is only built if P's confidence interval does not
resolve the judge question on its own, and is not needed yet.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
POOL_GLOB = "outputs/*/rejudge_qwen3_4b_instruct_2507/trajectories.jsonl"
OUT_DIR = REPO_ROOT / "outputs" / "judge_validation"

SELF_FIELD = "y_safety_self_judge"
INDEP_FIELD = "y_safety"


def load_pool() -> list[dict[str, Any]]:
    """All rows from open-loop arms (excludes the 5 `*_indepjudge` arms,
    which are the independent judge re-scoring itself). Deterministic
    order: sorted by (arm, trajectory_id, turn) so the draw below does
    not depend on filesystem iteration order."""

    files = sorted(REPO_ROOT.glob(POOL_GLOB))
    files = [f for f in files if not f.parent.parent.name.endswith("_indepjudge")]
    rows: list[dict[str, Any]] = []
    for f in files:
        arm = f.parent.parent.name
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            row["_arm"] = arm
            rows.append(row)
    rows.sort(key=lambda r: (r["_arm"], r["trajectory_id"], r["turn"]))
    return rows


def to_item(row: dict[str, Any], uid: str) -> dict[str, str]:
    return {
        "uid": uid,
        "harmful_goal": row["plain_query"],
        "question": row["attacker_query"],
        "reply": row["agent_message"],
    }


def key_entry(row: dict[str, Any], stratum: str, duplicate_of: str | None = None) -> dict[str, Any]:
    entry = {
        "arm": row["_arm"],
        "trajectory_id": row["trajectory_id"],
        "turn": row["turn"],
        "seed": row["seed"],
        "y_safety_self_judge": row[SELF_FIELD],
        "y_safety": row[INDEP_FIELD],
        "refusal_flag": row["refusal_flag"],
        "stratum": stratum,
    }
    if duplicate_of is not None:
        entry["duplicate_of"] = duplicate_of
    return entry


def build_pilot() -> None:
    rows = load_pool()

    stratum_a = [r for r in rows if r[SELF_FIELD] < 1.0 and r[INDEP_FIELD] == 1.0]
    stratum_c_nonrefusal = [
        r for r in rows if r[SELF_FIELD] == 1.0 and r[INDEP_FIELD] == 1.0 and not r["refusal_flag"]
    ]
    refusal_pool = [r for r in rows if r["refusal_flag"]]

    expected_pools = {"A": (stratum_a, 610), "C_nonrefusal": (stratum_c_nonrefusal, 1044), "refusal": (refusal_pool, 85)}
    for name, (pool, expected) in expected_pools.items():
        if len(pool) != expected:
            print(f"FATAL: pool {name} has {len(pool)} rows, expected {expected}. Stopping.", file=sys.stderr)
            sys.exit(1)

    rng = np.random.default_rng(0)

    def draw(pool: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
        idx = rng.choice(len(pool), size=n, replace=False)
        return [pool[i] for i in idx]

    drawn_a = draw(stratum_a, 40)
    drawn_c = draw(stratum_c_nonrefusal, 10)
    drawn_refusal = draw(refusal_pool, 10)

    primary = (
        [(r, "A") for r in drawn_a]
        + [(r, "C_nonrefusal") for r in drawn_c]
        + [(r, "refusal") for r in drawn_refusal]
    )

    # Fix which 10 of the 60 primary draws get duplicated *before* the
    # primary list is shuffled, so the positions still mean something.
    dup_rows = [primary[i] for i in rng.choice(len(primary), size=10, replace=False)]

    rng.shuffle(primary)

    blind_items: list[dict[str, str]] = []
    key: dict[str, Any] = {}
    uid_of_row: dict[tuple[str, str, int], str] = {}
    for i, (row, stratum) in enumerate(primary):
        uid = f"item_{i:04d}"
        blind_items.append(to_item(row, uid))
        key[uid] = key_entry(row, stratum)
        uid_of_row[(row["_arm"], row["trajectory_id"], row["turn"])] = uid

    for j, (row, _stratum) in enumerate(dup_rows):
        uid = f"item_{len(primary) + j:04d}"
        original_uid = uid_of_row[(row["_arm"], row["trajectory_id"], row["turn"])]
        blind_items.append(to_item(row, uid))
        key[uid] = key_entry(row, "duplicate", duplicate_of=original_uid)

    rng.shuffle(blind_items)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    blind_path = OUT_DIR / "pilot_blind_items.json"
    key_path = OUT_DIR / "pilot_key.json"
    if blind_path.exists() or key_path.exists():
        print(f"FATAL: {blind_path} or {key_path} already exists. Refusing to overwrite.", file=sys.stderr)
        sys.exit(1)

    blind_path.write_text(json.dumps(blind_items, indent=2))
    key_path.write_text(json.dumps(key, indent=2))

    print(f"wrote {len(blind_items)} items to {blind_path}")
    print(f"wrote key for {len(key)} uids to {key_path}")
    print(f"strata: A={len(drawn_a)} C_nonrefusal={len(drawn_c)} refusal={len(drawn_refusal)} duplicate={len(dup_rows)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pilot", action="store_true", required=True, help="build task P's 70-item pilot sample")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.pilot:
        build_pilot()


if __name__ == "__main__":
    main()
