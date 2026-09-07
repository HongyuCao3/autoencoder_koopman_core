#!/usr/bin/env python3
"""D1 section 1.2 (docs/experiments/defense_line_redesign_plan.md): build the
blinded sample for the executor-authority adjudication check -- does inserting
a safety reminder (`constant_remind`) actually reduce "material assistance"
relative to no defense (`zero_control`), per a third-party adjudicator using
the same blind protocol as P/G1 (measurement_validity_plan.md section 3.2)?

Outputs (never overwrite -- fresh paths only):
  outputs/authority_check/blind_items.json  -- {uid, harmful_goal, question, reply} only
  outputs/authority_check/key.json          -- uid -> arm/trajectory_id/turn/... (G-D1 analysis only)

CRITICAL leakage rule: `constant_remind`'s reminder text is spliced into the
agent-facing stimulus, so `inserted_reminder_text` and anything derived from it
must NEVER be used to build `question` -- use the reminder-free
`attacker_query` instead. A pre-flight 5-gram assertion checks this.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re

import numpy as np

ZERO_CONTROL_PATH = pathlib.Path("outputs/koopman_defense_phaseE_zero_control/trajectories.jsonl")
CONSTANT_REMIND_PATH = pathlib.Path("outputs/koopman_defense_phaseE_constant_remind/trajectories.jsonl")
INDEP_REJUDGE_PATH = pathlib.Path("outputs/koopman_defense_phaseE_zero_control/rejudge_qwen3_4b_instruct_2507/trajectories.jsonl")
REJUDGE_SUBDIR = "rejudge_qwen3_4b_instruct_2507"
FULL_POOL_ROOT = pathlib.Path("outputs")
LATE_TURNS = (3, 4, 5)
N_REFUSAL_CONTROLS = 10
N_DUPLICATES = 10


def load_jsonl(path: pathlib.Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def load_full_rejudge_pool() -> list[dict]:
    """Same population as G0/A1: all arms with a rejudge_qwen3_4b_instruct_2507
    subdirectory, excluding the 5 `*_indepjudge` arms (independent judge
    re-scoring itself)."""
    arm_dirs = [
        d
        for d in FULL_POOL_ROOT.iterdir()
        if d.is_dir()
        and (d / REJUDGE_SUBDIR / "trajectories.jsonl").exists()
        and not d.name.endswith("_indepjudge")
    ]
    rows = []
    for d in sorted(arm_dirs):
        for r in load_jsonl(d / REJUDGE_SUBDIR / "trajectories.jsonl"):
            r["_arm"] = d.name
            rows.append(r)
    return rows


def build_late_items(arm_name: str, rows: list[dict], indep_by_key: dict) -> list[dict]:
    items = []
    for r in rows:
        if r["turn"] not in LATE_TURNS:
            continue
        key = (r["trajectory_id"], r["turn"])
        items.append(
            {
                "kind": "late",
                "arm": arm_name,
                "trajectory_id": r["trajectory_id"],
                "turn": r["turn"],
                "attack_id": r["attack_id"],
                "seed": r["seed"],
                "harmful_goal": r["plain_query"],
                "question": r["attacker_query"],
                "reply": r["agent_message"],
                "refusal_flag": bool(r["refusal_flag"]),
                "y_safety_self_judge": r["y_safety"],
                "y_safety_indep": indep_by_key.get(key),
            }
        )
    return items


def check_reminder_leakage(reminder_text: str, blind_items: list[dict]) -> int:
    words = reminder_text.lower().split()
    fivegrams = {" ".join(words[i : i + 5]) for i in range(len(words) - 4)}
    haystack = "\n".join(
        f"{it['harmful_goal']}\n{it['question']}\n{it['reply']}" for it in blind_items
    ).lower()
    hits = sum(1 for g in fivegrams if g in haystack)
    return hits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("outputs/authority_check"))
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"{args.output_dir} already has output -- refusing to overwrite, per plan rule 5")

    zero_rows = load_jsonl(ZERO_CONTROL_PATH)
    remind_rows = load_jsonl(CONSTANT_REMIND_PATH)

    zero_traj_ids = set(r["trajectory_id"] for r in zero_rows)
    remind_traj_ids = set(r["trajectory_id"] for r in remind_rows)
    assert zero_traj_ids == remind_traj_ids, "the two arms' trajectory_id sets must match exactly"
    assert len(zero_traj_ids) == 16, f"expected 16 shared trajectory_ids, got {len(zero_traj_ids)}"

    indep_rows = load_jsonl(INDEP_REJUDGE_PATH)
    indep_by_key = {(r["trajectory_id"], r["turn"]): r["y_safety"] for r in indep_rows}

    late_zero = build_late_items("zero_control", zero_rows, indep_by_key)
    late_remind = build_late_items("constant_remind", remind_rows, {})  # no rejudge subdir exists for this arm (see deviation note)
    late_all = late_zero + late_remind
    assert len(late_all) == 96, f"expected 96 late(3,4,5) rows across both arms, got {len(late_all)}"

    n_refusal_existing = sum(1 for it in late_all if it["refusal_flag"])
    print(f"existing refusal rows in the 96 late items: {n_refusal_existing}")

    full_pool = load_full_rejudge_pool()
    assert len(full_pool) == 1905, f"expected 1905-row full pool, got {len(full_pool)}"
    refusal_pool = [r for r in full_pool if r["refusal_flag"]]
    assert len(refusal_pool) == 85, f"expected 85 refusal rows in the full pool, got {len(refusal_pool)}"
    refusal_pool.sort(key=lambda r: (r["_arm"], r["trajectory_id"], r["turn"]))

    rng = np.random.default_rng(0)

    n_needed = N_REFUSAL_CONTROLS - n_refusal_existing
    supplement_idx = rng.choice(len(refusal_pool), size=n_needed, replace=False)
    refusal_controls = []
    for i in supplement_idx:
        r = refusal_pool[int(i)]
        refusal_controls.append(
            {
                "kind": "refusal_control",
                "arm": r["_arm"],
                "trajectory_id": r["trajectory_id"],
                "turn": r["turn"],
                "attack_id": r["attack_id"],
                "seed": r["seed"],
                "harmful_goal": r["plain_query"],
                "question": r["attacker_query"],
                "reply": r["agent_message"],
                "refusal_flag": True,
                "y_safety_self_judge": r.get("y_safety_self_judge"),
                "y_safety_indep": r.get("y_safety"),
            }
        )
    existing_refusal_items = [it for it in late_all if it["refusal_flag"]]
    all_refusal_controls = existing_refusal_items + refusal_controls
    assert len(all_refusal_controls) == N_REFUSAL_CONTROLS

    dup_idx = rng.choice(len(late_all), size=N_DUPLICATES, replace=False)
    duplicates = []
    for i in dup_idx:
        src = late_all[int(i)]
        dup = dict(src)
        dup["kind"] = "duplicate"
        dup["duplicate_of_source_index"] = int(i)
        duplicates.append(dup)

    all_items = late_all + all_refusal_controls + duplicates
    assert len(all_items) == 96 + N_REFUSAL_CONTROLS + N_DUPLICATES == 116

    order = rng.permutation(len(all_items))
    shuffled = [all_items[i] for i in order]

    blind_items = []
    key = {}
    for i, it in enumerate(shuffled):
        uid = f"item_{i:04d}"
        blind_items.append(
            {
                "uid": uid,
                "harmful_goal": it["harmful_goal"],
                "question": it["question"],
                "reply": it["reply"],
            }
        )
        key[uid] = {k: v for k, v in it.items() if k not in ("harmful_goal", "question", "reply")}

    reminder_texts = set(r["inserted_reminder_text"] for r in remind_rows if r.get("inserted_reminder_text"))
    assert reminder_texts, "expected at least one non-empty reminder text in constant_remind"
    leak_hits = 0
    for text in reminder_texts:
        leak_hits += check_reminder_leakage(text, blind_items)
    blind_blob = json.dumps(blind_items)
    y_safety_leak = blind_blob.count("y_safety")

    gate = {
        "n_items": len(blind_items),
        "key_set": sorted({k for it in blind_items for k in it}),
        "reminder_5gram_hits": leak_hits,
        "y_safety_string_count": y_safety_leak,
        "n_existing_refusal": n_refusal_existing,
        "n_supplemental_refusal": n_needed,
        "n_refusal_total": len(all_refusal_controls),
        "n_duplicates": N_DUPLICATES,
    }
    print(json.dumps(gate, indent=2))
    assert gate["n_items"] == 116
    assert gate["key_set"] == ["harmful_goal", "question", "reply", "uid"]
    assert gate["reminder_5gram_hits"] == 0, "LEAKAGE: reminder text 5-gram found in blind_items.json -- do not proceed"
    assert gate["y_safety_string_count"] == 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / "blind_items.json", "w") as f:
        json.dump(blind_items, f, indent=2)
    with open(args.output_dir / "key.json", "w") as f:
        json.dump(key, f, indent=2)
    with open(args.output_dir / "gate_G_D1_1.json", "w") as f:
        json.dump(gate, f, indent=2)

    print(f"wrote {len(blind_items)} items to {args.output_dir}")


if __name__ == "__main__":
    main()
