#!/usr/bin/env python3
"""B1 section 4.2 (docs/experiments/defense_line_redesign_plan.md): blind-label
the len1024 zero_control diagnostic arm to see whether lifting the reply cap
from 256 to 1024 tokens raises the blind assisted-rate baseline above D1's
~9.8% (measured at 256 tokens) -- this decides whether the defense line has
anything left to defend, or whether the 256-token cap was hiding it.

Same blind protocol as D1 (build_authority_check_sample.py): only
{uid, harmful_goal, question, reply} reach the labeller. This time there is
only one arm (zero_control_len1024), so no reminder-text leakage risk exists,
but the {uid,...} field discipline still applies.

Outputs (never overwrite):
  outputs/authority_check_len1024/blind_items.json
  outputs/authority_check_len1024/key.json
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

ARM_PATH = pathlib.Path("outputs/koopman_defense_phaseE_zero_control_len1024/trajectories.jsonl")
REJUDGE_SUBDIR = "rejudge_qwen3_4b_instruct_2507"
FULL_POOL_ROOT = pathlib.Path("outputs")
LATE_TURNS = (3, 4, 5)
N_REFUSAL_CONTROLS = 5
N_DUPLICATES = 5


def load_jsonl(path: pathlib.Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def load_full_rejudge_pool() -> list[dict]:
    """Same population as G0/A1/D1: all arms with a rejudge_qwen3_4b_instruct_2507
    subdirectory, excluding the 5 `*_indepjudge` arms."""
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


def build_late_items(rows: list[dict]) -> list[dict]:
    items = []
    for r in rows:
        if r["turn"] not in LATE_TURNS:
            continue
        items.append(
            {
                "kind": "late",
                "arm": "zero_control_len1024",
                "trajectory_id": r["trajectory_id"],
                "turn": r["turn"],
                "attack_id": r["attack_id"],
                "seed": r["seed"],
                "harmful_goal": r["plain_query"],
                "question": r["attacker_query"],
                "reply": r["agent_message"],
                "refusal_flag": bool(r["refusal_flag"]),
                "y_safety_self_judge": r["y_safety"],
                "y_safety_indep": None,  # no rejudge subdir exists for this arm yet
            }
        )
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("outputs/authority_check_len1024"))
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"{args.output_dir} already has output -- refusing to overwrite, per plan rule 5")

    arm_rows = load_jsonl(ARM_PATH)
    n_traj = len(set(r["trajectory_id"] for r in arm_rows))
    assert n_traj == 16, f"expected 16 trajectories, got {n_traj}"

    late_all = build_late_items(arm_rows)
    assert len(late_all) == 48, f"expected 48 late(3,4,5) rows, got {len(late_all)}"
    n_refusal_in_arm = sum(1 for it in late_all if it["refusal_flag"])
    print(f"refusal rows incidentally present among the 48 late rows: {n_refusal_in_arm}")

    full_pool = load_full_rejudge_pool()
    assert len(full_pool) == 1905, f"expected 1905-row full pool, got {len(full_pool)}"
    refusal_pool = [r for r in full_pool if r["refusal_flag"]]
    assert len(refusal_pool) == 85, f"expected 85 refusal rows in the full pool, got {len(refusal_pool)}"
    refusal_pool.sort(key=lambda r: (r["_arm"], r["trajectory_id"], r["turn"]))

    rng = np.random.default_rng(0)

    refusal_idx = rng.choice(len(refusal_pool), size=N_REFUSAL_CONTROLS, replace=False)
    refusal_controls = []
    for i in refusal_idx:
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
    assert len(refusal_controls) == N_REFUSAL_CONTROLS

    dup_idx = rng.choice(len(late_all), size=N_DUPLICATES, replace=False)
    duplicates = []
    for i in dup_idx:
        src = late_all[int(i)]
        dup = dict(src)
        dup["kind"] = "duplicate"
        dup["duplicate_of_source_index"] = int(i)
        duplicates.append(dup)

    all_items = late_all + refusal_controls + duplicates
    assert len(all_items) == 48 + N_REFUSAL_CONTROLS + N_DUPLICATES == 58

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

    blind_blob = json.dumps(blind_items)
    y_safety_leak = blind_blob.count("y_safety")

    gate = {
        "n_items": len(blind_items),
        "key_set": sorted({k for it in blind_items for k in it}),
        "y_safety_string_count": y_safety_leak,
        "n_refusal_in_arm_incidental": n_refusal_in_arm,
        "n_refusal_controls": N_REFUSAL_CONTROLS,
        "n_duplicates": N_DUPLICATES,
    }
    print(json.dumps(gate, indent=2))
    assert gate["n_items"] == 58
    assert gate["key_set"] == ["harmful_goal", "question", "reply", "uid"]
    assert gate["y_safety_string_count"] == 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / "blind_items.json", "w") as f:
        json.dump(blind_items, f, indent=2)
    with open(args.output_dir / "key.json", "w") as f:
        json.dump(key, f, indent=2)
    with open(args.output_dir / "gate_G_B1_2.json", "w") as f:
        json.dump(gate, f, indent=2)

    print(f"wrote {len(blind_items)} items to {args.output_dir}")


if __name__ == "__main__":
    main()
