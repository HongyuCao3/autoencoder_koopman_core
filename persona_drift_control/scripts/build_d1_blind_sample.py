#!/usr/bin/env python3
"""D1 baseline-rate screening (docs/experiments/two_task_success_plan.md
section four, D1): build the blinded sample used to compute each arm's
assisted rate for gate G-D1-1.

Per arm (screen_qwen4b, screen_alt_qwen3_4b_instruct_2507):
  - 60 rows drawn uniformly (rng.default_rng(0), no replacement) from that
    arm's late(3,4,5) trajectories (300 rows/arm).
  - 5 refusal controls, drawn from that arm's refusal_flag==True rows
    (any turn), same rng stream.
  - 5 duplicates of already-drawn main-sample rows (new uid, identical
    content), for within-round self-consistency.
  => 70 items/arm, 140 total.

The two arms' 140 items are pooled and shuffled together (arm identity is
NOT put in blind_items.json -- failure mode 6 forbids leaking arm identity
into a paired comparison, and the labeler must not be able to tell which
arm an item came from either).

Outputs (never overwrite -- fresh paths only):
  outputs/d1_blind_sample/blind_items.json  -- {uid, harmful_goal, question, reply} only
  outputs/d1_blind_sample/key.json          -- uid -> arm/trajectory_id/turn/kind/... (analysis only)
  outputs/d1_blind_sample/gate_G_D1_sample.json -- pre-flight leakage/shape checks
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
ARMS = {
    "screen_qwen4b": REPO_ROOT / "outputs/d1_screen_qwen4b/trajectories.jsonl",
    "screen_alt_qwen3_4b_instruct_2507": REPO_ROOT
    / "outputs/d1_screen_alt_qwen3_4b_instruct_2507/trajectories.jsonl",
}
LATE_TURNS = (3, 4, 5)
N_MAIN = 60
N_REFUSAL = 5
N_DUPLICATES = 5
OUT_DIR = REPO_ROOT / "outputs/d1_blind_sample"


def load_jsonl(path: pathlib.Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def to_item(r: dict, arm: str, kind: str) -> dict:
    return {
        "kind": kind,
        "arm": arm,
        "trajectory_id": r["trajectory_id"],
        "attack_id": r["attack_id"],
        "turn": r["turn"],
        "seed": r["seed"],
        "category": r["category"],
        "refusal_flag": bool(r["refusal_flag"]),
        "y_safety_self_judge": r["y_safety"],
        "harmful_goal": r["plain_query"],
        "question": r["attacker_query"],
        "reply": r["agent_message"],
    }


def build_arm_sample(arm: str, rows: list[dict], rng: np.random.Generator) -> list[dict]:
    assert len(rows) == 500, f"{arm}: expected 500 rows, got {len(rows)}"
    late_rows = [r for r in rows if r["turn"] in LATE_TURNS]
    assert len(late_rows) == 300, f"{arm}: expected 300 late rows, got {len(late_rows)}"

    main_idx = rng.choice(len(late_rows), size=N_MAIN, replace=False)
    main_items = [to_item(late_rows[int(i)], arm, "main") for i in main_idx]

    refusal_pool = [r for r in rows if r["refusal_flag"]]
    assert len(refusal_pool) >= N_REFUSAL, (
        f"{arm}: need {N_REFUSAL} refusal rows, pool only has {len(refusal_pool)}"
    )
    refusal_idx = rng.choice(len(refusal_pool), size=N_REFUSAL, replace=False)
    refusal_items = [to_item(refusal_pool[int(i)], arm, "refusal_control") for i in refusal_idx]

    dup_idx = rng.choice(len(main_items), size=N_DUPLICATES, replace=False)
    dup_items = []
    for i in dup_idx:
        d = dict(main_items[int(i)])
        d["kind"] = "duplicate"
        d["duplicate_of_trajectory_turn"] = (d["trajectory_id"], d["turn"])
        dup_items.append(d)

    arm_items = main_items + refusal_items + dup_items
    assert len(arm_items) == N_MAIN + N_REFUSAL + N_DUPLICATES == 70
    return arm_items


def main() -> None:
    if OUT_DIR.exists() and any(OUT_DIR.iterdir()):
        raise SystemExit(f"{OUT_DIR} already has output -- refusing to overwrite")

    all_items: list[dict] = []
    for arm, path in ARMS.items():
        rng = np.random.default_rng(0)  # fresh, arm-local stream per plan wording ("各抽")
        rows = load_jsonl(path)
        all_items.extend(build_arm_sample(arm, rows, rng))

    assert len(all_items) == 140

    shuffle_rng = np.random.default_rng(0)
    order = shuffle_rng.permutation(len(all_items))
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
    leak_hits = sum(blind_blob.count(arm) for arm in ARMS)
    y_safety_leak = blind_blob.count("y_safety")

    gate = {
        "n_items": len(blind_items),
        "n_per_arm": {arm: sum(1 for v in key.values() if v["arm"] == arm) for arm in ARMS},
        "key_set": sorted({k for it in blind_items for k in it}),
        "arm_name_leak_hits": leak_hits,
        "y_safety_string_count": y_safety_leak,
    }
    print(json.dumps(gate, indent=2))
    assert gate["n_items"] == 140
    assert gate["key_set"] == ["harmful_goal", "question", "reply", "uid"]
    assert gate["arm_name_leak_hits"] == 0, "LEAKAGE: arm name found in blind_items.json"
    assert gate["y_safety_string_count"] == 0
    assert all(n == 70 for n in gate["n_per_arm"].values())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "blind_items.json", "w") as f:
        json.dump(blind_items, f, indent=2)
    with open(OUT_DIR / "key.json", "w") as f:
        json.dump(key, f, indent=2)
    with open(OUT_DIR / "gate_G_D1_sample.json", "w") as f:
        json.dump(gate, f, indent=2)

    print(f"wrote {len(blind_items)} items to {OUT_DIR}")


if __name__ == "__main__":
    main()
