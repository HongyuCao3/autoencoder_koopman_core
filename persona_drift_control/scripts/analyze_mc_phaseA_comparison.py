#!/usr/bin/env python3
"""Executor-authority comparison for the MMLU Phase A rerun (job 15570373
zero_control / 15570374 constant_remind), docs/experiments/
mc_sycophancy_screening_pilot.md's next-step 2.

Same design as docs/experiments/sycophancy_screening_pilot.md's Phase A
section on SYCON-Bench: per item, average y_consistency over turns 2-5
(pooling seeds), then a paired t-test of zero_control vs constant_remind
across items sharing the same --item-rng-seed 1 draw. Also reports the
turn-1-baseline-gated version (drop items where either arm's turn 1 wasn't
judged MAINTAINS in every seed) the way the SYCON-Bench Phase A write-up
did post hoc for sycon_fp_0107, since the baseline diagnostics in both arms'
own screening reports flag several such items here too.

CPU-only -- no GPU needed.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

from persona_drift.modeling.dataset import group_by_trajectory, load_trajectories  # noqa: E402

ZERO_CONTROL_DIR = "outputs/mc_sycophancy_phaseA_zero_control"
CONSTANT_REMIND_DIR = "outputs/mc_sycophancy_phaseA_constant_remind"
LATE_TURNS = (2, 3, 4, 5)


def per_item_late_mean(rows: list[dict]) -> tuple[dict[str, float], set[str]]:
    """Returns (item_id -> mean y_consistency over turns 2-5, pooled across
    seeds), and the set of item_ids where every seed's turn 1 was MAINTAINS."""
    by_traj = group_by_trajectory(rows)
    item_late_values: dict[str, list[float]] = {}
    item_turn1_ok: dict[str, list[bool]] = {}
    for traj_rows in by_traj.values():
        item_id = traj_rows[0]["item_id"]
        turn1 = next(r for r in traj_rows if r["turn"] == 1)
        item_turn1_ok.setdefault(item_id, []).append(turn1["stance_label"] == "MAINTAINS")
        late = [r["y_consistency"] for r in traj_rows if r["turn"] in LATE_TURNS]
        item_late_values.setdefault(item_id, []).extend(late)
    item_mean = {iid: float(np.mean(vals)) for iid, vals in item_late_values.items()}
    clean_items = {iid for iid, oks in item_turn1_ok.items() if all(oks)}
    return item_mean, clean_items


def paired_compare(zero_mean: dict[str, float], remind_mean: dict[str, float], item_ids: set[str]) -> dict:
    shared = sorted(item_ids & zero_mean.keys() & remind_mean.keys())
    zero_vals = np.array([zero_mean[i] for i in shared])
    remind_vals = np.array([remind_mean[i] for i in shared])
    diff = remind_vals - zero_vals
    if len(shared) >= 2:
        t_stat, p = stats.ttest_rel(remind_vals, zero_vals)
    else:
        t_stat, p = float("nan"), float("nan")
    return {
        "n_items": len(shared),
        "zero_control_mean": float(zero_vals.mean()) if len(shared) else float("nan"),
        "constant_remind_mean": float(remind_vals.mean()) if len(shared) else float("nan"),
        "mean_diff_remind_minus_zero": float(diff.mean()) if len(shared) else float("nan"),
        "t": float(t_stat),
        "p": float(p),
        "df": len(shared) - 1,
    }


def main() -> None:
    zero_rows = load_trajectories(f"{ZERO_CONTROL_DIR}/trajectories.jsonl")
    remind_rows = load_trajectories(f"{CONSTANT_REMIND_DIR}/trajectories.jsonl")

    zero_mean, zero_clean = per_item_late_mean(zero_rows)
    remind_mean, remind_clean = per_item_late_mean(remind_rows)

    all_items = set(zero_mean) | set(remind_mean)
    raw = paired_compare(zero_mean, remind_mean, all_items)

    clean_items = zero_clean & remind_clean
    gated = paired_compare(zero_mean, remind_mean, clean_items)

    dropped = sorted(all_items - clean_items)

    print("## Raw (all shared items, turns 2-5 pooled across seeds)")
    print(raw)
    print()
    print(f"## Turn-1-baseline-gated (dropped {len(dropped)} items with a non-MAINTAINS turn 1 in "
          f"either arm): {dropped}")
    print(gated)


if __name__ == "__main__":
    main()
