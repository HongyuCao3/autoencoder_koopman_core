#!/usr/bin/env python3
"""Executor-authority comparison for the MMLU Phase A rerun and its
reminder-design variants, docs/experiments/mc_sycophancy_screening_pilot.md's
Phase A section.

Same design as docs/experiments/sycophancy_screening_pilot.md's Phase A
section on SYCON-Bench: per item, average y_consistency over turns 2-5
(pooling seeds), then a paired t-test of arm A vs arm B across items shared
between the two directories. Also reports the turn-1-baseline-gated version
(drop items where either arm's turn 1 wasn't judged MAINTAINS in every
seed) the way the SYCON-Bench Phase A write-up did post hoc for
sycon_fp_0107, since the baseline diagnostics in this pool flag several such
items too.

Defaults to the original pair (job 15570373 zero_control / 15570374
constant_remind); pass --arm-a-dir/--arm-b-dir to compare any other two
same-item-pool runs, e.g. zero_control against the turn-1-exempt variant
(job 15590462, outputs/mc_sycophancy_phaseA_remind_from_turn2).

CPU-only -- no GPU needed.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

from persona_drift.modeling.dataset import group_by_trajectory, load_trajectories  # noqa: E402

LATE_TURNS = (2, 3, 4, 5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm-a-dir", default="outputs/mc_sycophancy_phaseA_zero_control", help="baseline arm")
    parser.add_argument("--arm-b-dir", default="outputs/mc_sycophancy_phaseA_constant_remind", help="comparison arm")
    return parser.parse_args()


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


def paired_compare(a_mean: dict[str, float], b_mean: dict[str, float], item_ids: set[str]) -> dict:
    shared = sorted(item_ids & a_mean.keys() & b_mean.keys())
    a_vals = np.array([a_mean[i] for i in shared])
    b_vals = np.array([b_mean[i] for i in shared])
    diff = b_vals - a_vals
    if len(shared) >= 2:
        t_stat, p = stats.ttest_rel(b_vals, a_vals)
    else:
        t_stat, p = float("nan"), float("nan")
    return {
        "n_items": len(shared),
        "arm_a_mean": float(a_vals.mean()) if len(shared) else float("nan"),
        "arm_b_mean": float(b_vals.mean()) if len(shared) else float("nan"),
        "mean_diff_b_minus_a": float(diff.mean()) if len(shared) else float("nan"),
        "t": float(t_stat),
        "p": float(p),
        "df": len(shared) - 1,
    }


def main() -> None:
    args = parse_args()
    a_rows = load_trajectories(f"{args.arm_a_dir}/trajectories.jsonl")
    b_rows = load_trajectories(f"{args.arm_b_dir}/trajectories.jsonl")

    a_mean, a_clean = per_item_late_mean(a_rows)
    b_mean, b_clean = per_item_late_mean(b_rows)

    all_items = set(a_mean) | set(b_mean)
    raw = paired_compare(a_mean, b_mean, all_items)

    clean_items = a_clean & b_clean
    gated = paired_compare(a_mean, b_mean, clean_items)

    dropped = sorted(all_items - clean_items)

    print(f"arm_a={args.arm_a_dir}  arm_b={args.arm_b_dir}")
    print("## Raw (all shared items, turns 2-5 pooled across seeds)")
    print(raw)
    print()
    print(f"## Turn-1-baseline-gated (dropped {len(dropped)} items with a non-MAINTAINS turn 1 in "
          f"either arm): {dropped}")
    print(gated)


if __name__ == "__main__":
    main()
