#!/usr/bin/env python3
"""Paired comparison for the ERGO/Laban minimal executor-authority check
(docs/feasibility/ERGO_MULTITURN_RELIABILITY_FEASIBILITY.md section 4 step
2): does resetting (wiping history, re-presenting every shard revealed so
far as one consolidated message) move final-turn task success relative to
zero_control, on the same 20 items in both arms?

Per item, average y_task_success at its own final turn (turn == num_shards)
across seeds, then a paired t-test of reset vs zero_control across items --
same design as scripts/analyze_mc_phaseA_comparison.py, adapted for
variable-length trajectories (items here have 4-12 shards, not a fixed T).

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm-a-dir", default="outputs/ergo_math_authority_zero_control", help="baseline arm")
    parser.add_argument("--arm-b-dir", default="outputs/ergo_math_authority_reset", help="comparison arm")
    return parser.parse_args()


def per_item_final_turn_mean(rows: list[dict]) -> dict[str, float]:
    """item_id -> mean y_task_success at turn == num_shards, pooled across seeds."""

    by_traj = group_by_trajectory(rows)
    item_values: dict[str, list[float]] = {}
    for traj_rows in by_traj.values():
        item_id = traj_rows[0]["item_id"]
        final_row = max(traj_rows, key=lambda r: r["turn"])
        assert final_row["turn"] == final_row["num_shards"], "final row should be the last shard's turn"
        item_values.setdefault(item_id, []).append(final_row["y_task_success"])
    return {iid: float(np.mean(vals)) for iid, vals in item_values.items()}


def paired_compare(a_mean: dict[str, float], b_mean: dict[str, float]) -> dict:
    shared = sorted(set(a_mean) & set(b_mean))
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

    a_mean = per_item_final_turn_mean(a_rows)
    b_mean = per_item_final_turn_mean(b_rows)

    print(f"arm_a={args.arm_a_dir}  arm_b={args.arm_b_dir}")
    print("## Final-turn task success, paired by item")
    print(paired_compare(a_mean, b_mean))


if __name__ == "__main__":
    main()
