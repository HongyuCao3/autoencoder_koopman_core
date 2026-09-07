#!/usr/bin/env python3
"""F3 Phase C comparison (docs/experiments/signal_resolution_plan.md
section 4.5): paired-by-item bootstrap on `final_turn_success` across the
five Phase C arms (zero_control / always_reset / fixed_t{1..4} / randsched_p100
/ mpc), plus the token-cost table section 4.2 requires ("currency is
tokens, not counts").

Item pairing: all arms are run on the exact same 58-item, 2-seed set
(--item-ids, section 4.4), so `final_turn_success` per item (averaged
across its 2 seeds) is directly comparable arm-to-arm without any
re-weighting.

Pre-registered gates (write down before running, do not adjust after
seeing the numbers):
1. gate 1: mpc - zero_control, 95% CI excludes 0 and is positive.
2. gate 2 (primary): mpc - randsched_p100 (equal-cost), 95% CI excludes 0
   and is positive.
3. gate 3 (reference, not admission): mpc - max_k(fixed_t{k}).
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from persona_drift.modeling.dataset import group_by_trajectory, load_trajectories  # noqa: E402

N_BOOTSTRAP = 10000
BOOTSTRAP_SEED = 0

ARM_DIRS = {
    "zero_control": "outputs/ergo_math_phaseC_zero_control",
    "always_reset": "outputs/ergo_math_phaseC_always_reset",
    "fixed_t1": "outputs/ergo_math_phaseC_fixed_t1",
    "fixed_t2": "outputs/ergo_math_phaseC_fixed_t2",
    "fixed_t3": "outputs/ergo_math_phaseC_fixed_t3",
    "fixed_t4": "outputs/ergo_math_phaseC_fixed_t4",
    "randsched_p100": "outputs/ergo_math_phaseC_randsched_p100",
    "mpc": "outputs/ergo_math_phaseC_mpc",
    # G4 (measurement_validity_plan.md section 7.1/7.2): the missing optimal
    # fixed arm -- reset on each item's own last shard turn.
    "fixed_last": "outputs/ergo_math_phaseC_fixed_last",
}


def _per_item_final_turn_success(rows: list[dict]) -> dict[str, float]:
    """Mean `y_task_success` at each trajectory's own final turn
    (turn == num_shards), averaged across the 2 seeds sharing an item_id."""

    per_item: dict[str, list[float]] = {}
    for traj in group_by_trajectory(rows).values():
        final_row = max(traj, key=lambda r: r["turn"])
        assert final_row["turn"] == final_row["num_shards"], "trajectory did not run to completion"
        per_item.setdefault(final_row["item_id"], []).append(float(final_row["y_task_success"]))
    return {item: float(np.mean(vals)) for item, vals in per_item.items()}


def _token_cost_per_trajectory(rows: list[dict]) -> float:
    total_tokens = sum(int(r.get("inserted_tokens", 0)) for r in rows)
    n_trajectories = len(group_by_trajectory(rows))
    return total_tokens / n_trajectories


def _paired_bootstrap(diffs: np.ndarray, n_resamples: int = N_BOOTSTRAP, seed: int = BOOTSTRAP_SEED) -> dict:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diffs), size=(n_resamples, len(diffs)))
    means = diffs[idx].mean(axis=1)
    return {
        "mean_diff": float(diffs.mean()),
        "ci_low": float(np.percentile(means, 2.5)),
        "ci_high": float(np.percentile(means, 97.5)),
        "n_pairs": int(len(diffs)),
        "excludes_zero_and_positive": bool(np.percentile(means, 2.5) > 0),
    }


def _diff_array(per_item_a: dict, per_item_b: dict) -> np.ndarray:
    common_items = sorted(set(per_item_a) & set(per_item_b))
    assert len(common_items) == len(per_item_a) == len(per_item_b), "item sets differ between arms -- stop and report"
    return np.array([per_item_a[i] - per_item_b[i] for i in common_items])


def main() -> None:
    loaded = {}
    for arm, directory in ARM_DIRS.items():
        path = pathlib.Path(directory) / "trajectories.jsonl"
        if not path.exists():
            print(f"!! {arm}: {path} missing (arm not run yet), skipped")
            continue
        rows = load_trajectories(path)
        per_item = _per_item_final_turn_success(rows)
        loaded[arm] = {
            "per_item": per_item,
            "mean_final_turn_success": float(np.mean(list(per_item.values()))),
            "n_items": len(per_item),
            "token_cost_per_trajectory": _token_cost_per_trajectory(rows),
        }

    print("\n=== per-arm summary ===")
    print(f"{'arm':<18}{'n_items':>8}{'mean_success':>14}{'tokens/traj':>14}")
    for arm, d in loaded.items():
        print(f"{arm:<18}{d['n_items']:>8}{d['mean_final_turn_success']:>14.4f}{d['token_cost_per_trajectory']:>14.1f}")

    if "mpc" not in loaded:
        print("\nmpc arm not available yet -- gates cannot be computed.")
        return

    gates = {}
    if "zero_control" in loaded:
        diffs = _diff_array(loaded["mpc"]["per_item"], loaded["zero_control"]["per_item"])
        gates["gate1_mpc_vs_zero_control"] = _paired_bootstrap(diffs)
    if "randsched_p100" in loaded:
        diffs = _diff_array(loaded["mpc"]["per_item"], loaded["randsched_p100"]["per_item"])
        gates["gate2_mpc_vs_randsched_p100"] = _paired_bootstrap(diffs)
    fixed_arms = [a for a in ("fixed_t1", "fixed_t2", "fixed_t3", "fixed_t4") if a in loaded]
    if fixed_arms:
        best_fixed_arm = max(fixed_arms, key=lambda a: loaded[a]["mean_final_turn_success"])
        diffs = _diff_array(loaded["mpc"]["per_item"], loaded[best_fixed_arm]["per_item"])
        gates["gate3_mpc_vs_best_fixed_t"] = {"best_fixed_arm": best_fixed_arm, **_paired_bootstrap(diffs)}

    # G4 (measurement_validity_plan.md section 7.2): is the k=1 count-budget
    # setting degenerate -- is the optimal single reset point just "the last
    # shard turn", a fixed rule needing no feedback?
    if "fixed_last" in loaded:
        if "randsched_p100" in loaded:
            diffs = _diff_array(loaded["fixed_last"]["per_item"], loaded["randsched_p100"]["per_item"])
            gates["gate4_fixed_last_vs_randsched_p100"] = _paired_bootstrap(diffs)
        if "fixed_t4" in loaded:
            diffs = _diff_array(loaded["fixed_last"]["per_item"], loaded["fixed_t4"]["per_item"])
            gates["gate5_fixed_last_vs_fixed_t4"] = _paired_bootstrap(diffs)
        if "always_reset" in loaded:
            diffs = _diff_array(loaded["always_reset"]["per_item"], loaded["fixed_last"]["per_item"])
            gates["gate6_always_reset_vs_fixed_last"] = _paired_bootstrap(diffs)

    print("\n=== pre-registered gates ===")
    for name, g in gates.items():
        extra = f" (best={g['best_fixed_arm']})" if "best_fixed_arm" in g else ""
        print(
            f"{name}{extra}: mean_diff={g['mean_diff']:+.4f} 95% CI=[{g['ci_low']:+.4f}, {g['ci_high']:+.4f}] "
            f"n_pairs={g['n_pairs']} excludes_zero_and_positive={g['excludes_zero_and_positive']}"
        )

    # G4 added the fixed_last arm and gates 4-6 on top of the original F3
    # report; written to a new path rather than overwriting the pre-existing
    # F3-only outputs/ergo_case_study/phaseC_comparison_report.json.
    out_path = pathlib.Path("outputs/ergo_case_study/phaseC_comparison_report_g4.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        raise SystemExit(f"refusing to overwrite existing {out_path}")
    out_path.write_text(json.dumps({"arms": loaded, "gates": gates}, indent=2))
    print(f"\nreport written to {out_path}")


if __name__ == "__main__":
    main()
