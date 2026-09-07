#!/usr/bin/env python3
"""T1a follow-up (docs/experiments/adaptive_vs_fixed_claim_plan.md section 二):
recomputes the two claim gates with `threshold_ymin1` (the independent-judge,
most-sensitive-threshold rerun) added, WITHOUT modifying
`scripts/analyze_readout_state.py` or overwriting its
`outputs/koopman_case_study/readout_state_report.json` (T0's committed
artifact -- this script only imports its helpers).

`threshold_ymin1` has no self-judged counterpart (it only exists under the
independent judge, by construction), so it is compared only against
`zero_control`'s independent-judge numbers -- exactly the comparison this
arm exists to inform (plan section 0.1).

CPU-only. Run directly (no sbatch).
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[0]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from analyze_readout_state import (  # noqa: E402
    ARM_SOURCES,
    mixed_baseline_bootstrap,
    paired_bootstrap,
    per_trajectory_arm_metrics,
)

THRESHOLD_YMIN1_PATH = "outputs/koopman_defense_phaseJ_budget1_threshold_indepjudge_ymin1/trajectories.jsonl"
OUT_PATH = pathlib.Path("outputs/koopman_case_study/readout_state_report_t1a.json")


def main() -> None:
    if OUT_PATH.exists():
        raise SystemExit(f"refusing to overwrite existing {OUT_PATH}")

    # Same 8-arm self/indep metrics T0 computes, plus the new arm's
    # independent-judge-only metrics.
    self_metrics = {arm: per_trajectory_arm_metrics(paths[0]) for arm, paths in ARM_SOURCES.items()}
    indep_metrics = {arm: per_trajectory_arm_metrics(paths[1]) for arm, paths in ARM_SOURCES.items()}
    threshold_ymin1_metrics = per_trajectory_arm_metrics(THRESHOLD_YMIN1_PATH)

    common = set.intersection(*(set(m) for m in self_metrics.values()))
    common &= set.intersection(*(set(m) for m in indep_metrics.values()))
    common &= set(threshold_ymin1_metrics)
    common = sorted(common)
    assert len(common) == 16, f"expected 16 common trajectories, got {len(common)}"

    def arrays(metrics: dict[str, dict], arm: str, field: str) -> np.ndarray:
        return np.array([metrics[arm][tid][field] for tid in common])

    point_estimate = {
        "n_trajectories": len(common),
        "reminders_per_traj_common16": float(np.mean([threshold_ymin1_metrics[tid]["reminders"] for tid in common])),
        "late_y": float(np.mean([threshold_ymin1_metrics[tid]["late_y"] for tid in common])),
        "terminal_y": float(np.mean([threshold_ymin1_metrics[tid]["terminal_y"] for tid in common])),
    }

    zero_indep_late = arrays(indep_metrics, "zero_control", "late_y")
    zero_indep_terminal = arrays(indep_metrics, "zero_control", "terminal_y")
    thr_late = np.array([threshold_ymin1_metrics[tid]["late_y"] for tid in common])
    thr_terminal = np.array([threshold_ymin1_metrics[tid]["terminal_y"] for tid in common])

    gate1 = {
        "late_y": paired_bootstrap(thr_late - zero_indep_late),
        "terminal_y": paired_bootstrap(thr_terminal - zero_indep_terminal),
    }

    # Gate 2: threshold_ymin1 vs. equal-cost random mix of zero_control +
    # each fixed arm, at threshold_ymin1's OWN independent-judge spend rate
    # (0.150 over all 40 trajectories per the sbatch run; recomputed here on
    # the common-16 subset for consistency with T0's convention).
    spend_prob = float(np.mean([threshold_ymin1_metrics[tid]["reminders"] for tid in common]))
    gate2 = {}
    for arm in ARM_SOURCES:
        if not arm.startswith("fixed_t"):
            continue
        fixed_indep = arrays(indep_metrics, arm, "late_y")
        gate2[arm] = mixed_baseline_bootstrap(thr_late, zero_indep_late, fixed_indep, spend_prob)

    report = {
        "arm": "threshold_ymin1",
        "source": THRESHOLD_YMIN1_PATH,
        "n_rows": 200,
        "n_trajectories_all": 40,
        "point_estimate_on_common16": point_estimate,
        "spend_prob_used_for_gate2": spend_prob,
        "gate1_vs_zero_control_indep": gate1,
        "gate2_vs_equal_cost_random_mix_indep": gate2,
    }

    print("=== T1a: threshold_ymin1 (independent judge) vs. zero_control (independent judge) ===")
    print(f"n_trajectories (all)={40}  reminders/traj (all 40)=0.150")
    print(f"n_trajectories (common16)={len(common)}  reminders/traj (common16)={spend_prob:.4f}")
    print(f"late_y={point_estimate['late_y']:.4f}  terminal_y={point_estimate['terminal_y']:.4f}")
    print(f"zero_control_indep late_y={zero_indep_late.mean():.4f}  terminal_y={zero_indep_terminal.mean():.4f}")
    print("\ngate 1 (threshold_ymin1 vs. zero_control, independent judge):")
    for key, stats_ in gate1.items():
        passed = stats_["ci_low"] > 0 or stats_["ci_high"] < 0
        print(
            f"  {key:<12} mean_diff={stats_['mean_diff']:+.4f} 95% CI [{stats_['ci_low']:+.4f}, {stats_['ci_high']:+.4f}]"
            f" n={stats_['n_pairs']} {'PASS' if passed else 'n.s.'}"
        )
    print("\ngate 2 (threshold_ymin1 vs. equal-cost random mix, independent judge):")
    for key, stats_ in gate2.items():
        passed = stats_["ci_low"] > 0 or stats_["ci_high"] < 0
        print(
            f"  {key:<12} mean_diff={stats_['mean_diff']:+.4f} 95% CI [{stats_['ci_low']:+.4f}, {stats_['ci_high']:+.4f}]"
            f" (spend_prob={stats_['spend_prob']:.4f}) {'PASS' if passed else 'n.s.'}"
        )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nreport written to {OUT_PATH}")


if __name__ == "__main__":
    main()
