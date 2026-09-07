#!/usr/bin/env python3
"""T2 follow-up (docs/experiments/adaptive_vs_fixed_claim_plan.md section 四):
recomputes the two claim gates with `randsched_p100`/`randsched_p75` (the
equal-cost random-allocation baseline) added, WITHOUT modifying
`scripts/analyze_readout_state.py` or overwriting its
`outputs/koopman_case_study/readout_state_report.json` (T0's committed
artifact -- this script only imports its helpers).

Both new arms only have self-judged scores (section 4.5: no independent-judge
version for this task), so everything here is self-judge (自评) only -- the
same footing as the other 7 Phase J arms T0 already reports self-judge
numbers for.

Gate 1 (beats doing nothing): each new arm vs. `zero_control`, self-judge.
Gate 2 (beats equal-cost random allocation): `koopman_b1` vs. `randsched_p75`
(the arm that actually matches koopman's own self-judged spend rate,
30/40 = 0.750) and, for completeness, vs. `randsched_p100`. This SUPERSEDES
T0's `mixed_baseline_bootstrap` construction for gate 2 -- that function
built a resampled stand-in baseline because no real random-schedule arm
existed yet; now that one has actually been run, the direct paired
comparison is the primary number and the mixed-baseline numbers in T0's
report stay as a historical cross-check, not the new gate.

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
    paired_bootstrap,
    per_trajectory_arm_metrics,
)

NEW_ARM_PATHS = {
    "randsched_p100": "outputs/koopman_defense_phaseJ_budget1_randsched_p100/trajectories.jsonl",
    "randsched_p75": "outputs/koopman_defense_phaseJ_budget1_randsched_p75/trajectories.jsonl",
}
PREREGISTERED_RANGE = (-0.03, 0.08)  # plan section 4.7, koopman_b1 vs. randsched_p75, late_y
OUT_PATH = pathlib.Path("outputs/koopman_case_study/readout_state_report_t2.json")


def main() -> None:
    if OUT_PATH.exists():
        raise SystemExit(f"refusing to overwrite existing {OUT_PATH}")

    # Same 8-arm self-judge metrics T0 computes (paths[0] of each ARM_SOURCES
    # entry), plus the two new arms' self-judge-only metrics.
    self_metrics = {arm: per_trajectory_arm_metrics(paths[0]) for arm, paths in ARM_SOURCES.items()}
    new_metrics = {arm: per_trajectory_arm_metrics(path) for arm, path in NEW_ARM_PATHS.items()}

    common = set.intersection(*(set(m) for m in self_metrics.values()))
    common &= set.intersection(*(set(m) for m in new_metrics.values()))
    common = sorted(common)
    assert len(common) == 16, f"expected 16 common trajectories, got {len(common)}"

    def arr(metrics: dict[str, dict], arm: str, field: str) -> np.ndarray:
        return np.array([metrics[arm][tid][field] for tid in common])

    point_estimate = {
        arm: {
            "n_trajectories": len(common),
            "reminders_per_traj_common16": float(np.mean([m[tid]["reminders"] for tid in common])),
            "late_y": float(np.mean([m[tid]["late_y"] for tid in common])),
            "terminal_y": float(np.mean([m[tid]["terminal_y"] for tid in common])),
        }
        for arm, m in new_metrics.items()
    }

    zero_late = arr(self_metrics, "zero_control", "late_y")
    zero_terminal = arr(self_metrics, "zero_control", "terminal_y")
    koopman_late = arr(self_metrics, "koopman_b1", "late_y")
    koopman_terminal = arr(self_metrics, "koopman_b1", "terminal_y")

    gate1 = {}
    for arm, m in new_metrics.items():
        late = arr(new_metrics, arm, "late_y")
        terminal = arr(new_metrics, arm, "terminal_y")
        gate1[arm] = {
            "late_y": paired_bootstrap(late - zero_late),
            "terminal_y": paired_bootstrap(terminal - zero_terminal),
        }

    gate2 = {}
    for arm, m in new_metrics.items():
        late = arr(new_metrics, arm, "late_y")
        terminal = arr(new_metrics, arm, "terminal_y")
        gate2[arm] = {
            "late_y": paired_bootstrap(koopman_late - late),
            "terminal_y": paired_bootstrap(koopman_terminal - terminal),
        }

    g2_p75_late = gate2["randsched_p75"]["late_y"]
    lo, hi = PREREGISTERED_RANGE
    in_range = lo <= g2_p75_late["mean_diff"] <= hi
    ns = g2_p75_late["ci_low"] < 0 < g2_p75_late["ci_high"]
    preregistered_check = {
        "basis": "koopman_b1 - randsched_p75, late_y, self-judge, common16",
        "mean_diff": g2_p75_late["mean_diff"],
        "ci_low": g2_p75_late["ci_low"],
        "ci_high": g2_p75_late["ci_high"],
        "preregistered_range": list(PREREGISTERED_RANGE),
        "in_preregistered_range": in_range,
        "ci_crosses_zero": ns,
        "matches_preregistered_expectation": in_range and ns,
    }

    report = {
        "new_arms": list(NEW_ARM_PATHS),
        "n_rows_per_arm": 200,
        "n_trajectories_all": 40,
        "point_estimate_on_common16": point_estimate,
        "gate1_vs_zero_control_self": gate1,
        "gate2_vs_koopman_b1_self": gate2,
        "preregistered_check": preregistered_check,
    }

    print("=== T2: randsched_p100 / randsched_p75 (self-judge) vs. zero_control / koopman_b1 (self-judge) ===")
    for arm in NEW_ARM_PATHS:
        pe = point_estimate[arm]
        reminders_all40 = float(np.mean([r["reminders"] for r in new_metrics[arm].values()]))
        print(
            f"{arm:<16} n_traj(all)=40 reminders/traj(all40)={reminders_all40:.4f} "
            f"reminders/traj(common16)={pe['reminders_per_traj_common16']:.4f} late_y={pe['late_y']:.4f} terminal_y={pe['terminal_y']:.4f}"
        )
    print(f"zero_control(self) late_y={zero_late.mean():.4f} terminal_y={zero_terminal.mean():.4f}")
    print(f"koopman_b1(self)   late_y={koopman_late.mean():.4f} terminal_y={koopman_terminal.mean():.4f}")

    print("\ngate 1 (new arm vs. zero_control, self-judge):")
    for arm, cols in gate1.items():
        for key, stats_ in cols.items():
            passed = stats_["ci_low"] > 0 or stats_["ci_high"] < 0
            print(
                f"  {arm:<16}{key:<12} mean_diff={stats_['mean_diff']:+.4f} 95% CI [{stats_['ci_low']:+.4f}, {stats_['ci_high']:+.4f}]"
                f" n={stats_['n_pairs']} {'PASS' if passed else 'n.s.'}"
            )

    print("\ngate 2 (koopman_b1 vs. new arm, self-judge -- direct real-arm comparison, supersedes mixed baseline):")
    for arm, cols in gate2.items():
        for key, stats_ in cols.items():
            passed = stats_["ci_low"] > 0 or stats_["ci_high"] < 0
            print(
                f"  {arm:<16}{key:<12} mean_diff={stats_['mean_diff']:+.4f} 95% CI [{stats_['ci_low']:+.4f}, {stats_['ci_high']:+.4f}]"
                f" n={stats_['n_pairs']} {'PASS' if passed else 'n.s.'}"
            )

    print(
        f"\npreregistered check (koopman_b1 - randsched_p75, late_y): mean_diff={g2_p75_late['mean_diff']:+.4f} "
        f"95% CI [{g2_p75_late['ci_low']:+.4f}, {g2_p75_late['ci_high']:+.4f}] "
        f"range={PREREGISTERED_RANGE} in_range={in_range} ci_crosses_zero={ns} "
        f"-> {'MATCHES' if preregistered_check['matches_preregistered_expectation'] else 'DEVIATES FROM'} preregistered expectation"
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nreport written to {OUT_PATH}")


if __name__ == "__main__":
    main()
