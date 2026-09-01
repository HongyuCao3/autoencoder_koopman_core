"""Phase F helpfulness-cost readout (docs/experiments/koopman_defense_pilot.md):
per-controller-arm summary of the rows produced by
benign_trajectory.run_benign_trajectory. Not a gate like
analysis_adversarial.analyze_adversarial_screening -- there is no
pass/fail hypothesis here, only a descriptive cost comparison, so this
module has no "pass" field.

analyze_benign_screening summarizes a SINGLE arm's rows (mirrors
analyze_adversarial_screening's shape: one report per controller/output_dir,
written by benign_screening.run_benign_screening). compare_arms_to_zero_control
is a separate, later step that pairs a treatment arm's per-session y_help
against zero_control's on the same (benign_id, seed) session -- content is
identical across arms (only u_remind differs, exactly like Phase E), so this
is a paired comparison, not an independent-samples one.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from scipy import stats

from .group_stats import summary_by_category, summary_by_turn


def analyze_benign_screening(rows: list[dict[str, Any]]) -> dict[str, Any]:
    df = pd.DataFrame(rows)
    if df.empty:
        return {
            "diagnostics": {
                "mean_y_help": float("nan"),
                "sd_y_help": float("nan"),
                "refusal_rate": float("nan"),
                "judge_parse_failure_rate": float("nan"),
                "helpfulness_parse_failure_rate": float("nan"),
                "n_reminders_inserted": 0,
                "n_rows": 0,
                "total_inserted_tokens": 0,
                "y_help_by_category": {},
                "y_help_by_turn": {},
            },
            "per_session_mean_y_help": {},
        }

    result: dict[str, Any] = {
        "diagnostics": {
            "mean_y_help": float(df["y_help"].mean()),
            "sd_y_help": float(df["y_help"].std()),
            "refusal_rate": float(df["refusal_flag"].mean()),
            "judge_parse_failure_rate": float(df["judge_parse_failure"].mean()),
            "helpfulness_parse_failure_rate": float(df["helpfulness_parse_failure"].mean()),
            "n_reminders_inserted": int(df["u_remind"].sum()),
            "n_rows": int(len(df)),
            "total_inserted_tokens": int(df["inserted_tokens"].sum()),
            "y_help_by_category": summary_by_category(df, "y_help"),
            "y_help_by_turn": summary_by_turn(df, "y_help"),
        },
        # trajectory_id encodes (benign_id, seed) -- e.g. "mtbench_coding__seed0"
        # -- so this is already keyed per (benign_id, seed) session, ready for
        # compare_arms_to_zero_control's pairing.
        "per_session_mean_y_help": {
            tid: float(g["y_help"].mean()) for tid, g in df.groupby("trajectory_id")
        },
    }
    return result


def compare_arms_to_zero_control(
    per_session_mean_y_help_by_arm: dict[str, dict[str, float]],
    zero_control_arm: str = "zero_control",
) -> dict[str, Any]:
    """`per_session_mean_y_help_by_arm` is {arm_name: {trajectory_id: mean_y_help}},
    one entry per arm's analyze_benign_screening()["per_session_mean_y_help"].
    Returns, for every arm except zero_control_arm, a paired comparison
    (scipy.stats.ttest_rel) over the trajectory_ids common to both arms --
    positive mean_diff means the treatment arm is MORE helpful than
    zero_control, negative means less."""

    baseline = per_session_mean_y_help_by_arm[zero_control_arm]
    result: dict[str, Any] = {}
    for arm, sessions in per_session_mean_y_help_by_arm.items():
        if arm == zero_control_arm:
            continue
        shared_ids = sorted(set(sessions) & set(baseline))
        treatment_vals = [sessions[tid] for tid in shared_ids]
        baseline_vals = [baseline[tid] for tid in shared_ids]
        if len(shared_ids) >= 2:
            t_stat, p_value = stats.ttest_rel(treatment_vals, baseline_vals)
            mean_diff = float(pd.Series(treatment_vals).sub(pd.Series(baseline_vals)).mean())
        else:
            t_stat, p_value, mean_diff = float("nan"), float("nan"), float("nan")
        result[arm] = {
            "n_paired_sessions": len(shared_ids),
            "mean_diff": mean_diff,
            "t": float(t_stat),
            "p": float(p_value),
        }
    return result
