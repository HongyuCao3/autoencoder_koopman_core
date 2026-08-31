"""New gate question for the escalating persona-pressure confirmation pilot
(docs/experiments/drift_confirmation_pilot.md, "下一步的小范围实验建议"): does
Qwen3-4B show measurable persona/style drift when the simulated user actively
and progressively pressures it to deviate, as opposed to the flat null under
passive self-chat (zero_control) already established by
signal_screening_pilot.md and drift_confirmation_pilot.md?

Reuses analysis.py's q1_drift_trend pattern (per-trajectory full-sequence OLS
slope of y_probe vs turn, aggregated to one value per prompt across its
seeds, one-sample t-test vs 0) but splits rows by `user_mode` instead of
`excitation_design`: both conditions in this pilot use ZeroControlController
(u_remind == 0 throughout -- the pressure comes from the simulated user's
script, channel A plays no role here), so `excitation_design` alone can't
tell the baseline and pressure conditions apart. `user_mode` ("live" vs
"pressure") can.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from scipy import stats


def _slope_test(df: pd.DataFrame) -> dict[str, Any]:
    per_trajectory_slope: dict[str, float] = {}
    prompt_id_by_trajectory: dict[str, str] = {}
    for tid, g in df.groupby("trajectory_id"):
        g = g.sort_values("turn")
        prompt_id_by_trajectory[tid] = g.iloc[0]["system_prompt_id"]
        if g["turn"].nunique() >= 2 and g["y_probe"].notna().all():
            slope, _intercept, _r, _p, _stderr = stats.linregress(g["turn"], g["y_probe"])
            per_trajectory_slope[tid] = float(slope)

    per_prompt_slopes: dict[str, list[float]] = {}
    for tid, slope in per_trajectory_slope.items():
        per_prompt_slopes.setdefault(prompt_id_by_trajectory[tid], []).append(slope)
    per_prompt_mean_slope = {pid: sum(vals) / len(vals) for pid, vals in per_prompt_slopes.items()}

    prompt_slopes = list(per_prompt_mean_slope.values())
    if len(prompt_slopes) >= 2:
        t_stat, t_p = stats.ttest_1samp(prompt_slopes, 0.0)
    else:
        t_stat, t_p = float("nan"), float("nan")
    return {
        "per_trajectory_slope": per_trajectory_slope,
        "per_prompt_mean_slope": per_prompt_mean_slope,
        "n_prompts": len(prompt_slopes),
        "n_negative_slope_prompts": sum(1 for s in prompt_slopes if s < 0),
        "n_positive_slope_prompts": sum(1 for s in prompt_slopes if s > 0),
        "t_test_mean_slope_vs_zero": {
            "t": float(t_stat),
            "p": float(t_p),
            "df": len(prompt_slopes) - 1,
        },
        # negative slope = y_probe declining = drifting away from the
        # persona/style instruction, the direction escalating pressure
        # predicts (and the opposite of what zero_control/live is expected
        # to show, per the prior pilots' clean null).
        "pass": bool(len(prompt_slopes) >= 2 and t_p == t_p and t_p < 0.05 and t_stat < 0),
    }


def analyze_pressure_screening(rows: list[dict[str, Any]]) -> dict[str, Any]:
    df = pd.DataFrame(rows)
    result: dict[str, Any] = {}

    baseline = df[df["user_mode"] == "live"] if not df.empty else df
    pressure = df[df["user_mode"] == "pressure"] if not df.empty else df

    result["q1_baseline_no_pressure"] = _slope_test(baseline)
    result["q1_escalating_pressure"] = _slope_test(pressure)

    result["diagnostics"] = {
        "refusal_rate": float(df["refusal_flag"].mean()) if not df.empty else float("nan"),
        "scorer_failure_rate": float(df["parse_failure"].mean()) if not df.empty else float("nan"),
        "y_probe_by_condition_and_category": {
            f"{mode}/{category}": {"mean": float(g["y_probe"].mean()), "sd": float(g["y_probe"].std())}
            for (mode, category), g in df.groupby(["user_mode", "prompt_category"])
        }
        if not df.empty
        else {},
    }
    return result
