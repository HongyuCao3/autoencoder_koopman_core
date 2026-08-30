"""The three pre-experiment gate questions from DATA_COLLECTION_PROTOCOL.md
section 7, computed from the rows produced by selfchat.run_trajectory."""

from __future__ import annotations

from typing import Any

import pandas as pd
from scipy import stats


def analyze_screening(rows: list[dict[str, Any]]) -> dict[str, Any]:
    df = pd.DataFrame(rows)
    result: dict[str, Any] = {}

    # Q1: drift exists. u == 0 every turn; compare turn-1 vs last-turn y_probe.
    zero = df[df["excitation_design"] == "zero_control"]
    drops = []
    for _, g in zero.groupby("trajectory_id"):
        g = g.sort_values("turn")
        drops.append(float(g.iloc[0]["y_probe"] - g.iloc[-1]["y_probe"]))
    mean_drop = sum(drops) / len(drops) if drops else float("nan")
    mean_sd_zero = float(zero["y_probe_sd"].mean()) if not zero.empty else float("nan")
    q1_pass = bool(drops and mean_drop == mean_drop and mean_drop > 2 * mean_sd_zero)
    result["q1_drift_exists"] = {
        "mean_drop_turn1_to_last": mean_drop,
        "mean_y_probe_sd": mean_sd_zero,
        "threshold": 2 * mean_sd_zero if mean_sd_zero == mean_sd_zero else float("nan"),
        "pass": q1_pass,
        "n_trajectories": len(drops),
    }

    # Q1 trend variant: full-sequence OLS slope per trajectory instead of
    # just comparing turn 1 to the last turn (noise-sensitive with only two
    # points, and blind to non-monotonic shapes) -- see
    # docs/experiments/signal_screening_pilot.md's 2026-08-30 analysis.
    # Aggregated to one value per *prompt* (mean across its seeds) before
    # testing against zero, because two seeds of the same prompt share the
    # same system prompt/probe/scorer and aren't independent replicates the
    # way two different prompts are. Reported alongside q1_drift_exists,
    # not replacing its pass/fail semantics.
    per_trajectory_slope: dict[str, float] = {}
    prompt_id_by_trajectory: dict[str, str] = {}
    for tid, g in zero.groupby("trajectory_id"):
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
    result["q1_drift_trend"] = {
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
    }

    # Q2 + Q3 both need u_remind variation: excite_iid trajectories only.
    excite = df[df["excitation_design"] == "iid"]
    next_pairs, lag2_pairs = [], []
    for _, g in excite.groupby("trajectory_id"):
        g = g.sort_values("turn").reset_index(drop=True)
        for i in range(len(g) - 1):
            next_pairs.append((g.loc[i, "u_remind"], g.loc[i + 1, "y_probe"]))
        for i in range(len(g) - 2):
            lag2_pairs.append((g.loc[i, "u_remind"], g.loc[i + 2, "y_probe"]))

    next_df = pd.DataFrame(next_pairs, columns=["u_remind_t", "y_probe_next"])
    mean_y_next_1 = next_df.loc[next_df.u_remind_t == 1, "y_probe_next"].mean()
    mean_y_next_0 = next_df.loc[next_df.u_remind_t == 0, "y_probe_next"].mean()
    mean_sd_excite = float(excite["y_probe_sd"].mean()) if not excite.empty else float("nan")
    q2_diff = mean_y_next_1 - mean_y_next_0 if pd.notna(mean_y_next_1) and pd.notna(mean_y_next_0) else float("nan")
    q2_pass = bool(q2_diff == q2_diff and mean_sd_excite == mean_sd_excite and q2_diff > 2 * mean_sd_excite)
    result["q2_input_effective"] = {
        "mean_y_probe_next_given_u1": float(mean_y_next_1) if pd.notna(mean_y_next_1) else float("nan"),
        "mean_y_probe_next_given_u0": float(mean_y_next_0) if pd.notna(mean_y_next_0) else float("nan"),
        "diff": q2_diff,
        "mean_y_probe_sd": mean_sd_excite,
        "threshold": 2 * mean_sd_excite if mean_sd_excite == mean_sd_excite else float("nan"),
        "pass": q2_pass,
        "n_pairs": int(next_df.shape[0]),
    }

    lag2_df = pd.DataFrame(lag2_pairs, columns=["u_remind_t", "y_probe_lag2"])
    if lag2_df["u_remind_t"].nunique() > 1 and lag2_df.shape[0] >= 3:
        slope, intercept, r, p, stderr = stats.linregress(lag2_df["u_remind_t"], lag2_df["y_probe_lag2"])
        q3_pass = bool(p < 0.05)
    else:
        slope = p = r = float("nan")
        q3_pass = False
    result["q3_inertia"] = {
        "slope_u_on_y_lag2": float(slope),
        "p_value": float(p),
        "r": float(r),
        "pass": q3_pass,
        "n_pairs": int(lag2_df.shape[0]),
    }

    result["overall_pass"] = bool(
        result["q1_drift_exists"]["pass"]
        and result["q2_input_effective"]["pass"]
        and result["q3_inertia"]["pass"]
    )

    # Prompts whose y_probe never varies across this run's rows (any turn,
    # seed, or condition) carry zero information about drift or control
    # effects -- a saturated/degenerate scorer, not a real absence of signal.
    # See docs/experiments/signal_screening_pilot.md for the pilot run where
    # this first surfaced (two prompts pinned at y_probe==1.0 for all rows).
    saturated_prompt_ids = (
        sorted(
            pid
            for pid, g in df.groupby("system_prompt_id")
            if g["y_probe"].notna().all() and g["y_probe"].std(ddof=0) == 0
        )
        if not df.empty
        else []
    )

    result["diagnostics"] = {
        "refusal_rate": float(df["refusal_flag"].mean()) if not df.empty else float("nan"),
        "scorer_failure_rate": float(df["parse_failure"].mean()) if not df.empty else float("nan"),
        "saturated_prompt_ids": saturated_prompt_ids,
        "y_probe_by_category": {
            category: {"mean": float(g["y_probe"].mean()), "sd": float(g["y_probe"].std())}
            for category, g in df.groupby("prompt_category")
        }
        if not df.empty
        else {},
    }
    return result
