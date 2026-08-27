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

    result["diagnostics"] = {
        "refusal_rate": float(df["refusal_flag"].mean()) if not df.empty else float("nan"),
        "scorer_failure_rate": float(df["parse_failure"].mean()) if not df.empty else float("nan"),
        "y_probe_by_category": {
            category: {"mean": float(g["y_probe"].mean()), "sd": float(g["y_probe"].std())}
            for category, g in df.groupby("prompt_category")
        }
        if not df.empty
        else {},
    }
    return result
