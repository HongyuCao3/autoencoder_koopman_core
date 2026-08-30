"""Analysis functions for the free surface features (surface_features.py)
applied to an already-collected trajectories.jsonl -- the same two gate-
style tests as analysis.py's q1_drift_trend / q2_input_effective /
q3_inertia, but run per surface feature instead of y_probe. See
docs/experiments/surface_features_backfill.md for the motivation and
results this was written to produce.

Kept separate from analysis.py (which is specifically "the three protocol
gate questions on y_probe") because these operate on a different, optional
family of readouts and are meant to be run as an offline backfill, not as
part of the standard screening pipeline.
"""

from __future__ import annotations

import json
import pathlib

import pandas as pd
from scipy import stats

from .surface_features import extract_surface_features


def backfill_dataframe(trajectories_path: pathlib.Path) -> pd.DataFrame:
    """Read a trajectories.jsonl and append one column per
    surface_features.SURFACE_FEATURE_NAMES, computed from `agent_message`.
    Read-only on the input file."""
    rows = [json.loads(line) for line in pathlib.Path(trajectories_path).read_text().splitlines() if line.strip()]
    df = pd.DataFrame(rows)
    feature_rows = df["agent_message"].fillna("").map(extract_surface_features)
    feature_df = pd.DataFrame(list(feature_rows))
    return pd.concat([df.reset_index(drop=True), feature_df], axis=1)


def analyze_zero_control_drift(df: pd.DataFrame, feature: str) -> dict:
    """Same shape/logic as analysis.py's q1_drift_trend: per-trajectory OLS
    slope over zero_control turns, aggregated to one value per prompt (its
    seeds aren't independent replicates), one-sample t-test of prompt-level
    slopes against zero. Plus the simpler turn1-vs-last mean drop for an
    at-a-glance number."""
    zero = df[df["excitation_design"] == "zero_control"]
    drops: list[float] = []
    slopes: dict[str, float] = {}
    prompt_of: dict[str, str] = {}
    for trajectory_id, g in zero.groupby("trajectory_id"):
        g = g.sort_values("turn")
        vals = g[feature]
        if vals.isna().any():
            continue
        drops.append(float(vals.iloc[0] - vals.iloc[-1]))
        prompt_of[trajectory_id] = g.iloc[0]["system_prompt_id"]
        if g["turn"].nunique() >= 2:
            slope, _intercept, _r, _p, _stderr = stats.linregress(g["turn"], vals)
            slopes[trajectory_id] = float(slope)

    per_prompt_slopes: dict[str, list[float]] = {}
    for trajectory_id, slope in slopes.items():
        per_prompt_slopes.setdefault(prompt_of[trajectory_id], []).append(slope)
    prompt_mean_slopes = [sum(v) / len(v) for v in per_prompt_slopes.values()]

    if len(prompt_mean_slopes) >= 2:
        t_stat, t_p = stats.ttest_1samp(prompt_mean_slopes, 0.0)
        t_stat, t_p = float(t_stat), float(t_p)
    else:
        t_stat, t_p = float("nan"), float("nan")

    return {
        "feature": feature,
        "mean_drop_turn1_to_last": sum(drops) / len(drops) if drops else float("nan"),
        "n_trajectories": len(drops),
        "n_prompts": len(prompt_mean_slopes),
        "n_negative_slope_prompts": sum(1 for s in prompt_mean_slopes if s < 0),
        "n_positive_slope_prompts": sum(1 for s in prompt_mean_slopes if s > 0),
        "t_test_mean_slope_vs_zero": {"t": t_stat, "p": t_p, "df": len(prompt_mean_slopes) - 1},
        "raw_sd_across_all_zero_control_rows": float(zero[feature].std()) if not zero.empty else float("nan"),
    }


def analyze_input_effect(df: pd.DataFrame, feature: str) -> dict:
    """Same method as analysis.py's q2_input_effective + q3_inertia, run on
    `feature` instead of y_probe, restricted to excite_iid ("iid")
    trajectories: does u_remind_t predict feature_{t+1} (Q2-style) or
    feature_{t+2} (Q3-style, inertia)?"""
    excite = df[df["excitation_design"] == "iid"]
    next_pairs, lag2_pairs = [], []
    for _trajectory_id, g in excite.groupby("trajectory_id"):
        g = g.sort_values("turn").reset_index(drop=True)
        for i in range(len(g) - 1):
            next_pairs.append((g.loc[i, "u_remind"], g.loc[i + 1, feature]))
        for i in range(len(g) - 2):
            lag2_pairs.append((g.loc[i, "u_remind"], g.loc[i + 2, feature]))

    next_df = pd.DataFrame(next_pairs, columns=["u_remind_t", "feature_next"]).dropna()
    mean_next_1 = next_df.loc[next_df.u_remind_t == 1, "feature_next"].mean()
    mean_next_0 = next_df.loc[next_df.u_remind_t == 0, "feature_next"].mean()
    q2_diff = (
        float(mean_next_1 - mean_next_0)
        if pd.notna(mean_next_1) and pd.notna(mean_next_0)
        else float("nan")
    )
    u_group = next_df.loc[next_df.u_remind_t == 1, "feature_next"]
    no_u_group = next_df.loc[next_df.u_remind_t == 0, "feature_next"]
    if len(u_group) >= 2 and len(no_u_group) >= 2:
        t_stat, t_p = stats.ttest_ind(u_group, no_u_group, equal_var=False)
        t_stat, t_p = float(t_stat), float(t_p)
    else:
        t_stat, t_p = float("nan"), float("nan")

    lag2_df = pd.DataFrame(lag2_pairs, columns=["u_remind_t", "feature_lag2"]).dropna()
    if lag2_df["u_remind_t"].nunique() > 1 and lag2_df.shape[0] >= 3:
        slope, _intercept, r, p, _stderr = stats.linregress(lag2_df["u_remind_t"], lag2_df["feature_lag2"])
        slope, r, p = float(slope), float(r), float(p)
    else:
        slope = r = p = float("nan")

    return {
        "feature": feature,
        "q2_next_turn": {
            "mean_given_u1": float(mean_next_1) if pd.notna(mean_next_1) else float("nan"),
            "mean_given_u0": float(mean_next_0) if pd.notna(mean_next_0) else float("nan"),
            "diff": q2_diff,
            "welch_t": t_stat,
            "p": t_p,
            "n_pairs": int(next_df.shape[0]),
        },
        "q3_lag2_inertia": {
            "slope_u_on_feature_lag2": slope,
            "r": r,
            "p": p,
            "n_pairs": int(lag2_df.shape[0]),
        },
    }


def render_drift_markdown(report: dict[str, dict]) -> str:
    lines = [
        "# Surface-feature drift backfill report",
        "",
        "Backfilled onto already-generated `agent_message` text from a completed run "
        "(no new GPU generation). Each row below re-runs the zero_control drift test "
        "(same method as analysis.py's q1_drift_trend) on one free surface feature "
        "instead of y_probe.",
        "",
        "| feature | mean drop (turn1->last) | n_prompts | neg/pos slope prompts | slope-vs-0 p | significant? |",
        "|---|---:|---:|---|---:|---|",
    ]
    for feature, r in report.items():
        p = r["t_test_mean_slope_vs_zero"]["p"]
        sig = "yes (p<0.05)" if isinstance(p, float) and p == p and p < 0.05 else "no"
        lines.append(
            f"| {feature} | {r['mean_drop_turn1_to_last']:.4f} | {r['n_prompts']} | "
            f"{r['n_negative_slope_prompts']}/{r['n_positive_slope_prompts']} | {p} | {sig} |"
        )
    return "\n".join(lines) + "\n"


def render_input_effect_markdown(report: dict[str, dict]) -> str:
    lines = [
        "# Surface-feature input-effect (Q2/Q3-style) report",
        "",
        "Does `u_remind` predict next-turn (Q2-style) or two-turns-later (Q3-style, inertia) "
        "values of each free surface feature, on the excite_iid trajectories? Read from the "
        "already-backfilled trajectories_with_surface_features.jsonl -- no new generation.",
        "",
        "| feature | Q2 diff (u=1 - u=0) | Q2 p (Welch t) | Q3 slope | Q3 p | n(Q2)/n(Q3) |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for feature, r in report.items():
        q2, q3 = r["q2_next_turn"], r["q3_lag2_inertia"]
        lines.append(
            f"| {feature} | {q2['diff']:.4f} | {q2['p']} | {q3['slope_u_on_feature_lag2']:.4f} | "
            f"{q3['p']} | {q2['n_pairs']}/{q3['n_pairs']} |"
        )
    return "\n".join(lines) + "\n"
