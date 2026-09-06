"""Minimal-pilot analysis for ergo_math_trajectory.py's rows
(docs/feasibility/ERGO_MULTITURN_RELIABILITY_FEASIBILITY.md's step 2: does
the reset actuator move anything at all, before any further engineering).

Deliberately lean compared to analysis_sycophancy.py/analysis_adversarial.py
-- no discrete-flip machinery, no per-category breakdown (single task, no
categories), no `pass`-gated hypothesis tests. The one statistical test here
(`new_q1_escalation`) is, structurally, a **third** near-identical copy of
the OLS-slope-per-trajectory + one-sample-t-test-vs-0 pattern
analysis_adversarial.py and analysis_sycophancy.py already each have their
own instance of -- analysis_sycophancy.py's docstring already flagged "one
short of the rule-of-three bar" for this exact thing. Not extracted into a
shared helper here on purpose: doing that refactor is a separate, deliberate
decision (touches two already-frozen analysis modules), out of scope for a
minimal authority-check pilot; noting the debt rather than silently
incurring a third copy.

`final_turn_success` (not `success_by_turn`) is the primary number for the
authority check: items have variable shard counts (4-12, resources/
PROVENANCE.md), so "the turn all information has been revealed" is
per-trajectory `num_shards`, not a fixed cross-item turn index the way
sycophancy's turns-2-5 window was.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


def analyze_ergo_math_screening(rows: list[dict[str, Any]]) -> dict[str, Any]:
    df = pd.DataFrame(rows)
    result: dict[str, Any] = {}

    # Primary metric: per-trajectory score at its own final turn (turn == num_shards).
    final_rows = df[df["turn"] == df["num_shards"]]
    result["final_turn_success"] = {
        "mean": float(final_rows["y_task_success"].mean()),
        "n": int(len(final_rows)),
        "per_trajectory": dict(zip(final_rows["trajectory_id"], final_rows["y_task_success"])),
        "per_item": dict(zip(final_rows["item_id"], final_rows["y_task_success"])),
    }

    # Descriptive: mean score by absolute turn index (only meaningful up to
    # the shortest item's shard count without over-interpreting -- reported
    # as-is, n shrinks at higher turns since not every item has that many shards).
    by_turn: dict[int, dict[str, float]] = {}
    for turn, g in df.groupby("turn"):
        by_turn[int(turn)] = {"mean": float(g["y_task_success"].mean()), "n": int(len(g))}
    result["success_by_turn"] = by_turn

    # new-Q1 analog: per-trajectory OLS slope of y_task_success vs turn (own
    # turn range, not a fixed T), aggregated per item (mean across seeds),
    # one-sample t-test vs 0 -- same recipe as analysis_sycophancy.new_q1_escalation.
    per_trajectory_slope: dict[str, float] = {}
    item_by_trajectory: dict[str, str] = {}
    for tid, g in df.groupby("trajectory_id"):
        g = g.sort_values("turn")
        item_by_trajectory[tid] = g.iloc[0]["item_id"]
        if g["turn"].nunique() >= 3:
            slope, _intercept, _r, _p, _stderr = stats.linregress(g["turn"], g["y_task_success"])
            per_trajectory_slope[tid] = float(slope)
    per_item_slopes: dict[str, list[float]] = {}
    for tid, slope in per_trajectory_slope.items():
        per_item_slopes.setdefault(item_by_trajectory[tid], []).append(slope)
    per_item_mean_slope = {iid: float(np.mean(vals)) for iid, vals in per_item_slopes.items()}
    item_slopes = list(per_item_mean_slope.values())
    if len(item_slopes) >= 2:
        t_stat, t_p = stats.ttest_1samp(item_slopes, 0.0)
    else:
        t_stat, t_p = float("nan"), float("nan")
    result["new_q1_escalation"] = {
        "per_item_mean_slope": per_item_mean_slope,
        "n_items": len(item_slopes),
        "n_positive_slope_items": sum(1 for s in item_slopes if s > 0),
        "n_negative_slope_items": sum(1 for s in item_slopes if s < 0),
        "t_test_mean_slope_vs_zero": {"t": float(t_stat), "p": float(t_p), "df": len(item_slopes) - 1},
    }

    # new-Q3 analog: lag-1 autocorrelation of y_task_success within trajectory.
    # Same guard as analysis_sycophancy.py's new_q3_autocorrelation: linregress
    # raises if the x values (y_t here) are all identical, which a
    # constant-answer agent (e.g. this module's own orchestration tests'
    # FakeChatModel) triggers.
    lag_pairs = []
    for _tid, g in df.groupby("trajectory_id"):
        g = g.sort_values("turn")
        y = g["y_task_success"].to_numpy()
        lag_pairs.extend(zip(y[:-1], y[1:]))
    y_t = np.array([p[0] for p in lag_pairs])
    y_t1 = np.array([p[1] for p in lag_pairs])
    if len(lag_pairs) >= 3 and len(set(y_t.tolist())) > 1:
        slope, _intercept, r, p_value, _stderr = stats.linregress(y_t, y_t1)
    else:
        slope, r, p_value = float("nan"), float("nan"), float("nan")
    result["new_q3_autocorrelation"] = {
        "slope": float(slope),
        "r": float(r),
        "p_value": float(p_value),
        "n_pairs": len(lag_pairs),
    }

    result["diagnostics"] = {
        "refusal_rate": float(df["refusal_flag"].mean()),
        "judge_parse_failure_rate": float(df["judge_parse_failure"].mean()),
    }
    return result
