"""Backfill free surface features (persona_drift.surface_features) onto an
ALREADY-COMPLETED trajectories.jsonl and re-run the zero_control Q1-style
drift test (same pattern as analysis.py's q1_drift_trend) on each new
feature instead of y_probe. Read-only on the input file -- writes results
to new files alongside it. No GPU, no new model generation.

Motivation (docs/OPEN_DATASETS_AND_TRAJECTORY_ACCELERATION.md,
docs/ALHAFNI_LINGUISTIC_CONTROL_FEASIBILITY.md): switching y_t to a free
surface feature was flagged as a real risk, not a free win -- a different
construct than y_probe might show the same "no detectable drift" result
that signal_screening_pilot found, or it might not. This checks that on
the text signal_screening_pilot already generated, with zero additional
GPU time, before spending any more GPU budget on the question.

Usage:
    python scripts/backfill_surface_features.py \
        --trajectories-path outputs/signal_screening/trajectories.jsonl \
        --output-dir outputs/signal_screening
"""

from __future__ import annotations

import argparse
import json
import pathlib

import pandas as pd
from scipy import stats

from persona_drift.surface_features import SURFACE_FEATURE_NAMES, extract_surface_features


def backfill(trajectories_path: pathlib.Path) -> pd.DataFrame:
    rows = [json.loads(line) for line in trajectories_path.read_text().splitlines() if line.strip()]
    df = pd.DataFrame(rows)
    feature_rows = df["agent_message"].fillna("").map(extract_surface_features)
    feature_df = pd.DataFrame(list(feature_rows))
    return pd.concat([df.reset_index(drop=True), feature_df], axis=1)


def analyze_feature_drift(df: pd.DataFrame, feature: str) -> dict:
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


def render_markdown(report: dict[str, dict]) -> str:
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--trajectories-path",
        type=pathlib.Path,
        default=pathlib.Path("outputs/signal_screening/trajectories.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("outputs/signal_screening"),
    )
    args = parser.parse_args()

    df = backfill(args.trajectories_path)

    augmented_path = args.output_dir / "trajectories_with_surface_features.jsonl"
    df.to_json(augmented_path, orient="records", lines=True)

    report = {feature: analyze_feature_drift(df, feature) for feature in SURFACE_FEATURE_NAMES}
    report_path = args.output_dir / "surface_features_drift_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    (args.output_dir / "surface_features_drift_report.md").write_text(render_markdown(report))

    print(f"wrote {augmented_path}")
    print(f"wrote {report_path}")
    print(f"wrote {args.output_dir / 'surface_features_drift_report.md'}")


if __name__ == "__main__":
    main()
