#!/usr/bin/env python3
"""Detection-design option 1 (docs/experiments/koopman_detection_design.md):
one-step-ahead prediction residual ("innovation") from the already-fit
Phase C/D koopman surrogate, computed on Phase E's four closed-loop arms.

Not a new fit: loads the existing richer_abs_sign (nu=1, mu=2) model from
outputs/koopman_defense_phaseB_random_excite/koopman_fit_report.json and
replays it turn-by-turn over Phase E's held-out trajectories (attacks that
were never in the Phase B/C fitting data), always re-grounding on the true
observed z_t before predicting y_(t+1) -- one-step error, not free rollout.

CPU-only, pure numpy/pandas -- no GPU needed. Run directly (no sbatch).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from persona_drift.modeling.dataset import (  # noqa: E402
    ReducedStateConfig,
    build_reduced_state_pairs,
    group_by_trajectory,
    load_trajectories,
)
from persona_drift.modeling.koopman import (  # noqa: E402
    abs_sign_extra_features,
    surrogate_from_arrays,
)

ARMS = ["zero_control", "constant_remind", "threshold", "koopman_mpc"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fit-report",
        type=pathlib.Path,
        default=pathlib.Path("outputs/koopman_defense_phaseB_random_excite/koopman_fit_report.json"),
    )
    parser.add_argument("--model-key", default="richer_abs_sign")
    parser.add_argument(
        "--phase-e-dir-template",
        default="outputs/koopman_defense_phaseE_{arm}/trajectories.jsonl",
    )
    parser.add_argument(
        "--out-dir", type=pathlib.Path, default=pathlib.Path("outputs/koopman_detection_innovation")
    )
    parser.add_argument("--erosion-threshold", type=float, default=0.7, help="matches ThresholdController y_min")
    return parser.parse_args()


def load_model(fit_report_path: pathlib.Path, model_key: str):
    report = json.loads(fit_report_path.read_text())
    cfg = report["config"]
    model_report = report[model_key]
    extra_features_fn = abs_sign_extra_features if model_key == "richer_abs_sign" else None
    if extra_features_fn is None:
        raise ValueError(f"unsupported model_key {model_key!r} for this script")
    config = ReducedStateConfig(nu=cfg["nu"], mu=cfg["mu"])
    model = surrogate_from_arrays(
        A=model_report["A"],
        B=model_report["B"],
        b=model_report["b"],
        C=model_report["C"],
        state_dim=config.state_dim,
        extra_features_fn=extra_features_fn,
        ridge=cfg["ridge"],
    )
    return model, config


def per_turn_residuals(model, config, rows, arm_name):
    records = []
    for traj_id, traj_rows in group_by_trajectory(rows, id_col="trajectory_id").items():
        pairs = build_reduced_state_pairs(traj_rows, config, y_col="y_safety", u_col="u_remind")
        # pairs[k] predicts the turn at (start + k + 1); recover that row's
        # metadata by walking traj_rows in lockstep with the same skip rule
        # build_reduced_state_pairs uses (NaN turns dropped from every pair
        # that would include them).
        start = max(config.nu - 1, config.mu)
        valid_rows = [r for r in traj_rows if r["y_safety"] == r["y_safety"]]  # drop NaN
        # valid_rows is only a correct index map when no NaNs exist inside
        # the window; assert that here rather than silently misaligning.
        if len(valid_rows) != len(traj_rows):
            raise RuntimeError(f"{traj_id}: NaN y_safety present, index alignment unsupported by this script")
        for k, pair in enumerate(pairs):
            t = start + k + 1
            predicted_next = model.readout(model.step(pair["z"], pair["v"]))
            true_next = pair["z_next"][config.nu - 1]
            row_t = traj_rows[t]
            row_prev = traj_rows[t - 1]
            records.append(
                {
                    "arm": arm_name,
                    "trajectory_id": traj_id,
                    "attack_id": row_t["attack_id"],
                    "turn": row_t["turn"],
                    "u_remind_prev": row_prev["u_remind"],
                    "y_prev": row_prev["y_safety"],
                    "y_true": true_next,
                    "y_pred": predicted_next,
                    "residual": true_next - predicted_next,
                    "actual_drop": true_next - row_prev["y_safety"],
                    "erosion_event": row_prev["y_safety"] >= 0.7 and true_next < 0.7,
                }
            )
    return records


def main() -> None:
    args = parse_args()
    model, config = load_model(args.fit_report, args.model_key)

    all_records = []
    for arm in ARMS:
        path = pathlib.Path(args.phase_e_dir_template.format(arm=arm))
        rows = load_trajectories(path)
        all_records.extend(per_turn_residuals(model, config, rows, arm))

    df = pd.DataFrame.from_records(all_records)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_dir / "innovation_residuals.csv", index=False)

    summary = (
        df.groupby("arm")["residual"]
        .agg(n="count", mean="mean", std="std", mean_abs=lambda s: s.abs().mean())
        .reindex(ARMS)
    )
    persistence_error = (df["y_true"] - df["y_prev"]).abs()
    summary["persistence_mae"] = persistence_error.groupby(df["arm"]).mean().reindex(ARMS)

    erosion = df[df["erosion_event"]]
    non_erosion = df[~df["erosion_event"]]
    corr = np.corrcoef(df["residual"], df["actual_drop"])[0, 1] if len(df) > 1 else float("nan")

    report = {
        "model_key": args.model_key,
        "config": {"nu": config.nu, "mu": config.mu},
        "n_total_pairs": len(df),
        "summary_by_arm": summary.reset_index().to_dict(orient="records"),
        "residual_vs_actual_drop_corr": float(corr),
        "erosion_event_count": int(len(erosion)),
        "erosion_event_mean_abs_residual": float(erosion["residual"].abs().mean()) if len(erosion) else float("nan"),
        "non_erosion_mean_abs_residual": float(non_erosion["residual"].abs().mean()) if len(non_erosion) else float("nan"),
    }
    (args.out_dir / "innovation_summary.json").write_text(json.dumps(report, indent=2))

    print(summary.to_string())
    print()
    print(f"residual vs actual_drop correlation: {corr:.4f}")
    print(f"erosion events: {len(erosion)}/{len(df)}")
    print(f"mean |residual| at erosion events: {report['erosion_event_mean_abs_residual']:.4f}")
    print(f"mean |residual| elsewhere:         {report['non_erosion_mean_abs_residual']:.4f}")
    print(f"report written to {args.out_dir / 'innovation_summary.json'}")


if __name__ == "__main__":
    main()
