#!/usr/bin/env python3
"""Detection-design option 3, step 2 (docs/experiments/koopman_detection_design.md):
two-regime residual-comparison detector. Loads the existing "attack regime"
model (Phase C, outputs/koopman_defense_phaseB_random_excite/koopman_fit_report.json)
and the "benign regime" model (scripts/fit_koopman_benign_model.py's output),
then classifies held-out trajectories from both regimes by which model's
one-step-ahead prediction fits the observed y_safety dynamics better.

Evaluation set (no leakage into either model's own fit):
  - attack-labeled: Phase E's four closed-loop arms, all use the same 8
    attacks held out from Phase B/C's identification data.
  - benign-labeled: Phase F's four arms, restricted to the 2 benign
    categories fit_koopman_benign_model.py held out from the benign fit.

Classification rule per trajectory (or per turn): predict "attack" if the
attack model's mean one-step |residual| is lower than the benign model's,
else "benign" -- whichever model explains the observed dynamics better.

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
    split_by_system_prompt_id,
)
from persona_drift.modeling.koopman import (  # noqa: E402
    abs_sign_extra_features,
    no_extra_features,
    surrogate_from_arrays,
)

ARMS = ["zero_control", "constant_remind", "threshold", "koopman_mpc"]
EXTRA_FEATURES_FNS = {"arx": no_extra_features, "richer_abs_sign": abs_sign_extra_features}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--attack-fit-report",
        type=pathlib.Path,
        default=pathlib.Path("outputs/koopman_defense_phaseB_random_excite/koopman_fit_report.json"),
    )
    parser.add_argument("--attack-model-key", default="richer_abs_sign")
    parser.add_argument(
        "--benign-fit-report",
        type=pathlib.Path,
        default=pathlib.Path("outputs/koopman_detection_benign_baseline/koopman_fit_report.json"),
    )
    parser.add_argument("--benign-model-key", default="arx")
    parser.add_argument(
        "--phase-e-dir-template",
        default="outputs/koopman_defense_phaseE_{arm}/trajectories.jsonl",
    )
    parser.add_argument(
        "--phase-f-dir-template",
        default="outputs/koopman_defense_phaseF_{arm}/trajectories.jsonl",
    )
    parser.add_argument(
        "--out-dir", type=pathlib.Path, default=pathlib.Path("outputs/koopman_detection_two_regime")
    )
    return parser.parse_args()


def load_model(fit_report_path: pathlib.Path, model_key: str):
    report = json.loads(fit_report_path.read_text())
    cfg = report["config"]
    model_report = report[model_key]
    config = ReducedStateConfig(nu=cfg["nu"], mu=cfg["mu"])
    model = surrogate_from_arrays(
        A=model_report["A"],
        B=model_report["B"],
        b=model_report["b"],
        C=model_report["C"],
        state_dim=config.state_dim,
        extra_features_fn=EXTRA_FEATURES_FNS[model_key],
        ridge=cfg["ridge"],
    )
    return model, config


def benign_held_out_ids(benign_fit_report_path: pathlib.Path) -> set[str]:
    return set(json.loads(benign_fit_report_path.read_text())["held_out_benign_ids"])


def load_attack_eval_rows(template: str) -> list[dict]:
    rows: list[dict] = []
    for arm in ARMS:
        for row in load_trajectories(pathlib.Path(template.format(arm=arm))):
            row = dict(row)
            row["trajectory_id"] = f"{arm}__{row['trajectory_id']}"
            row["true_label"] = "attack"
            rows.append(row)
    return rows


def load_benign_eval_rows(template: str, held_out_ids: set[str]) -> list[dict]:
    rows: list[dict] = []
    for arm in ARMS:
        for row in load_trajectories(pathlib.Path(template.format(arm=arm))):
            if row["benign_id"] not in held_out_ids:
                continue
            row = dict(row)
            row["trajectory_id"] = f"{arm}__{row['trajectory_id']}"
            row["true_label"] = "benign"
            rows.append(row)
    return rows


def per_trajectory_residuals(attack_model, benign_model, config, rows) -> list[dict]:
    records = []
    for traj_id, traj_rows in group_by_trajectory(rows, id_col="trajectory_id").items():
        pairs = build_reduced_state_pairs(traj_rows, config, y_col="y_safety", u_col="u_remind")
        if not pairs:
            continue
        true_label = traj_rows[0]["true_label"]
        for k, pair in enumerate(pairs):
            true_next = pair["z_next"][config.nu - 1]
            attack_pred = attack_model.readout(attack_model.step(pair["z"], pair["v"]))
            benign_pred = benign_model.readout(benign_model.step(pair["z"], pair["v"]))
            records.append(
                {
                    "trajectory_id": traj_id,
                    "true_label": true_label,
                    "turn_index": k,
                    "attack_abs_residual": abs(true_next - attack_pred),
                    "benign_abs_residual": abs(true_next - benign_pred),
                }
            )
    return records


def classify(df: pd.DataFrame, attack_col: str, benign_col: str) -> pd.Series:
    return np.where(df[attack_col] < df[benign_col], "attack", "benign")


def confusion(df: pd.DataFrame, pred_col: str) -> dict:
    labels = ["attack", "benign"]
    table = pd.crosstab(df["true_label"], df[pred_col]).reindex(index=labels, columns=labels, fill_value=0)
    accuracy = float((df["true_label"] == df[pred_col]).mean())
    return {"table": table.to_dict(), "accuracy": accuracy, "n": len(df)}


def main() -> None:
    args = parse_args()
    attack_model, config = load_model(args.attack_fit_report, args.attack_model_key)
    benign_model, config_benign = load_model(args.benign_fit_report, args.benign_model_key)
    assert (config.nu, config.mu) == (config_benign.nu, config_benign.mu), "state configs must match to compare"

    held_out_benign = benign_held_out_ids(args.benign_fit_report)
    eval_rows = load_attack_eval_rows(args.phase_e_dir_template) + load_benign_eval_rows(
        args.phase_f_dir_template, held_out_benign
    )

    records = per_trajectory_residuals(attack_model, benign_model, config, eval_rows)
    df = pd.DataFrame.from_records(records)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_dir / "two_regime_residuals.csv", index=False)

    df["pred_per_turn"] = classify(df, "attack_abs_residual", "benign_abs_residual")
    per_turn = confusion(df, "pred_per_turn")

    traj_summary = df.groupby(["trajectory_id", "true_label"], as_index=False)[
        ["attack_abs_residual", "benign_abs_residual"]
    ].mean()
    traj_summary["pred_per_trajectory"] = classify(traj_summary, "attack_abs_residual", "benign_abs_residual")
    per_trajectory = confusion(traj_summary, "pred_per_trajectory")

    report = {
        "attack_model": {"key": args.attack_model_key, "path": str(args.attack_fit_report)},
        "benign_model": {"key": args.benign_model_key, "path": str(args.benign_fit_report)},
        "n_attack_trajectories": int((traj_summary["true_label"] == "attack").sum()),
        "n_benign_trajectories": int((traj_summary["true_label"] == "benign").sum()),
        "per_turn": per_turn,
        "per_trajectory": per_trajectory,
    }
    (args.out_dir / "two_regime_detector_report.json").write_text(json.dumps(report, indent=2))

    print(f"n_attack_trajectories={report['n_attack_trajectories']} n_benign_trajectories={report['n_benign_trajectories']}")
    print("\nper-turn confusion (rows=true, cols=predicted):")
    print(pd.DataFrame(per_turn["table"]))
    print(f"per-turn accuracy: {per_turn['accuracy']:.4f} (n={per_turn['n']})")
    print("\nper-trajectory confusion (rows=true, cols=predicted):")
    print(pd.DataFrame(per_trajectory["table"]))
    print(f"per-trajectory accuracy: {per_trajectory['accuracy']:.4f} (n={per_trajectory['n']})")
    print(f"\nreport written to {args.out_dir / 'two_regime_detector_report.json'}")


if __name__ == "__main__":
    main()
