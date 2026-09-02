#!/usr/bin/env python3
"""LSTM surrogate baseline (docs/experiments/lstm_baseline_plan.md): fits
`modeling.lstm_baseline.LSTMSurrogate` on the exact same Phase B
random-excitation trajectories and the exact same `attack_id` 75/25 split
(seed=0) that `fit_koopman_defense_model.py` used for `richer_abs_sign`
(`nu=1, mu=2`), then reports one-step (teacher-forced) and rollout MSE on a
turn-window matched to that Koopman model so the two numbers are directly
comparable -- see `BASELINES.md`'s layer-3 ablation this closes.

Sweeps a small grid of hidden sizes (default 1/2/4/8) rather than a single
size, since the honest comparison this baseline is meant to make includes
whether any advantage survives shrinking the LSTM toward Koopman's own
parameter count (~40 for richer_abs_sign) -- see the plan doc's "parameter
count alignment" section.

CPU-only (small model, ~60 trajectories). Run directly (no sbatch).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from persona_drift.modeling.dataset import group_by_trajectory, load_trajectories, split_by_system_prompt_id  # noqa: E402
from persona_drift.modeling.lstm_baseline import (  # noqa: E402
    mse_from_predictions,
    rollout_predictions,
    teacher_forced_predictions,
    train_lstm_surrogate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rows-path", type=pathlib.Path, default=pathlib.Path("outputs/koopman_defense_phaseB_random_excite/trajectories.jsonl")
    )
    parser.add_argument(
        "--koopman-fit-report",
        type=pathlib.Path,
        default=pathlib.Path("outputs/koopman_defense_phaseB_random_excite/koopman_fit_report.json"),
        help="richer_abs_sign's A/B/b/C shapes are read from here purely to report a comparable parameter count; the fitted matrices themselves are not used.",
    )
    parser.add_argument("--held-out-frac", type=float, default=0.25)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[1, 2, 4, 8])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--train-seed", type=int, default=0)
    parser.add_argument(
        "--min-turn-index",
        type=int,
        default=2,
        help="matches richer_abs_sign's nu=1,mu=2 -> start=max(nu-1,mu)=2; only turn indices >= this are counted in the *_matched metrics.",
    )
    parser.add_argument("--out-path", type=pathlib.Path, default=pathlib.Path("outputs/koopman_lstm_baseline/lstm_fit_report.json"))
    return parser.parse_args()


def koopman_param_count(fit_report_path: pathlib.Path, model_key: str = "richer_abs_sign") -> int:
    report = json.loads(fit_report_path.read_text())[model_key]
    total = 0
    for key in ("A", "B", "b", "C"):
        value = report[key]
        n = 1
        shape = []
        arr = value
        while isinstance(arr, list):
            shape.append(len(arr))
            arr = arr[0] if arr else []
        for dim in shape:
            n *= dim
        total += n
    return total


def main() -> None:
    args = parse_args()
    rows = load_trajectories(args.rows_path)

    split = split_by_system_prompt_id(
        rows, train_frac=1.0 - args.held_out_frac, val_frac=0.0, seed=args.split_seed, split_col="attack_id"
    )
    train_rows_by_traj = list(group_by_trajectory(split["train"]).values())
    held_out_rows_by_traj = list(group_by_trajectory(split["test"]).values())
    n_train_traj, n_held_out_traj = len(train_rows_by_traj), len(held_out_rows_by_traj)

    koopman_params = koopman_param_count(args.koopman_fit_report) if args.koopman_fit_report.exists() else None

    results = []
    for hidden_size in args.hidden_sizes:
        t0 = time.perf_counter()
        model, train_info = train_lstm_surrogate(
            hidden_size=hidden_size,
            train_rows_by_traj=train_rows_by_traj,
            held_out_rows_by_traj=held_out_rows_by_traj,
            y_col="y_safety",
            u_col="u_remind",
            epochs=args.epochs,
            lr=args.lr,
            patience=args.patience,
            seed=args.train_seed,
            min_turn_index=args.min_turn_index,
        )
        train_seconds = time.perf_counter() - t0
        n_params = sum(p.numel() for p in model.parameters())

        train_one_step_preds = [teacher_forced_predictions(model, rows, y_col="y_safety") for rows in train_rows_by_traj]
        held_out_rollout_preds = [rollout_predictions(model, rows, y_col="y_safety") for rows in held_out_rows_by_traj]

        result = {
            "hidden_size": hidden_size,
            "n_params": n_params,
            "n_train_trajectories": n_train_traj,
            "n_held_out_trajectories": n_held_out_traj,
            "train_seconds": train_seconds,
            "n_epochs_run": train_info["n_epochs_run"],
            "best_epoch": train_info["best_epoch"],
            "train_one_step_mse_full": mse_from_predictions(train_one_step_preds, min_turn_index=0),
            "train_one_step_mse_matched": mse_from_predictions(train_one_step_preds, min_turn_index=args.min_turn_index),
            "held_out_rollout_mse_full": mse_from_predictions(held_out_rollout_preds, min_turn_index=0),
            "held_out_rollout_mse_matched": mse_from_predictions(held_out_rollout_preds, min_turn_index=args.min_turn_index),
        }
        results.append(result)
        print(
            f"H={hidden_size:>2} params={n_params:>3} epochs={result['n_epochs_run']:>3} "
            f"train_one_step_mse_matched={result['train_one_step_mse_matched']:.4f} "
            f"held_out_rollout_mse_matched={result['held_out_rollout_mse_matched']:.4f} "
            f"({train_seconds:.1f}s)"
        )

    report = {
        "config": {
            "rows_path": str(args.rows_path),
            "held_out_frac": args.held_out_frac,
            "split_seed": args.split_seed,
            "epochs": args.epochs,
            "lr": args.lr,
            "patience": args.patience,
            "train_seed": args.train_seed,
            "min_turn_index": args.min_turn_index,
        },
        "n_train_trajectories": n_train_traj,
        "n_held_out_trajectories": n_held_out_traj,
        "koopman_richer_abs_sign_n_params": koopman_params,
        "koopman_richer_abs_sign_held_out_rollout_mse": (
            json.loads(args.koopman_fit_report.read_text())["richer_abs_sign"]["held_out_rollout_mse"]
            if args.koopman_fit_report.exists()
            else None
        ),
        "koopman_richer_abs_sign_train_one_step_mse": (
            json.loads(args.koopman_fit_report.read_text())["richer_abs_sign"]["train_one_step_mse"]
            if args.koopman_fit_report.exists()
            else None
        ),
        "results": results,
    }
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_path.write_text(json.dumps(report, indent=2))
    print(f"koopman richer_abs_sign: n_params={koopman_params} held_out_rollout_mse={report['koopman_richer_abs_sign_held_out_rollout_mse']:.4f}")
    print(f"report written to {args.out_path}")


if __name__ == "__main__":
    main()
