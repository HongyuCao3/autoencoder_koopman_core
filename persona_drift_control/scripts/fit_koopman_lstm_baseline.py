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

Defaults to the v-ALIGNED pairing (`--contemporaneous-v`, on by default,
matching fit_koopman_ae_baseline.py): the 2026-09-01 run of this script
predates the `modeling/dataset.py` v-alignment fix, so its numbers are only
comparable to `koopman_fit_report.json`, not to the corrected
`koopman_fit_report_valigned.json` every other baseline is now scored
against. `--no-contemporaneous-v` reproduces the original run bit for bit
(it also has to be pointed back at the old report/out paths, see the
defaults of --koopman-fit-report/--out-path).

`--min-turn-index` defaults to whatever the chosen alignment makes the
Koopman-matched window: `build_reduced_state_pairs`' first predicted turn is
`start = max(nu-1, mu-shift)`, i.e. 2 unaligned and 1 v-aligned for
`nu=1, mu=2`. Hard-coding 2 for both would silently score the two runs on
different turn sets.

CPU-only (small model, ~60 trajectories). Run directly (no sbatch), or via
environment/run_lstm_baseline_valigned.sbatch where no python is available.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
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
        default=pathlib.Path("outputs/koopman_defense_phaseB_random_excite/koopman_fit_report_valigned.json"),
        help="richer_abs_sign's A/B/b/C shapes are read from here purely to report a comparable parameter count, plus its own MSEs for the printed side-by-side; the fitted matrices themselves are not used. Must match --contemporaneous-v: the _valigned report is the v-aligned fit, the unsuffixed one is the pre-fix fit.",
    )
    parser.add_argument(
        "--contemporaneous-v",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="pair y_t with v_(t+1) (the reminder acting on the turn being predicted) instead of v_t -- the LSTM counterpart of ReducedStateConfig.contemporaneous_v. On by default; --no-contemporaneous-v reproduces the 2026-09-01 pre-fix run.",
    )
    parser.add_argument("--held-out-frac", type=float, default=0.25)
    # 0.0 = the 2026-09-01 protocol: early-stop on the very trajectories that
    # get reported, which makes the reported held-out MSE the best-of-300
    # epochs BY that set -- optimistic, and harmless only while the LSTM is
    # losing anyway. Pass a fraction (0.15 matches the AE baseline) to move
    # the early-stopping set into the training attacks instead; the held-out
    # set is unchanged either way.
    parser.add_argument("--val-frac", type=float, default=0.0)
    # Only meaningful together with --val-frac: `held_out` keeps the reduced
    # training set but early-stops on the reported trajectories again, which
    # is the ablation that separates "the fair protocol costs accuracy" from
    # "carving out a validation split costs training data" -- the two move
    # the number the same way and would otherwise be reported as one effect.
    parser.add_argument("--early-stop-on", choices=["auto", "val", "held_out"], default="auto")
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[1, 2, 4, 8])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--patience", type=int, default=30)
    # Plural, unlike the 2026-09-01 run's single `--train-seed 0`: the AE
    # baseline (docs/experiments/ae_baseline_plan.md) found a one-seed result
    # that reversed once two more seeds were added, and gradient-descent
    # training on ~44 short trajectories has enough seed variance to make a
    # single number misleading either way. Reproduce the original run with
    # `--train-seeds 0`.
    parser.add_argument("--train-seeds", type=int, nargs="*", default=[0, 1, 2])
    # The LSTM has no (nu, mu) of its own -- its memory is the hidden state.
    # These only describe the Koopman model being compared against, so the
    # matched evaluation window can be derived from it rather than hard-coded.
    parser.add_argument("--nu", type=int, default=1, help="the COMPARED Koopman model's nu (only used to derive --min-turn-index).")
    parser.add_argument("--mu", type=int, default=2, help="the COMPARED Koopman model's mu (only used to derive --min-turn-index).")
    parser.add_argument(
        "--min-turn-index",
        type=int,
        default=None,
        help="only turn indices >= this are counted in the *_matched metrics. Defaults to richer_abs_sign's own first predicted turn, start=max(nu-1, mu-shift): 1 with --contemporaneous-v, 2 without.",
    )
    parser.add_argument(
        "--out-path",
        type=pathlib.Path,
        default=pathlib.Path("outputs/koopman_lstm_baseline/lstm_fit_report_valigned.json"),
        help="the pre-fix run's report stays at lstm_fit_report.json and is not overwritten.",
    )
    args = parser.parse_args()
    if args.min_turn_index is None:
        args.min_turn_index = max(args.nu - 1, args.mu - (1 if args.contemporaneous_v else 0))
    return args


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
    early_stop_rows_by_traj = held_out_rows_by_traj

    if args.val_frac > 0:
        # Carve the early-stopping set out of the TRAINING attacks instead of
        # early-stopping on the trajectories that get reported (see
        # train_lstm_surrogate's docstring). `split_by_system_prompt_id`
        # shuffles the attack ids once from `seed` and then slices
        # train/val/test in order, so shrinking train_frac by exactly
        # val_frac moves attacks from train into val and leaves the test
        # slice byte-identical -- asserted rather than assumed, because a
        # changed held-out set would silently make every number here
        # incomparable to the Koopman report it is printed against.
        val_split = split_by_system_prompt_id(
            rows,
            train_frac=1.0 - args.held_out_frac - args.val_frac,
            val_frac=args.val_frac,
            seed=args.split_seed,
            split_col="attack_id",
        )
        assert {r["attack_id"] for r in val_split["test"]} == {r["attack_id"] for r in split["test"]}
        train_rows_by_traj = list(group_by_trajectory(val_split["train"]).values())
        val_rows_by_traj = list(group_by_trajectory(val_split["val"]).values())
        if not val_rows_by_traj:
            raise SystemExit(f"--val-frac {args.val_frac} rounds to zero validation attacks; raise it.")
        early_stop_rows_by_traj = held_out_rows_by_traj if args.early_stop_on == "held_out" else val_rows_by_traj
    elif args.early_stop_on == "val":
        raise SystemExit("--early-stop-on val needs --val-frac > 0 (there is no validation split otherwise).")

    n_train_traj, n_held_out_traj = len(train_rows_by_traj), len(held_out_rows_by_traj)
    n_early_stop_traj = len(early_stop_rows_by_traj)

    koopman_params = koopman_param_count(args.koopman_fit_report) if args.koopman_fit_report.exists() else None

    results = []
    for hidden_size in args.hidden_sizes:
        for train_seed in args.train_seeds:
            t0 = time.perf_counter()
            model, train_info = train_lstm_surrogate(
                hidden_size=hidden_size,
                train_rows_by_traj=train_rows_by_traj,
                early_stop_rows_by_traj=early_stop_rows_by_traj,
                y_col="y_safety",
                u_col="u_remind",
                epochs=args.epochs,
                lr=args.lr,
                patience=args.patience,
                seed=train_seed,
                min_turn_index=args.min_turn_index,
                contemporaneous_v=args.contemporaneous_v,
            )
            train_seconds = time.perf_counter() - t0
            n_params = sum(p.numel() for p in model.parameters())

            train_one_step_preds = [
                teacher_forced_predictions(model, rows, y_col="y_safety", contemporaneous_v=args.contemporaneous_v)
                for rows in train_rows_by_traj
            ]
            held_out_rollout_preds = [
                rollout_predictions(model, rows, y_col="y_safety", contemporaneous_v=args.contemporaneous_v)
                for rows in held_out_rows_by_traj
            ]

            result = {
                "hidden_size": hidden_size,
                "train_seed": train_seed,
                "n_params": n_params,
                "n_train_trajectories": n_train_traj,
                "n_held_out_trajectories": n_held_out_traj,
                "n_early_stop_trajectories": n_early_stop_traj,
                "train_seconds": train_seconds,
                "n_epochs_run": train_info["n_epochs_run"],
                "best_epoch": train_info["best_epoch"],
                "train_one_step_mse_full": mse_from_predictions(train_one_step_preds, min_turn_index=0),
                "train_one_step_mse_matched": mse_from_predictions(
                    train_one_step_preds, min_turn_index=args.min_turn_index
                ),
                "held_out_rollout_mse_full": mse_from_predictions(held_out_rollout_preds, min_turn_index=0),
                "held_out_rollout_mse_matched": mse_from_predictions(
                    held_out_rollout_preds, min_turn_index=args.min_turn_index
                ),
            }
            results.append(result)
            print(
                f"H={hidden_size:>2} seed={train_seed} params={n_params:>3} epochs={result['n_epochs_run']:>3} "
                f"train_one_step_mse_matched={result['train_one_step_mse_matched']:.4f} "
                f"held_out_rollout_mse_matched={result['held_out_rollout_mse_matched']:.4f} "
                f"({train_seconds:.1f}s)"
            )

    # Per-hidden-size mean +/- population sd across train seeds, mirroring how
    # the AE baseline reports its own seed sweep (docs/experiments/ae_baseline_plan.md)
    # -- a single seed's number is not stable enough on ~44 short trajectories
    # to compare against a closed-form ridge fit that has no seed at all.
    by_hidden_size = []
    for hidden_size in args.hidden_sizes:
        runs = [r for r in results if r["hidden_size"] == hidden_size]
        rollout = [r["held_out_rollout_mse_matched"] for r in runs]
        one_step = [r["train_one_step_mse_matched"] for r in runs]
        by_hidden_size.append(
            {
                "hidden_size": hidden_size,
                "n_params": runs[0]["n_params"],
                "n_seeds": len(runs),
                "held_out_rollout_mse_matched_mean": float(np.mean(rollout)),
                "held_out_rollout_mse_matched_sd": float(np.std(rollout)),
                "train_one_step_mse_matched_mean": float(np.mean(one_step)),
                "train_one_step_mse_matched_sd": float(np.std(one_step)),
            }
        )
        print(
            f"H={hidden_size:>2} params={runs[0]['n_params']:>3} over {len(runs)} seeds: "
            f"held_out_rollout_mse_matched={np.mean(rollout):.4f} +/- {np.std(rollout):.4f}"
        )

    report = {
        "config": {
            "rows_path": str(args.rows_path),
            "held_out_frac": args.held_out_frac,
            "val_frac": args.val_frac,
            "early_stop_on": args.early_stop_on,
            "split_seed": args.split_seed,
            "epochs": args.epochs,
            "lr": args.lr,
            "patience": args.patience,
            "train_seeds": list(args.train_seeds),
            "min_turn_index": args.min_turn_index,
            "contemporaneous_v": args.contemporaneous_v,
            "nu": args.nu,
            "mu": args.mu,
            "koopman_fit_report": str(args.koopman_fit_report),
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
        "by_hidden_size": by_hidden_size,
        "results": results,
    }
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_path.write_text(json.dumps(report, indent=2))
    print(f"koopman richer_abs_sign: n_params={koopman_params} held_out_rollout_mse={report['koopman_richer_abs_sign_held_out_rollout_mse']:.4f}")
    print(f"report written to {args.out_path}")


if __name__ == "__main__":
    main()
