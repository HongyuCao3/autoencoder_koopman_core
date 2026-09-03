#!/usr/bin/env python3
"""AE (encoder-decoder) Koopman baseline (docs/experiments/ae_baseline_plan.md):
fits `modeling.ae_baseline.AEKoopmanSurrogate` on the exact same Phase B
random-excitation trajectories, `attack_id` 75/25 split (seed=0), and
`nu=1, mu=2, contemporaneous_v=True` state config that
`fit_koopman_defense_model.py` used for the v-alignment-corrected
`richer_abs_sign`/`arx` numbers in `koopman_fit_report_valigned.json`, then
reports one-step (train) and rollout (held-out) MSE via the SAME
`modeling.evaluate.one_step_error`/`rollout_output_error` those models use
-- unlike the LSTM baseline, no separate eval code is needed because this
model's `step`/`readout` operate on the same raw `ReducedStateConfig` `z`
(see the plan doc's "why not write new eval code" section).

Sweeps a small grid of `latent_dim` (default 1/2/4), `hidden_dim` fixed
small (default 4) -- see the plan doc's "parameter count alignment" section
for why these sizes were chosen over `core.py`'s much larger defaults.

CPU-only. Run directly (no sbatch).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from persona_drift.modeling.ae_baseline import AEKoopmanConfig, AEKoopmanSurrogate  # noqa: E402
from persona_drift.modeling.dataset import (  # noqa: E402
    ReducedStateConfig,
    build_identification_dataset,
    load_trajectories,
    split_by_system_prompt_id,
)
from persona_drift.modeling.evaluate import one_step_error, rollout_output_error  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rows-path", type=pathlib.Path, default=pathlib.Path("outputs/koopman_defense_phaseB_random_excite/trajectories.jsonl")
    )
    parser.add_argument(
        "--koopman-fit-report",
        type=pathlib.Path,
        default=pathlib.Path("outputs/koopman_defense_phaseB_random_excite/koopman_fit_report_valigned.json"),
        help="richer_abs_sign's/arx's params and metrics are read from here purely for a comparable report; not used to fit anything.",
    )
    parser.add_argument("--nu", type=int, default=1)
    parser.add_argument("--mu", type=int, default=2)
    parser.add_argument("--contemporaneous-v", action="store_true", default=True)
    parser.add_argument("--held-out-frac", type=float, default=0.25)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--latent-dims", type=int, nargs="*", default=[1, 2, 4])
    parser.add_argument("--hidden-dim", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--num-epochs", type=int, default=300, help="hard cap; with --early-stopping-patience set, training usually stops earlier")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--dynamics-alpha", type=float, default=1e-4)
    parser.add_argument("--train-seed", type=int, default=0)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=None,
        help="null/unset = old behavior (always run exactly --num-epochs); "
        "int = stop stage-1 (reconstruction) once a validation split carved "
        "out of the training attacks stops improving for this many epochs",
    )
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-6)
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.15,
        help="fraction of ALL attacks set aside as an early-stopping validation "
        "split, carved OUT of the training block (held-out test fraction is "
        "unchanged) -- only used when --early-stopping-patience is set",
    )
    parser.add_argument("--out-path", type=pathlib.Path, default=pathlib.Path("outputs/koopman_ae_baseline/ae_fit_report.json"))
    return parser.parse_args()


def _n_params(fit_report: dict, model_key: str) -> int:
    total = 0
    for key in ("A", "B", "b", "C"):
        value = fit_report[model_key][key]
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

    early_stopping_requested = args.early_stopping_patience is not None
    val_frac = args.val_frac if early_stopping_requested else 0.0
    # Same seed/shuffle as the val_frac=0.0 case below: carving a validation
    # slice out of the training block leaves the held-out test block (the
    # 25% used for held_out_rollout_mse, comparable across every baseline)
    # unchanged, up to rounding of attack counts.
    split = split_by_system_prompt_id(
        rows,
        train_frac=1.0 - args.held_out_frac - val_frac,
        val_frac=val_frac,
        seed=args.split_seed,
        split_col="attack_id",
    )
    train_rows = split["train"]
    val_rows = split["val"] if early_stopping_requested else []
    held_out_rows = split["test"]
    n_train_attacks = len({r["attack_id"] for r in train_rows})
    n_val_attacks = len({r["attack_id"] for r in val_rows})
    held_out_attack_ids = sorted({r["attack_id"] for r in held_out_rows})
    n_held_out_attacks = len(held_out_attack_ids)

    config = ReducedStateConfig(nu=args.nu, mu=args.mu, contemporaneous_v=args.contemporaneous_v)
    train_dataset = build_identification_dataset(train_rows, config, y_col="y_safety")
    val_dataset = (
        build_identification_dataset(val_rows, config, y_col="y_safety") if early_stopping_requested else None
    )
    if early_stopping_requested and val_dataset["Z"].shape[0] == 0:
        raise ValueError(
            f"--val-frac={val_frac} produced an empty validation split "
            f"({n_val_attacks} attacks); raise --val-frac or drop --early-stopping-patience"
        )

    koopman_report = json.loads(args.koopman_fit_report.read_text()) if args.koopman_fit_report.exists() else None

    results = []
    for latent_dim in args.latent_dims:
        t0 = time.perf_counter()
        model = AEKoopmanSurrogate(
            state_dim=config.state_dim,
            config=AEKoopmanConfig(
                latent_dim=latent_dim,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                num_epochs=args.num_epochs,
                learning_rate=args.learning_rate,
                dynamics_alpha=args.dynamics_alpha,
                random_state=args.train_seed,
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_min_delta=args.early_stopping_min_delta,
            ),
        ).fit(train_dataset, val_dataset=val_dataset)
        train_seconds = time.perf_counter() - t0
        last_history = model.training_history_[-1]

        result = {
            "latent_dim": latent_dim,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "n_params": model.n_params(),
            "epochs_run": len(model.training_history_),
            "early_stopped": last_history.get("early_stopped"),
            "restored_best_epoch": last_history.get("restored_best_epoch"),
            "restored_best_val_reconstruction_loss": last_history.get("restored_best_val_loss"),
            "train_seconds": train_seconds,
            "final_reconstruction_loss": model.training_history_[-1]["reconstruction_loss"],
            "train_one_step_mse": one_step_error(model, train_dataset),
            "held_out_rollout_mse": rollout_output_error(model, held_out_rows, config, y_col="y_safety"),
        }
        results.append(result)
        print(
            f"latent_dim={latent_dim:>2} params={result['n_params']:>3} "
            f"epochs_run={result['epochs_run']:>4} "
            f"final_reconstruction_loss={result['final_reconstruction_loss']:.4f} "
            f"train_one_step_mse={result['train_one_step_mse']:.4f} "
            f"held_out_rollout_mse={result['held_out_rollout_mse']:.4f} "
            f"({train_seconds:.1f}s)"
        )

    report = {
        "config": {
            "rows_path": str(args.rows_path),
            "nu": args.nu,
            "mu": args.mu,
            "contemporaneous_v": args.contemporaneous_v,
            "held_out_frac": args.held_out_frac,
            "val_frac": val_frac,
            "n_val_attacks": n_val_attacks,
            "split_seed": args.split_seed,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "num_epochs": args.num_epochs,
            "early_stopping_patience": args.early_stopping_patience,
            "early_stopping_min_delta": args.early_stopping_min_delta,
            "learning_rate": args.learning_rate,
            "dynamics_alpha": args.dynamics_alpha,
            "train_seed": args.train_seed,
        },
        "n_train_attacks": n_train_attacks,
        "n_held_out_attacks": n_held_out_attacks,
        "held_out_attack_ids": held_out_attack_ids,
        "koopman_richer_abs_sign_n_params": _n_params(koopman_report, "richer_abs_sign") if koopman_report else None,
        "koopman_richer_abs_sign_train_one_step_mse": (
            koopman_report["richer_abs_sign"]["train_one_step_mse"] if koopman_report else None
        ),
        "koopman_richer_abs_sign_held_out_rollout_mse": (
            koopman_report["richer_abs_sign"]["held_out_rollout_mse"] if koopman_report else None
        ),
        "koopman_arx_held_out_rollout_mse": koopman_report["arx"]["held_out_rollout_mse"] if koopman_report else None,
        "results": results,
    }
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_path.write_text(json.dumps(report, indent=2))
    print(f"n_train_attacks={n_train_attacks} n_held_out_attacks={n_held_out_attacks}")
    if koopman_report:
        print(
            f"koopman richer_abs_sign: n_params={report['koopman_richer_abs_sign_n_params']} "
            f"held_out_rollout_mse={report['koopman_richer_abs_sign_held_out_rollout_mse']:.4f}"
        )
    print(f"report written to {args.out_path}")


if __name__ == "__main__":
    main()
