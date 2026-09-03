#!/usr/bin/env python3
"""Task-scoped Hydra entrypoint for the AE (encoder-decoder) Koopman
baseline (same fitting code as scripts/fit_koopman_ae_baseline.py, which
keeps working unchanged for reproducing the already-written-up numbers --
this is an additive entrypoint for new fits, not a replacement).

See scripts/fit_koopman_hydra.py's docstring for why parameters come from
conf/task/<name>.yaml's `fit` block instead of argparse flags.

Usage:
    python scripts/fit_koopman_ae_hydra.py                                       # task=defense (default)
    python scripts/fit_koopman_ae_hydra.py task.fit.learning_rate=5e-4 task.fit.latent_dims=[2,4,8]
    python scripts/fit_koopman_ae_hydra.py task=persona_drift task.fit.rows_path=outputs/foo/trajectories.jsonl

CPU-only. Run directly (no sbatch).
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import hydra  # noqa: E402
from hydra.core.hydra_config import HydraConfig  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

from persona_drift.modeling.ae_baseline import AEKoopmanConfig, AEKoopmanSurrogate  # noqa: E402
from persona_drift.modeling.dataset import (  # noqa: E402
    ReducedStateConfig,
    build_identification_dataset,
    load_trajectories,
    split_by_system_prompt_id,
)
from persona_drift.modeling.evaluate import one_step_error, rollout_output_error  # noqa: E402

CONFIG_DIR = str(pathlib.Path(__file__).resolve().parents[1] / "conf")


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="fit_koopman_ae")
def main(cfg: DictConfig) -> None:
    fit = cfg.task.fit
    out_dir = pathlib.Path(HydraConfig.get().runtime.output_dir)
    rows = load_trajectories(fit.rows_path)

    early_stopping_requested = fit.early_stopping_patience is not None
    val_frac = fit.val_frac if early_stopping_requested else 0.0
    split = split_by_system_prompt_id(
        rows,
        train_frac=1.0 - fit.held_out_frac - val_frac,
        val_frac=val_frac,
        seed=fit.split_seed,
        split_col=fit.split_col,
    )
    train_rows = split["train"]
    val_rows = split["val"] if early_stopping_requested else []
    held_out_rows = split["test"]
    n_train = len({r[fit.split_col] for r in train_rows})
    n_val = len({r[fit.split_col] for r in val_rows})
    held_out_ids = sorted({r[fit.split_col] for r in held_out_rows})

    state_config = ReducedStateConfig(nu=fit.nu, mu=fit.mu, contemporaneous_v=fit.contemporaneous_v)
    train_dataset = build_identification_dataset(train_rows, state_config, y_col=fit.y_col)
    val_dataset = (
        build_identification_dataset(val_rows, state_config, y_col=fit.y_col) if early_stopping_requested else None
    )
    if early_stopping_requested and val_dataset["Z"].shape[0] == 0:
        raise ValueError(
            f"task.fit.val_frac={val_frac} produced an empty validation split "
            f"({n_val} groups); raise it or unset task.fit.early_stopping_patience"
        )

    results = []
    for latent_dim in fit.latent_dims:
        t0 = time.perf_counter()
        model = AEKoopmanSurrogate(
            state_dim=state_config.state_dim,
            config=AEKoopmanConfig(
                latent_dim=latent_dim,
                hidden_dim=fit.hidden_dim,
                num_layers=fit.num_layers,
                num_epochs=fit.num_epochs,
                learning_rate=fit.learning_rate,
                dynamics_alpha=fit.dynamics_alpha,
                random_state=fit.train_seed,
                early_stopping_patience=fit.early_stopping_patience,
                early_stopping_min_delta=fit.early_stopping_min_delta,
            ),
        ).fit(train_dataset, val_dataset=val_dataset)
        train_seconds = time.perf_counter() - t0
        last_history = model.training_history_[-1]

        result = {
            "latent_dim": latent_dim,
            "hidden_dim": fit.hidden_dim,
            "num_layers": fit.num_layers,
            "n_params": model.n_params(),
            "epochs_run": len(model.training_history_),
            "early_stopped": last_history.get("early_stopped"),
            "restored_best_epoch": last_history.get("restored_best_epoch"),
            "restored_best_val_reconstruction_loss": last_history.get("restored_best_val_loss"),
            "train_seconds": train_seconds,
            "final_reconstruction_loss": model.training_history_[-1]["reconstruction_loss"],
            "train_one_step_mse": one_step_error(model, train_dataset),
            "held_out_rollout_mse": rollout_output_error(model, held_out_rows, state_config, y_col=fit.y_col),
        }
        results.append(result)
        print(
            f"latent_dim={latent_dim:>2} params={result['n_params']:>3} "
            f"epochs_run={result['epochs_run']:>4} "
            f"held_out_rollout_mse={result['held_out_rollout_mse']:.4f} ({train_seconds:.1f}s)"
        )

    report = {
        "task": cfg.task.name,
        "config": OmegaConf.to_container(fit, resolve=True),
        "n_train": n_train,
        "n_held_out": len(held_out_ids),
        "held_out_ids": held_out_ids,
        "results": results,
    }
    out_path = out_dir / "ae_fit_report.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"task={cfg.task.name} n_train={n_train} n_held_out={len(held_out_ids)}")
    print(f"report written to {out_path}")


if __name__ == "__main__":
    main()
