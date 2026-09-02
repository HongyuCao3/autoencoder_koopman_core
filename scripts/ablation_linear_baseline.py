#!/usr/bin/env python3
"""Ablation phase 2: does the AE's nonlinear lift beat a pure-linear Koopman fit?

Standalone, additive script for `ABLATION_STUDY.md` phase 2. It does not modify
`scripts/train.py` or `src/koopman_ae/core.py`; it only calls the existing
`AugmentedKoopmanModel` (already implemented in `core.py` but never wired into
the CLI) through the same `one_step_predictions` / `rollout_augmented_from_trajectories`
evaluation helpers `scripts/train.py` uses for the deep AE model, so the numbers
are directly comparable to the `results/sentence_length_t10-memory-*` runs from
ablation phase 1.

Usage: python scripts/ablation_linear_baseline.py
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd

from koopman_ae import (
    AugmentedKoopmanModel,
    AugmentedStateConfig,
    augmented_prediction_metrics,
    build_augmented_state_dataset,
    one_step_predictions,
    rollout_augmented_from_trajectories,
    rollout_metrics,
)

PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[1]
DATASET_PATH = PACKAGE_ROOT / "datasets/scalar/sentence_length_t10/trajectories.jsonl"
OUTPUT_COLUMNS = ("normalized_output",)
TARGET_COLUMNS = ("effective_norm",)
COMMON_SEED_TURNS = 4  # matches configs/dataset/sentence_length_t10.yaml
LAG = 3  # memory-3, the phase-1 winning state family

# Same run_name convention as scripts/train.py, with training_mode="linear_ridge"
# so it never collides with an AE run_name (those are always {training_mode}
# in {"joint", "reconstruction_then_ridge"}).
RUN_NAME = f"sentence_length_t10-memory-lag{LAG}-linear_ridge-k0-seed0"


def _split(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    return frame[frame["topic_split"].eq(name)].copy()


def main() -> None:
    frame = pd.read_json(DATASET_PATH, lines=True)
    frame["turn"] = pd.to_numeric(frame["turn"], errors="coerce").astype(int)
    frame = frame.sort_values(["trajectory_id", "turn"]).reset_index(drop=True)

    state_cfg = AugmentedStateConfig(
        output_memory=LAG + 1,
        input_memory=0,
        control_mode="error",
        output_columns=OUTPUT_COLUMNS,
        target_columns=TARGET_COLUMNS,
    )

    train_frame = _split(frame, "train")
    train_dataset = build_augmented_state_dataset(train_frame, state_cfg)

    model = AugmentedKoopmanModel(output_dim=train_dataset.output_dim, alpha=1e-6)
    model.name = "linear_ridge"
    model.fit(train_dataset.Z_t, train_dataset.R, train_dataset.Z_next)

    metrics: dict[str, dict[str, float]] = {}
    for split in ("train", "validation", "test"):
        split_frame = _split(frame, split)
        if split_frame.empty:
            continue
        dataset = build_augmented_state_dataset(split_frame, state_cfg)
        one_step = one_step_predictions(model, dataset)
        rollout = rollout_augmented_from_trajectories(
            model, split_frame, state_cfg, observed_seed_turns=COMMON_SEED_TURNS
        )
        one_step_summary = augmented_prediction_metrics(one_step)
        rollout_summary = rollout_metrics(rollout)
        metrics[split] = {
            "one_step_mse": one_step_summary["one_step_mse"],
            "one_step_mae": one_step_summary["one_step_mae"],
            "one_step_z_mse": one_step_summary["z_space_mse"],
            "one_step_z_mae": one_step_summary["z_space_mae"],
            "one_step_n": one_step_summary["n"],
            "rollout_mse": rollout_summary["rollout_mse"],
            "rollout_mae": rollout_summary["rollout_mae"],
            "rollout_z_mse": rollout_summary["z_space_mse"],
            "rollout_z_mae": rollout_summary["z_space_mae"],
            "rollout_n": rollout_summary["n"],
        }

    result_dir = PACKAGE_ROOT / "results" / RUN_NAME
    result_dir.mkdir(parents=True, exist_ok=True)
    run_metadata = {
        "run_name": RUN_NAME,
        "dataset": str(DATASET_PATH),
        "state_family": "memory",
        "lag": LAG,
        "model": "AugmentedKoopmanModel (pure affine, no AE lift)",
        "common_seed_turns": COMMON_SEED_TURNS,
        "metrics": metrics,
        "A_shape": list(np.asarray(model.A).shape),
        "B_shape": list(np.asarray(model.B).shape),
    }
    (result_dir / "run.json").write_text(json.dumps(run_metadata, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    print(f"[done] result_dir={result_dir}")


if __name__ == "__main__":
    main()
