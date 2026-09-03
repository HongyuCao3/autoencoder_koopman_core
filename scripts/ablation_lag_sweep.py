#!/usr/bin/env python3
"""Ablation phase 7b: does `lag` (state dimension) explain the AE-vs-linear
split, independent of the (already-refuted, phase 6) horizon hypothesis and
the (already-refuted, phase 7a) dataset-size hypothesis?

`ablation_horizon_probe.py` (phase 6) found `average_word_length_t5`'s huge
AE advantage at `lag=3` (44.6%) collapsed to ~2-4% at `lag=1`, suggesting
`lag` itself might drive the effect size. This script checks the biggest
opposite-direction outlier -- `sentiment_t5`, where linear beats AE by 78.1%
at `lag=3` -- across `lag in {1, 2}` (both legal on T=5 data with
`common_seed_turns=4` fixed, so horizon stays at 1 throughout, no horizon
confound), linear-baseline side only (mirrors ablation_linear_baseline.py).
The AE side is `scripts/train.py dataset=sentiment_t5 state=memory
state.lag=<1|2> ...` (see ABLATION_STUDY.md phase 7b for exact commands).
Additive script.

Usage: python scripts/ablation_lag_sweep.py
"""

from __future__ import annotations

import json
import pathlib

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
DATASET_PATH = PACKAGE_ROOT / "datasets/scalar/sentiment_t5/trajectories.jsonl"
OUTPUT_COLUMNS = ("normalized_output",)
TARGET_COLUMNS = ("effective_norm",)
COMMON_SEED_TURNS = 4
LAGS = [1, 2]


def _split(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    return frame[frame["topic_split"].eq(name)].copy()


def run_lag(lag: int) -> dict:
    frame = pd.read_json(DATASET_PATH, lines=True)
    frame["turn"] = pd.to_numeric(frame["turn"], errors="coerce").astype(int)
    frame = frame.sort_values(["trajectory_id", "turn"]).reset_index(drop=True)

    state_cfg = AugmentedStateConfig(
        output_memory=lag + 1,
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
            "rollout_mse": rollout_summary["rollout_mse"],
            "rollout_n": rollout_summary["n"],
        }

    run_name = f"sentiment_t5-memory-lag{lag}-linear_ridge-k0-seed0"
    result_dir = PACKAGE_ROOT / "results" / run_name
    result_dir.mkdir(parents=True, exist_ok=True)
    run_metadata = {
        "run_name": run_name,
        "lag": lag,
        "model": "AugmentedKoopmanModel (pure affine, no AE lift)",
        "metrics": metrics,
    }
    (result_dir / "run.json").write_text(json.dumps(run_metadata, indent=2) + "\n")
    print(f"[done] lag={lag} result_dir={result_dir}")
    return run_metadata


def main() -> None:
    for lag in LAGS:
        run_lag(lag)


if __name__ == "__main__":
    main()
