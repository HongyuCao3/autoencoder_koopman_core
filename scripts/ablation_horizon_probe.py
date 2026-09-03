#!/usr/bin/env python3
"""Ablation phase 6: isolate the rollout-horizon effect from task identity.

Phase 5 found AE beats linear on `average_word_length_t5` at rollout_horizon=1
(T=5, common_seed_turns=4 forces exactly 1 rollout step) but linear beats AE
on all three T=10 tasks (rollout_horizon=6). That's confounded: horizon and
task identity vary together, so the horizon story could just be a coincidence
of which tasks happen to be T=5 vs T=10.

This probe holds task AND lag fixed (`average_word_length_t5`, `state=memory,
lag=1`, chosen because lag=1 -> minimum_seed=2 -> `common_seed_turns=2` is
legal on T=5 data, unlike lag=3 which needs seed>=4) and varies ONLY
`common_seed_turns`:
  - seed_turns=4 -> rollout_horizon=1 (T - seed_turns = 5 - 4)
  - seed_turns=2 -> rollout_horizon=3 (5 - 2)

If linear closes the gap (or overtakes) at horizon=3 while AE keeps its edge
at horizon=1, that is direct evidence horizon -- not the task's readout
nonlinearity -- drives phase 5's AE-vs-linear split. Additive script; does
not modify train.py, core.py, or the other ablation scripts.

Usage: python scripts/ablation_horizon_probe.py
(run scripts/train.py for the AE side separately -- see ABLATION_STUDY.md
phase 6 for the exact commands; this script only does the linear-baseline
side, mirroring ablation_linear_baseline.py/_all_tasks.py.)
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
DATASET_PATH = PACKAGE_ROOT / "datasets/scalar/average_word_length_t5/trajectories.jsonl"
OUTPUT_COLUMNS = ("normalized_output",)
TARGET_COLUMNS = ("effective_norm",)
LAG = 1

CONDITIONS = [
    ("seedturns4-horizon1", 4),
    ("seedturns2-horizon3", 2),
]


def _split(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    return frame[frame["topic_split"].eq(name)].copy()


def run_condition(label: str, seed_turns: int) -> dict:
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
            model, split_frame, state_cfg, observed_seed_turns=seed_turns
        )
        one_step_summary = augmented_prediction_metrics(one_step)
        rollout_summary = rollout_metrics(rollout)
        metrics[split] = {
            "one_step_mse": one_step_summary["one_step_mse"],
            "rollout_mse": rollout_summary["rollout_mse"],
            "rollout_n": rollout_summary["n"],
        }

    run_name = f"average_word_length_t5-memory-lag{LAG}-linear_ridge-k0-{label}"
    result_dir = PACKAGE_ROOT / "results" / run_name
    result_dir.mkdir(parents=True, exist_ok=True)
    run_metadata = {
        "run_name": run_name,
        "lag": LAG,
        "common_seed_turns": seed_turns,
        "model": "AugmentedKoopmanModel (pure affine, no AE lift)",
        "metrics": metrics,
    }
    (result_dir / "run.json").write_text(json.dumps(run_metadata, indent=2) + "\n")
    print(f"[done] {label} result_dir={result_dir}")
    return run_metadata


def main() -> None:
    for label, seed_turns in CONDITIONS:
        run_condition(label, seed_turns)


if __name__ == "__main__":
    main()
