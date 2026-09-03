#!/usr/bin/env python3
"""Ablation phase 6 (continued): sweep rollout horizon on sentence_length_t10
itself, holding lag=3 (the state family phase 5's comparison actually used)
fixed -- unlike ablation_horizon_probe.py, which had to switch to lag=1 to get
a horizon>1 condition out of a T=5 dataset, T=10 data lets common_seed_turns
range from 4 (horizon=6, phase 2's original condition) up to 9 (horizon=1)
without changing the state family at all. This is the clean version of the
probe: task and lag both held fixed, only rollout horizon varies.

Linear-baseline side only (mirrors ablation_linear_baseline.py); the AE side
is `scripts/train.py dataset=sentence_length_t10 state=memory state.lag=3 ...
trainer.common_seed_turns=<9|8|6>` (see ABLATION_STUDY.md phase 6b for exact
commands). Additive script; does not modify any existing file.

Usage: python scripts/ablation_horizon_sweep_t10.py
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
DATASET_PATH = PACKAGE_ROOT / "datasets/scalar/sentence_length_t10/trajectories.jsonl"
OUTPUT_COLUMNS = ("normalized_output",)
TARGET_COLUMNS = ("effective_norm",)
LAG = 3

CONDITIONS = [
    ("seedturns9-horizon1", 9),
    ("seedturns8-horizon2", 8),
    ("seedturns6-horizon4", 6),
    ("seedturns4-horizon6", 4),  # replicates phase 2's original condition
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

    run_name = f"sentence_length_t10-memory-lag{LAG}-linear_ridge-k0-{label}"
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
