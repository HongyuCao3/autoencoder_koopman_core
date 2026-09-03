#!/usr/bin/env python3
"""Ablation phase 5: extend phase 2 (AE vs pure-linear Koopman baseline) from
`sentence_length_t10` alone to the other 7 registered tasks (ABLATION_STUDY.md
"后续阶段" list). Same method as `scripts/ablation_linear_baseline.py`
(`AugmentedKoopmanModel`, no AE, closed-form ridge, `state=memory,lag=3`, same
`one_step_predictions` / `rollout_augmented_from_trajectories` evaluation
helpers `scripts/train.py` uses for the deep AE), just parameterized over the
dataset registry instead of hardcoded to one dataset. Additive script; does
not modify `ablation_linear_baseline.py`, `scripts/train.py`, or `core.py`.

Usage: python scripts/ablation_linear_baseline_all_tasks.py
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
import yaml

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
LAG = 3  # memory-3, the phase-1 winning state family

TASKS = [
    "average_word_length_t5",
    "character_length_t5",
    "even_odd_t5",
    "formality_t5",
    "sentiment_t5",
    "vector_count_stage1_t10",
    "vector_count_stage2_t10",
]


def _split(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    return frame[frame["topic_split"].eq(name)].copy()


def run_task(task: str) -> dict:
    cfg = yaml.safe_load((PACKAGE_ROOT / "configs" / "dataset" / f"{task}.yaml").read_text())
    dataset_path = PACKAGE_ROOT / "datasets" / cfg["path"]
    output_columns = tuple(cfg["output_columns"])
    target_columns = tuple(cfg["target_columns"])
    common_seed_turns = cfg["common_seed_turns"]

    frame = pd.read_json(dataset_path, lines=True)
    frame["turn"] = pd.to_numeric(frame["turn"], errors="coerce").astype(int)
    frame = frame.sort_values(["trajectory_id", "turn"]).reset_index(drop=True)

    state_cfg = AugmentedStateConfig(
        output_memory=LAG + 1,
        input_memory=0,
        control_mode="error",
        output_columns=output_columns,
        target_columns=target_columns,
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
            model, split_frame, state_cfg, observed_seed_turns=common_seed_turns
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

    run_name = f"{task}-memory-lag{LAG}-linear_ridge-k0-seed0"
    result_dir = PACKAGE_ROOT / "results" / run_name
    result_dir.mkdir(parents=True, exist_ok=True)
    run_metadata = {
        "run_name": run_name,
        "dataset": str(dataset_path),
        "state_family": "memory",
        "lag": LAG,
        "model": "AugmentedKoopmanModel (pure affine, no AE lift)",
        "common_seed_turns": common_seed_turns,
        "metrics": metrics,
        "A_shape": list(np.asarray(model.A).shape),
        "B_shape": list(np.asarray(model.B).shape),
    }
    (result_dir / "run.json").write_text(json.dumps(run_metadata, indent=2) + "\n")
    print(f"[done] task={task} result_dir={result_dir}")
    return run_metadata


def main() -> None:
    all_results = {}
    for task in TASKS:
        all_results[task] = run_task(task)
    summary_path = PACKAGE_ROOT / "results" / "ablation_linear_baseline_all_tasks_summary.json"
    summary_path.write_text(json.dumps(all_results, indent=2) + "\n")
    print(f"[done] summary={summary_path}")


if __name__ == "__main__":
    main()
