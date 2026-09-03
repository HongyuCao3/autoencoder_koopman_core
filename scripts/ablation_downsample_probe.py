#!/usr/bin/env python3
"""Ablation phase 7a: does dataset size explain why AE loses badly on
`sentiment_t5` (60 trajectories total, AE rollout_mse 78% worse than linear)
but wins on `formality_t5` (252 trajectories, AE mildly better) -- phase 5's
two external-NLP-classifier-readout tasks, which otherwise look similar
(both T=5, both use a scorer model rather than a deterministic text metric)?

Downsamples `formality_t5` to the SAME per-split trajectory counts as
`sentiment_t5` (train=40, validation=10, test=10; sentiment_t5's own
train/val/test trajectory counts, read from the data rather than
hardcoded) with a fixed seed, writes the subsample to
`results/_scratch/formality_t5_downsampled/trajectories.jsonl` (under
`results/`, already gitignored -- this is a derived probe artifact, not
canonical data), then fits + evaluates the linear baseline on it exactly
like `ablation_linear_baseline_all_tasks.py`. The AE side is
`scripts/train.py dataset=custom dataset.path=<that file> ...` (see
ABLATION_STUDY.md phase 7a for the exact command). Additive script.

Usage: python scripts/ablation_downsample_probe.py
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
FORMALITY_PATH = PACKAGE_ROOT / "datasets/scalar/formality_t5/trajectories.jsonl"
SENTIMENT_PATH = PACKAGE_ROOT / "datasets/scalar/sentiment_t5/trajectories.jsonl"
OUT_PATH = PACKAGE_ROOT / "results/_scratch/formality_t5_downsampled/trajectories.jsonl"
OUTPUT_COLUMNS = ("normalized_output",)
TARGET_COLUMNS = ("effective_norm",)
LAG = 3
COMMON_SEED_TURNS = 4
DOWNSAMPLE_SEED = 0


def _split(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    return frame[frame["topic_split"].eq(name)].copy()


def build_downsampled_formality() -> pd.DataFrame:
    formality = pd.read_json(FORMALITY_PATH, lines=True)
    sentiment = pd.read_json(SENTIMENT_PATH, lines=True)
    target_counts = sentiment.groupby("topic_split")["trajectory_id"].nunique().to_dict()

    rng = np.random.default_rng(DOWNSAMPLE_SEED)
    kept_frames = []
    for split_name, target_n in target_counts.items():
        split_frame = _split(formality, split_name)
        traj_ids = split_frame["trajectory_id"].unique()
        if target_n > len(traj_ids):
            raise ValueError(f"{split_name}: need {target_n} trajectories, only have {len(traj_ids)}")
        keep = rng.choice(traj_ids, size=target_n, replace=False)
        kept_frames.append(split_frame[split_frame["trajectory_id"].isin(keep)])
    return pd.concat(kept_frames, ignore_index=True)


def main() -> None:
    downsampled = build_downsampled_formality()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    downsampled.to_json(OUT_PATH, orient="records", lines=True)
    counts = downsampled.groupby("topic_split")["trajectory_id"].nunique().to_dict()
    print(f"[wrote] {OUT_PATH} split_traj_counts={counts}")

    frame = downsampled.copy()
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
            "rollout_mse": rollout_summary["rollout_mse"],
            "rollout_n": rollout_summary["n"],
        }

    run_name = "formality_t5_downsampled-memory-lag3-linear_ridge-k0-seed0"
    result_dir = PACKAGE_ROOT / "results" / run_name
    result_dir.mkdir(parents=True, exist_ok=True)
    run_metadata = {
        "run_name": run_name,
        "dataset": str(OUT_PATH),
        "downsample_seed": DOWNSAMPLE_SEED,
        "split_traj_counts": counts,
        "model": "AugmentedKoopmanModel (pure affine, no AE lift)",
        "metrics": metrics,
    }
    (result_dir / "run.json").write_text(json.dumps(run_metadata, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    print(f"[done] result_dir={result_dir}")


if __name__ == "__main__":
    main()
