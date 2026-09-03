#!/usr/bin/env python3
"""Train and evaluate the standalone controlled Autoencoder--Koopman model.

Configuration is composed by Hydra from `configs/config.yaml` and its
`dataset/state/model/trainer` groups. Swap a whole group from the command
line (`dataset=character_length_t5`, `state=augmented`) or override a single
field (`model.latent_dim=32`, `trainer.epochs=400`, `trainer.device=cuda`).
See `configs/` for the available groups and `README.md` for examples.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import re
import sys
from typing import Any

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from koopman_ae import (
    AugmentedStateConfig,
    DeepAugmentedKoopmanAutoencoder,
    DeepAugmentedKoopmanConfig,
    augmented_prediction_metrics,
    build_augmented_state_dataset,
    build_augmented_state_sequences,
    one_step_predictions,
    rollout_augmented_from_trajectories,
    rollout_metrics,
)


PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[1]
CONFIG_DIR = PACKAGE_ROOT / "configs"


def _load_table(path: pathlib.Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"unsupported trajectory format: {path.suffix}")


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clean_trajectories(
    frame: pd.DataFrame,
    output_columns: tuple[str, ...],
    target_columns: tuple[str, ...],
) -> pd.DataFrame:
    required = {
        "trajectory_id",
        "topic_split",
        "turn",
        *output_columns,
        *target_columns,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"dataset is missing required columns: {missing}")

    cleaned = frame.copy()
    numeric_columns = ["turn", *output_columns, *target_columns]
    for column in numeric_columns:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    before = len(cleaned)
    cleaned = cleaned.dropna(subset=numeric_columns + ["trajectory_id", "topic_split"])
    cleaned = cleaned[np.isfinite(cleaned[numeric_columns]).all(axis=1)].copy()
    cleaned["turn"] = cleaned["turn"].astype(int)
    cleaned["trajectory_id"] = cleaned["trajectory_id"].astype(str)
    cleaned["topic_split"] = cleaned["topic_split"].astype(str)
    duplicate = cleaned.duplicated(["trajectory_id", "turn"])
    if duplicate.any():
        examples = cleaned.loc[duplicate, ["trajectory_id", "turn"]].head().to_dict("records")
        raise ValueError(f"duplicate trajectory/turn rows found: {examples}")
    if cleaned.empty:
        raise ValueError("no finite trajectory rows remain after validation")
    if len(cleaned) != before:
        print(f"[data] dropped {before - len(cleaned)} rows with missing/non-finite model fields")
    return cleaned.sort_values(["trajectory_id", "turn"]).reset_index(drop=True)


def _state_config(
    family: str,
    lag: int,
    control_mode: str,
    output_columns: tuple[str, ...],
    target_columns: tuple[str, ...],
) -> AugmentedStateConfig:
    """The ONE place `state.lag` becomes `output_memory`/`input_memory`.

    `lag` is a lag DEPTH (how many turns back to reach); the dataclass fields
    are TERM COUNTS (how many entries land in z). They differ by one, so
    `lag=3` -- the repo default in configs/state/memory.yaml -- means a
    four-term embedding `z_t = [y_t, y_(t-1), y_(t-2), y_(t-3)]`, not three.
    Every run name, results/ directory and run.json records the `lag` number
    (`-lag3-`), while the model and every error message report the term
    counts, so the two conventions are both visible in one run's artifacts
    and are off by one from each other on purpose. Keep the +1 here rather
    than pushing it into AugmentedStateConfig: changing which of the two a
    given number means would silently re-point every existing results/
    directory name at a different state definition.

    `markov` pins (1, 0) and ignores `lag` entirely -- configs/state/markov.yaml
    sets `lag: 0` only so the three families share one schema.

    Raising `lag` costs rollout horizon: build/rollout both need
    `max(output_memory, input_memory)` seed turns, so on a T=5 dataset
    `lag=3` leaves exactly one rollout step (ABLATION_STUDY.md phase 6).
    """

    if lag < 0:
        raise ValueError("state.lag must be non-negative")
    if family == "markov":
        output_memory, input_memory = 1, 0
    elif family == "memory":
        output_memory, input_memory = lag + 1, 0
    else:
        output_memory = input_memory = lag + 1
    return AugmentedStateConfig(
        output_memory=output_memory,
        input_memory=input_memory,
        control_mode=control_mode,
        output_columns=output_columns,
        target_columns=target_columns,
    )


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")


def _resolve_dataset_path(dataset_cfg: DictConfig, data_root: pathlib.Path) -> pathlib.Path:
    path = pathlib.Path(dataset_cfg.path)
    if not path.is_absolute():
        path = (data_root / path).resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"trajectory dataset not found: {path}. See DATASETS.md for setup."
        )
    return path


def _split(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    return frame[frame["topic_split"].eq(name)].copy()


def _evaluate(
    model: DeepAugmentedKoopmanAutoencoder,
    frame: pd.DataFrame,
    cfg: AugmentedStateConfig,
    common_seed_turns: int,
) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    for split in ("train", "validation", "test"):
        split_frame = _split(frame, split)
        if split_frame.empty:
            continue
        dataset = build_augmented_state_dataset(split_frame, cfg)
        one_step = one_step_predictions(model, dataset)
        rollout = rollout_augmented_from_trajectories(
            model,
            split_frame,
            cfg,
            observed_seed_turns=common_seed_turns,
        )
        one_step_summary = augmented_prediction_metrics(one_step)
        rollout_summary = rollout_metrics(rollout)
        results[split] = {
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
    return results


def _latest_checkpoint(checkpoint_dir: pathlib.Path) -> pathlib.Path | None:
    complete = [
        path
        for path in checkpoint_dir.glob("checkpoint-*")
        if (path / "state.pt").is_file() and (path / "_COMPLETE").is_file()
    ]
    return max(complete, default=None)


@hydra.main(config_path=str(CONFIG_DIR), config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    dataset_cfg = cfg.dataset
    state_cfg = cfg.state
    model_cfg = cfg.model
    trainer_cfg = cfg.trainer

    data_root = (
        pathlib.Path(trainer_cfg.data_root).resolve()
        if trainer_cfg.data_root
        else PACKAGE_ROOT / "datasets"
    )
    dataset_path = _resolve_dataset_path(dataset_cfg, data_root)

    output_columns = tuple(dataset_cfg.output_columns)
    target_columns = tuple(dataset_cfg.target_columns)
    frame = _clean_trajectories(_load_table(dataset_path), output_columns, target_columns)

    lag = int(state_cfg.lag)
    state = _state_config(state_cfg.family, lag, state_cfg.control_mode, output_columns, target_columns)

    minimum_seed = max(state.output_memory, state.input_memory)
    common_seed_turns = int(
        trainer_cfg.common_seed_turns
        if trainer_cfg.common_seed_turns is not None
        else (
            dataset_cfg.common_seed_turns
            if dataset_cfg.get("common_seed_turns") is not None
            else minimum_seed
        )
    )
    if common_seed_turns < minimum_seed:
        raise ValueError(
            f"common seed {common_seed_turns} is shorter than the required state history "
            f"({minimum_seed} turns)"
        )
    if common_seed_turns >= int(frame["turn"].max()):
        raise ValueError("common seed must be shorter than the trajectory horizon")

    train_frame = _split(frame, "train")
    if train_frame.empty:
        raise ValueError("dataset has no rows with topic_split='train'")
    train_dataset = build_augmented_state_dataset(train_frame, state)
    multi_step_requested = model_cfg.training_mode == "joint" and model_cfg.lambda_multi > 0
    sequences = build_augmented_state_sequences(train_frame, state) if multi_step_requested else None

    validation_frame = _split(frame, "validation")
    early_stopping_requested = trainer_cfg.early_stopping_patience is not None
    if early_stopping_requested and validation_frame.empty:
        raise ValueError(
            "trainer.early_stopping_patience is set but the dataset has no "
            "topic_split='validation' rows to early-stop against"
        )
    validation_dataset = (
        build_augmented_state_dataset(validation_frame, state)
        if early_stopping_requested
        else None
    )
    # The validation-split counterpart of `sequences`, under the same guard:
    # with `lambda_multi > 0` the stopping signal has to include the
    # multi-step rollout term the training steps include, or `best_state` is
    # chosen for one-step accuracy while the run is scored on rollout_mse.
    # See DeepAugmentedKoopmanAutoencoder._eval_loss.
    validation_sequences = (
        build_augmented_state_sequences(validation_frame, state)
        if multi_step_requested and validation_dataset is not None
        else None
    )

    run_name = _safe_name(
        trainer_cfg.run_name
        or (
            f"{dataset_cfg.name}-{state_cfg.family}-lag{lag}-"
            f"{model_cfg.training_mode}-k{model_cfg.latent_dim}-seed{trainer_cfg.seed}"
        )
    )
    checkpoint_dir = (
        pathlib.Path(trainer_cfg.checkpoint_dir).resolve()
        if trainer_cfg.checkpoint_dir
        else pathlib.Path(trainer_cfg.checkpoint_root) / run_name
    )
    result_dir = (
        pathlib.Path(trainer_cfg.result_dir).resolve()
        if trainer_cfg.result_dir
        else PACKAGE_ROOT / "results" / run_name
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    dataset_hash = _sha256(dataset_path)
    resume_spec = {
        "dataset": str(dataset_path),
        "dataset_sha256": dataset_hash,
        "state_family": state_cfg.family,
        "lag": lag,
        "control_mode": state_cfg.control_mode,
        "output_columns": list(output_columns),
        "target_columns": list(target_columns),
    }
    resume_spec_path = checkpoint_dir / "run_spec.json"
    if resume_spec_path.is_file():
        previous_spec = json.loads(resume_spec_path.read_text())
        if previous_spec != resume_spec:
            raise ValueError(
                "checkpoint directory belongs to a different dataset/state definition; "
                f"existing={previous_spec}, current={resume_spec}"
            )
    else:
        resume_spec_path.write_text(json.dumps(resume_spec, indent=2) + "\n")
    if trainer_cfg.no_resume and _latest_checkpoint(checkpoint_dir) is not None:
        raise ValueError(
            "trainer.no_resume requires an empty/new checkpoint directory; use a new "
            "trainer.run_name or trainer.checkpoint_dir"
        )

    model_config = DeepAugmentedKoopmanConfig(
        latent_dim=model_cfg.latent_dim,
        hidden_dim=model_cfg.hidden_dim,
        num_layers=model_cfg.num_layers,
        activation=model_cfg.activation,
        learning_rate=trainer_cfg.learning_rate,
        weight_decay=trainer_cfg.weight_decay,
        batch_size=trainer_cfg.batch_size,
        num_epochs=trainer_cfg.epochs,
        lambda_rec=model_cfg.lambda_rec,
        lambda_pred=model_cfg.lambda_pred,
        lambda_latent=model_cfg.lambda_latent,
        lambda_multi=model_cfg.lambda_multi,
        multi_step_horizon=model_cfg.multi_step_horizon,
        training_mode=model_cfg.training_mode,
        dynamics_alpha=model_cfg.dynamics_alpha,
        random_state=trainer_cfg.seed,
        device=trainer_cfg.device,
        early_stopping_patience=trainer_cfg.early_stopping_patience,
        early_stopping_min_delta=trainer_cfg.early_stopping_min_delta,
    )
    model = DeepAugmentedKoopmanAutoencoder(
        state_dim=train_dataset.state_dim,
        target_dim=train_dataset.target_dim,
        output_dim=train_dataset.output_dim,
        config=model_config,
        name=run_name,
    )
    print(f"[data] {dataset_path}")
    print(f"[data] rows={len(frame)} trajectories={frame['trajectory_id'].nunique()}")
    print(
        f"[fit] family={state_cfg.family} state_dim={train_dataset.state_dim} "
        f"latent_dim={model_cfg.latent_dim} mode={model_cfg.training_mode} device={model.device}"
    )
    print(f"[fit] checkpoint_dir={checkpoint_dir}")
    try:
        model.fit(
            train_dataset.Z_t,
            train_dataset.R,
            train_dataset.Z_next,
            multi_step_sequences=sequences,
            checkpoint_dir=checkpoint_dir,
            checkpoint_every_epochs=trainer_cfg.checkpoint_every_epochs,
            resume=not trainer_cfg.no_resume,
            Z_val=validation_dataset.Z_t if validation_dataset is not None else None,
            R_val=validation_dataset.R if validation_dataset is not None else None,
            Z_next_val=validation_dataset.Z_next if validation_dataset is not None else None,
            multi_step_sequences_val=validation_sequences,
        )
    except InterruptedError as exc:
        latest = _latest_checkpoint(checkpoint_dir)
        print(f"[stop] {exc}", file=sys.stderr)
        print(f"[stop] latest_resumable_checkpoint={latest}", file=sys.stderr)
        sys.exit(130)

    horizon = int(frame["turn"].max())
    metrics = _evaluate(model, frame, state, common_seed_turns)
    diagnostics = model.diagnostics(horizon=horizon)
    run_metadata: dict[str, Any] = {
        "run_name": run_name,
        "dataset": str(dataset_path),
        "dataset_sha256": dataset_hash,
        "rows": int(len(frame)),
        "trajectories": int(frame["trajectory_id"].nunique()),
        "splits": {key: int(value) for key, value in frame["topic_split"].value_counts().items()},
        "state_family": state_cfg.family,
        "lag": lag,
        "output_columns": list(output_columns),
        "target_columns": list(target_columns),
        "common_seed_turns": common_seed_turns,
        "checkpoint_dir": str(checkpoint_dir),
        "latest_resumable_checkpoint": str(_latest_checkpoint(checkpoint_dir)),
        "config": OmegaConf.to_container(cfg, resolve=True),
        "metrics": metrics,
    }
    (result_dir / "run.json").write_text(json.dumps(run_metadata, indent=2) + "\n")
    (result_dir / "koopman_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2) + "\n"
    )
    print(json.dumps(metrics, indent=2))
    print(f"[done] result_dir={result_dir}")
    print(f"[done] latest_resumable_checkpoint={_latest_checkpoint(checkpoint_dir)}")


if __name__ == "__main__":
    main()
