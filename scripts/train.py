#!/usr/bin/env python3
"""Train and evaluate the standalone controlled Autoencoder--Koopman model."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import sys
from typing import Any

import numpy as np
import pandas as pd

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
REGISTRY_PATH = PACKAGE_ROOT / "configs" / "datasets.json"
DEFAULT_DATA_ROOT = PACKAGE_ROOT / "datasets"
DEFAULT_CHECKPOINT_ROOT = pathlib.Path(
    "/scratch/ruimind/checkpoints/idea-LLMControl/autoencoder_koopman_core"
)


def _load_registry() -> dict[str, dict[str, Any]]:
    return json.loads(REGISTRY_PATH.read_text())


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
    if lag < 0:
        raise ValueError("--lag must be non-negative")
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


def _resolve_dataset(args: argparse.Namespace) -> tuple[pathlib.Path, dict[str, Any], str]:
    registry = _load_registry()
    if args.dataset_key:
        spec = registry[args.dataset_key]
        path = (args.data_root / spec["path"]).resolve()
        label = args.dataset_key
    else:
        spec = {}
        path = args.dataset.resolve()
        label = path.stem
    if not path.is_file():
        raise FileNotFoundError(
            f"trajectory dataset not found: {path}. See DATASETS.md for setup."
        )
    return path, spec, label


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


def build_parser() -> argparse.ArgumentParser:
    registry_keys = sorted(_load_registry())
    parser = argparse.ArgumentParser(
        description="Train a controlled Autoencoder--Koopman model on collected trajectories."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset-key", choices=registry_keys)
    source.add_argument("--dataset", type=pathlib.Path)
    parser.add_argument("--data-root", type=pathlib.Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-columns", nargs="+", default=None)
    parser.add_argument("--target-columns", nargs="+", default=None)
    parser.add_argument(
        "--state-family", choices=["markov", "memory", "augmented"], default="memory"
    )
    parser.add_argument("--lag", type=int, default=3)
    parser.add_argument(
        "--training-mode",
        choices=["joint", "reconstruction_then_ridge"],
        default="reconstruction_then_ridge",
    )
    parser.add_argument("--control-mode", choices=["error", "error_abs_sign"], default="error")
    parser.add_argument("--common-seed-turns", type=int, default=None)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--activation", choices=["tanh", "relu", "gelu", "silu"], default="tanh")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lambda-rec", type=float, default=1.0)
    parser.add_argument("--lambda-pred", type=float, default=1.0)
    parser.add_argument("--lambda-latent", type=float, default=0.1)
    parser.add_argument("--lambda-multi", type=float, default=0.0)
    parser.add_argument("--multi-step-horizon", type=int, default=0)
    parser.add_argument("--dynamics-alpha", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None, help="cpu, cuda, cuda:0, ...")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--checkpoint-dir", type=pathlib.Path, default=None)
    parser.add_argument("--checkpoint-every-epochs", type=int, default=20)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--result-dir", type=pathlib.Path, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    dataset_path, dataset_spec, dataset_label = _resolve_dataset(args)
    output_columns = tuple(
        args.output_columns or dataset_spec.get("output_columns", ["normalized_output"])
    )
    target_columns = tuple(
        args.target_columns or dataset_spec.get("target_columns", ["effective_norm"])
    )
    frame = _clean_trajectories(
        _load_table(dataset_path),
        output_columns,
        target_columns,
    )
    cfg = _state_config(
        args.state_family,
        args.lag,
        args.control_mode,
        output_columns,
        target_columns,
    )
    minimum_seed = max(cfg.output_memory, cfg.input_memory)
    common_seed_turns = int(
        args.common_seed_turns
        if args.common_seed_turns is not None
        else dataset_spec.get("common_seed_turns", minimum_seed)
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
    train_dataset = build_augmented_state_dataset(train_frame, cfg)
    sequences = (
        build_augmented_state_sequences(train_frame, cfg)
        if args.training_mode == "joint" and args.lambda_multi > 0
        else None
    )
    run_name = _safe_name(
        args.run_name
        or (
            f"{dataset_label}-{args.state_family}-lag{args.lag}-"
            f"{args.training_mode}-k{args.latent_dim}-seed{args.seed}"
        )
    )
    checkpoint_dir = (
        args.checkpoint_dir.resolve()
        if args.checkpoint_dir
        else DEFAULT_CHECKPOINT_ROOT / run_name
    )
    result_dir = (
        args.result_dir.resolve()
        if args.result_dir
        else PACKAGE_ROOT / "results" / run_name
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    dataset_hash = _sha256(dataset_path)
    resume_spec = {
        "dataset": str(dataset_path),
        "dataset_sha256": dataset_hash,
        "state_family": args.state_family,
        "lag": int(args.lag),
        "control_mode": args.control_mode,
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
    if args.no_resume and _latest_checkpoint(checkpoint_dir) is not None:
        raise ValueError(
            "--no-resume requires an empty/new checkpoint directory; use a new "
            "--run-name or --checkpoint-dir"
        )

    model_cfg = DeepAugmentedKoopmanConfig(
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        activation=args.activation,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        lambda_rec=args.lambda_rec,
        lambda_pred=args.lambda_pred,
        lambda_latent=args.lambda_latent,
        lambda_multi=args.lambda_multi,
        multi_step_horizon=args.multi_step_horizon,
        training_mode=args.training_mode,
        dynamics_alpha=args.dynamics_alpha,
        random_state=args.seed,
        device=args.device,
    )
    model = DeepAugmentedKoopmanAutoencoder(
        state_dim=train_dataset.state_dim,
        target_dim=train_dataset.target_dim,
        output_dim=train_dataset.output_dim,
        config=model_cfg,
        name=run_name,
    )
    print(f"[data] {dataset_path}")
    print(f"[data] rows={len(frame)} trajectories={frame['trajectory_id'].nunique()}")
    print(
        f"[fit] family={args.state_family} state_dim={train_dataset.state_dim} "
        f"latent_dim={args.latent_dim} mode={args.training_mode} device={model.device}"
    )
    print(f"[fit] checkpoint_dir={checkpoint_dir}")
    try:
        model.fit(
            train_dataset.Z_t,
            train_dataset.R,
            train_dataset.Z_next,
            multi_step_sequences=sequences,
            checkpoint_dir=checkpoint_dir,
            checkpoint_every_epochs=args.checkpoint_every_epochs,
            resume=not args.no_resume,
        )
    except InterruptedError as exc:
        latest = _latest_checkpoint(checkpoint_dir)
        print(f"[stop] {exc}", file=sys.stderr)
        print(f"[stop] latest_resumable_checkpoint={latest}", file=sys.stderr)
        return 130

    horizon = int(frame["turn"].max())
    metrics = _evaluate(model, frame, cfg, common_seed_turns)
    diagnostics = model.diagnostics(horizon=horizon)
    run_metadata = {
        "run_name": run_name,
        "dataset": str(dataset_path),
        "dataset_sha256": dataset_hash,
        "rows": int(len(frame)),
        "trajectories": int(frame["trajectory_id"].nunique()),
        "splits": {key: int(value) for key, value in frame["topic_split"].value_counts().items()},
        "state_family": args.state_family,
        "lag": int(args.lag),
        "output_columns": list(output_columns),
        "target_columns": list(target_columns),
        "common_seed_turns": common_seed_turns,
        "checkpoint_dir": str(checkpoint_dir),
        "latest_resumable_checkpoint": str(_latest_checkpoint(checkpoint_dir)),
        "metrics": metrics,
    }
    (result_dir / "run.json").write_text(json.dumps(run_metadata, indent=2) + "\n")
    (result_dir / "koopman_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2) + "\n"
    )
    print(json.dumps(metrics, indent=2))
    print(f"[done] result_dir={result_dir}")
    print(f"[done] latest_resumable_checkpoint={_latest_checkpoint(checkpoint_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
