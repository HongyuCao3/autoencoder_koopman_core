# Code Design Document

## Purpose

This directory is a standalone, runnable extraction of the controlled Deep Autoencoder--Koopman implementation and its canonical collected trajectory datasets. It intentionally excludes experiment-specific plotting, report generation, model caches, and historical checkpoints.

## System Overview

The package converts turn-indexed control trajectories into Markov or delay-embedded states, learns an autoencoder representation, advances that representation with affine controlled linear dynamics, and evaluates one-step and recursive rollout errors.

The learned equations are:

```text
xi_t     = E(z_t)
xi_t+1   = K xi_t + B r + c
z_hat_t  = D(xi_t)
y_hat_t  = z_hat_t[:output_dim]
```

Two optimization modes share the same architecture:

- `joint` optimizes reconstruction, decoded one-step prediction, latent linearity, and optional recursive multi-step losses together.
- `reconstruction_then_ridge` optimizes only AE reconstruction and then fits `K`, `B`, and `c` by ridge regression with an unregularized intercept.

## Repository Structure

- `src/koopman_ae/core.py`: state construction, affine baseline, deep AE--Koopman model, training, exact-resume checkpoints, rollout, and diagnostics.
- `src/koopman_ae/__init__.py`: stable public imports.
- `scripts/train.py`: Hydra entry point — dataset resolution, validation, training, evaluation, and lightweight artifact writing.
- `configs/`: Hydra configuration groups (`dataset/`, `state/`, `model/`, `trainer/`) plus the top-level `config.yaml` defaults list. `configs/dataset/*.yaml` holds the canonical dataset path and column contracts, one file per registered dataset.
- `datasets/`: repository-local copies of the eight canonical raw trajectory files so a clone is immediately runnable.
- `DATASET_MANIFEST.csv`: immutable size/count/hash inventory.
- `tests/test_core.py`: interface, training-mode, incomplete-checkpoint, and exact-resume tests.

## Core Components

### State construction

`AugmentedStateConfig` defines output history, control history, control transformation, output columns, and target columns. `build_augmented_state_dataset()` emits `(Z_t, R, Z_next)` transition arrays plus row metadata. `build_augmented_state_sequences()` emits full state sequences for optional multi-step joint loss.

For output dimension `d_y` and maximum lag `L`:

- Markov state dimension: `d_y`.
- Memory-L state dimension: `(L + 1) * d_y`.
- Augmented-L with error control: `(L + 1) * (d_y + d_y)`.
- `error_abs_sign` triples the control feature dimension.

The default control is `u_t = r[:d_y] - y_t`. Additional target/context dimensions may enter through `B` but do not define feedback error.

### Deep model

`DeepAugmentedKoopmanAutoencoder` owns the MLP encoder/decoder and trainable `K`, `B`, and `c`. The decoder reconstructs the complete delay state; `predict_y()` exposes its leading output block. `diagnostics()` exports matrices, eigenvalues, spectral radius, finite-horizon controllability matrix/singular values, and Gramian statistics.

### Exact-resume training

Custom AdamW training checkpoints contain model tensors, optimizer state, all relevant RNG states, DataLoader generator state, completed epoch, training history, dimensions, mode, and configuration. There is no learning-rate scheduler or AMP scaler in this implementation, so neither has runtime state to save.

Checkpoint directories are written through a temporary directory and atomically renamed only after `state.pt` and `_COMPLETE` exist. Resume scans only complete checkpoints, reports incomplete ones, validates dimensions/mode/configuration, and permits only `num_epochs` and `device` to differ. `scripts/train.py` additionally pins the dataset SHA-256 and state definition in `run_spec.json`.

`SIGTERM` and `SIGINT` request a checkpoint after the current epoch. The CLI reports the latest resumable checkpoint and exits with status 130.

## Data Flow and Control Flow

```text
dataset registry or explicit path
  -> JSONL/CSV/Parquet DataFrame
  -> schema/numeric/duplicate validation
  -> trajectory-level train/validation/test frames
  -> delay-state transition tensors
  -> AE--Koopman fit on train only
  -> one-step and common-prefix recursive rollout evaluation
  -> compact metrics and matrix diagnostics
```

The existing `topic_split` is authoritative. The CLI never performs row-level random splitting, avoiding leakage between turns of the same trajectory.

## Interfaces and Contracts

Every trajectory table must contain:

- `trajectory_id`: stable trajectory identifier;
- `topic_split`: normally `train`, `validation`, or `test`;
- `turn`: unique integer index within a trajectory;
- all configured output and target columns, finite and numeric.

Turns should be consecutive and start at 1. State construction skips transitions whose complete history is unavailable. The common observed seed must be at least `max(output_memory, input_memory)` and smaller than the maximum turn.

Canonical scalar datasets use `normalized_output` and `effective_norm`. Vector Stage 1 uses two count outputs/targets; Stage 2 uses three. Exact mappings are in `configs/datasets.json`.

## Configuration and Runtime Assumptions

Source, documentation, manifests, raw datasets, and compact JSON results live in the repository checkout itself so the directory can be published as one Git repository. This repository-local dataset placement is an explicit portability exception to the normal scratch-storage policy. Checkpoints default to `/scratch/$USER/checkpoints/autoencoder_koopman_core` (`trainer.checkpoint_root`, derived from `$USER`) and are excluded from Git. Slurm submission scripts for Palmetto 2 live in `slurm/`.

The package requires Python 3.10+, NumPy, pandas, and PyTorch. CPU is supported. CUDA is selected automatically when available unless `--device` is specified.

## Commands and Workflows

Install:

```bash
python -m pip install -e '.[test]'
```

Train the default two-stage sentence-length Memory-L3 model (this is also the config default, so a bare `python scripts/train.py` runs it):

```bash
python scripts/train.py \
  dataset=sentence_length_t10 \
  state=memory state.lag=3 \
  model.training_mode=reconstruction_then_ridge model.latent_dim=16 \
  trainer.epochs=200
```

`python scripts/train.py --help` lists the available `dataset/state/model/trainer` groups and prints the fully-resolved config.

Run validation:

```bash
pytest -q
```

## Tests and Validation

`tests/test_core.py` checks both training modes, finite one-step predictions, checkpoint completeness filtering, and numerical equality between uninterrupted six-epoch training and a three-plus-three exact-resume run.

The collected files are byte-identical to their source trajectory artifacts; `DATASET_MANIFEST.csv` records their SHA-256 hashes.

## Known Limitations and Risks

- Raw latent eigenvectors are basis-dependent because the autoencoder representation is not unique.
- Full controllability rank does not imply a well-conditioned controllability Gramian.
- T=5 datasets provide only one forecast turn under the default four-turn common prefix.
- The even/odd scalar encoding is a categorical interface/negative control, not strong operator evidence.
- Sentiment and formality conclusions remain limited by scorer/readout quality.
- The two-stage checkpoint stores the resumable AE optimization state before the deterministic final ridge solve; invoking the same completed command reloads the AE state and recomputes `K/B/c`.
- The CLI writes compact summaries only. Large prediction tables should remain under `/scratch` if added later.

## Extension Points

- Add a `configs/dataset/<name>.yaml` entry for another trajectory file without changing the core.
- Pass multi-output columns to train vector systems.
- Add task/context target columns so they enter the learned `B` matrix.
- Add state families by mapping them to output/control memory in `_state_config()`.
- Add compact spectral plotting based on `koopman_diagnostics.json` without coupling plots to training.

## Recent Changes

- 2026-08-27: Replaced the argparse CLI and `configs/datasets.json` registry with a Hydra-composed config layer (`configs/{dataset,state,model,trainer}/*.yaml` + `configs/config.yaml`). `core.py`'s public API and the custom exact-resume training loop are unchanged; this only reorganizes how `scripts/train.py` reads its parameters.
- 2026-08-26: Replaced the scratch-backed dataset symlink with repository-local data copies and added GitHub publication guidance. All current files remain below GitHub's regular-Git warning threshold.
- 2026-08-26: Extracted the current Model III AE--Koopman core, added a single runnable CLI, organized eight canonical raw datasets, and strengthened exact-resume validation for checkpoint completeness, configuration, and dataset identity.
