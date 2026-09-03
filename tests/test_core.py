from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from koopman_ae import (
    AugmentedStateConfig,
    DeepAugmentedKoopmanAutoencoder,
    DeepAugmentedKoopmanConfig,
    build_augmented_state_dataset,
    build_augmented_state_sequences,
    one_step_predictions,
)


def _toy_trajectories() -> pd.DataFrame:
    rows = []
    for trajectory, split, target, start in [
        ("train-a", "train", 0.2, 0.8),
        ("train-b", "train", 0.8, 0.1),
        ("test-a", "test", 0.5, 0.9),
    ]:
        value = start
        for turn in range(1, 7):
            rows.append(
                {
                    "trajectory_id": trajectory,
                    "topic": trajectory,
                    "topic_split": split,
                    "turn": turn,
                    "normalized_output": value,
                    "effective_norm": target,
                }
            )
            value = 0.65 * value + 0.35 * target
    return pd.DataFrame(rows)


@pytest.mark.parametrize("training_mode", ["joint", "reconstruction_then_ridge"])
def test_deep_koopman_training_modes_run(training_mode: str) -> None:
    pytest.importorskip("torch")
    state_cfg = AugmentedStateConfig(output_memory=2, input_memory=0)
    dataset = build_augmented_state_dataset(
        _toy_trajectories().query("topic_split == 'train'"), state_cfg
    )
    model = DeepAugmentedKoopmanAutoencoder(
        state_dim=dataset.state_dim,
        target_dim=dataset.target_dim,
        output_dim=dataset.output_dim,
        config=DeepAugmentedKoopmanConfig(
            latent_dim=3,
            hidden_dim=8,
            num_layers=1,
            num_epochs=3,
            batch_size=4,
            training_mode=training_mode,
            random_state=3,
            device="cpu",
        ),
    ).fit(dataset.Z_t, dataset.R, dataset.Z_next)
    predictions = one_step_predictions(model, dataset)
    assert np.isfinite(predictions["y_mae"]).all()
    assert model.diagnostics(horizon=3)["training_mode"] == training_mode


def test_exact_resume_matches_uninterrupted_training(tmp_path) -> None:
    pytest.importorskip("torch")
    state_cfg = AugmentedStateConfig(output_memory=2, input_memory=2)
    dataset = build_augmented_state_dataset(
        _toy_trajectories().query("topic_split == 'train'"), state_cfg
    )
    common = dict(
        latent_dim=3,
        hidden_dim=8,
        num_layers=1,
        batch_size=4,
        random_state=7,
        device="cpu",
    )
    uninterrupted = DeepAugmentedKoopmanAutoencoder(
        dataset.state_dim,
        dataset.target_dim,
        dataset.output_dim,
        DeepAugmentedKoopmanConfig(num_epochs=6, **common),
    ).fit(dataset.Z_t, dataset.R, dataset.Z_next)

    checkpoint_dir = tmp_path / "checkpoints"
    DeepAugmentedKoopmanAutoencoder(
        dataset.state_dim,
        dataset.target_dim,
        dataset.output_dim,
        DeepAugmentedKoopmanConfig(num_epochs=3, **common),
    ).fit(
        dataset.Z_t,
        dataset.R,
        dataset.Z_next,
        checkpoint_dir=checkpoint_dir,
        checkpoint_every_epochs=3,
    )
    resumed = DeepAugmentedKoopmanAutoencoder(
        dataset.state_dim,
        dataset.target_dim,
        dataset.output_dim,
        DeepAugmentedKoopmanConfig(num_epochs=6, **common),
    ).fit(
        dataset.Z_t,
        dataset.R,
        dataset.Z_next,
        checkpoint_dir=checkpoint_dir,
        checkpoint_every_epochs=3,
    )
    assert (checkpoint_dir / "checkpoint-000006" / "_COMPLETE").is_file()
    assert np.allclose(uninterrupted.K.detach().numpy(), resumed.K.detach().numpy())
    assert np.allclose(
        uninterrupted.predict_next_z(dataset.Z_t, dataset.R),
        resumed.predict_next_z(dataset.Z_t, dataset.R),
    )


def test_incomplete_checkpoint_is_ignored(tmp_path, capsys) -> None:
    checkpoint = tmp_path / "checkpoint-000999"
    checkpoint.mkdir()
    assert DeepAugmentedKoopmanAutoencoder._latest_complete_checkpoint(tmp_path) is None
    assert "skipping incomplete checkpoint" in capsys.readouterr().out


def _multi_step_model(**config_overrides) -> DeepAugmentedKoopmanAutoencoder:
    state_cfg = AugmentedStateConfig(output_memory=2, input_memory=0)
    dataset = build_augmented_state_dataset(
        _toy_trajectories().query("topic_split == 'train'"), state_cfg
    )
    config = DeepAugmentedKoopmanConfig(
        latent_dim=3,
        hidden_dim=8,
        num_layers=1,
        num_epochs=2,
        batch_size=4,
        random_state=3,
        device="cpu",
        **{"training_mode": "joint", **config_overrides},
    )
    return DeepAugmentedKoopmanAutoencoder(
        state_dim=dataset.state_dim,
        target_dim=dataset.target_dim,
        output_dim=dataset.output_dim,
        config=config,
    )


def test_eval_loss_includes_the_weighted_multi_step_term_when_sequences_are_given() -> None:
    # The reason `multi_step_sequences_val` exists: with lambda_multi > 0 the
    # training steps descend a multi-step rollout term, so a stopping signal
    # that omits it selects `best_state` for one-step accuracy while the run
    # is judged on rollout_mse.
    pytest.importorskip("torch")
    import torch.nn as nn

    state_cfg = AugmentedStateConfig(output_memory=2, input_memory=0)
    frame = _toy_trajectories().query("topic_split == 'train'")
    dataset = build_augmented_state_dataset(frame, state_cfg)
    sequences = build_augmented_state_sequences(frame, state_cfg)
    assert sequences, "toy trajectories must be long enough to form rollout sequences"

    model = _multi_step_model(lambda_multi=0.5, multi_step_horizon=2)
    z = model._tensor(dataset.Z_t.astype("float32"))
    r = model._tensor(dataset.R.astype("float32"))
    z_next = model._tensor(dataset.Z_next.astype("float32"))
    mse = nn.MSELoss()

    without = model._eval_loss(z, r, z_next, mse)
    with_multi = model._eval_loss(z, r, z_next, mse, sequences)
    multi = float(model._multi_step_loss(sequences, mse).detach().cpu().item())

    assert with_multi != without
    assert with_multi == pytest.approx(without + 0.5 * multi, rel=1e-6)


def test_eval_loss_ignores_sequences_when_the_multi_step_term_is_off() -> None:
    # Guard for every run recorded before this parameter existed: they used
    # lambda_multi=0 (or reconstruction_then_ridge), where the training steps
    # have no multi-step term either, so the stopping signal must not change.
    pytest.importorskip("torch")
    import torch.nn as nn

    state_cfg = AugmentedStateConfig(output_memory=2, input_memory=0)
    frame = _toy_trajectories().query("topic_split == 'train'")
    dataset = build_augmented_state_dataset(frame, state_cfg)
    sequences = build_augmented_state_sequences(frame, state_cfg)
    mse = nn.MSELoss()

    for overrides in (
        {"lambda_multi": 0.0},  # joint, the mode that HAS a multi-step branch
        {"training_mode": "reconstruction_then_ridge", "lambda_multi": 0.5},
    ):
        model = _multi_step_model(**overrides)
        z = model._tensor(dataset.Z_t.astype("float32"))
        r = model._tensor(dataset.R.astype("float32"))
        z_next = model._tensor(dataset.Z_next.astype("float32"))
        assert model._eval_loss(z, r, z_next, mse, sequences) == model._eval_loss(
            z, r, z_next, mse
        )


def test_fit_rejects_validation_sequences_without_a_validation_split() -> None:
    pytest.importorskip("torch")
    state_cfg = AugmentedStateConfig(output_memory=2, input_memory=0)
    frame = _toy_trajectories().query("topic_split == 'train'")
    dataset = build_augmented_state_dataset(frame, state_cfg)
    sequences = build_augmented_state_sequences(frame, state_cfg)
    model = _multi_step_model(lambda_multi=0.5, multi_step_horizon=2)
    with pytest.raises(ValueError, match="multi_step_sequences_val"):
        model.fit(
            dataset.Z_t,
            dataset.R,
            dataset.Z_next,
            multi_step_sequences_val=sequences,
        )


def test_training_history_logs_the_weighted_multi_step_loss() -> None:
    # The logged number used to be the raw multi_loss while the gradient used
    # lambda_multi * multi_loss, making training_history_ a third quantity
    # comparable to neither the objective nor the validation curve.
    pytest.importorskip("torch")
    import torch.nn as nn

    state_cfg = AugmentedStateConfig(output_memory=2, input_memory=0)
    frame = _toy_trajectories().query("topic_split == 'train'")
    dataset = build_augmented_state_dataset(frame, state_cfg)
    sequences = build_augmented_state_sequences(frame, state_cfg)
    mse = nn.MSELoss()

    lam = 0.5
    small = _multi_step_model(lambda_multi=lam, multi_step_horizon=2)
    small.fit(dataset.Z_t, dataset.R, dataset.Z_next, multi_step_sequences=sequences)
    logged = small.training_history_[-1]["loss"]

    # n_batches counts the per-batch steps plus the one multi-step step, so a
    # weighted contribution must scale with lambda_multi; an unweighted one
    # would not.
    big = _multi_step_model(lambda_multi=lam * 4, multi_step_horizon=2)
    big.fit(dataset.Z_t, dataset.R, dataset.Z_next, multi_step_sequences=sequences)
    assert big.training_history_[-1]["loss"] > logged
