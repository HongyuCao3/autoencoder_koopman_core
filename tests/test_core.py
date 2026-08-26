from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from koopman_ae import (
    AugmentedStateConfig,
    DeepAugmentedKoopmanAutoencoder,
    DeepAugmentedKoopmanConfig,
    build_augmented_state_dataset,
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
