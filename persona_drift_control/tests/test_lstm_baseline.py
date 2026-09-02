import numpy as np
import torch

from persona_drift.modeling.lstm_baseline import (
    LSTMSurrogate,
    mse_from_predictions,
    rollout_predictions,
    teacher_forced_predictions,
    train_lstm_surrogate,
)


def _toy_trajectory(y0: float, v_pattern: list[int], num_turns: int, decay: float = 0.8, gain: float = 0.15) -> list[dict]:
    rows = []
    y = y0
    for t in range(num_turns):
        v = v_pattern[t % len(v_pattern)]
        rows.append({"trajectory_id": "t0", "turn": t + 1, "y_safety": y, "u_remind": v})
        y = decay * y + gain * v
    return rows


def test_lstm_surrogate_satisfies_predictor_protocol():
    model = LSTMSurrogate(hidden_size=3)
    z = model.init_state()
    assert z.shape == (6,)

    y = model.readout(z)
    assert isinstance(y, float)

    z_next = model.step(z, np.array([1.0]))
    assert z_next.shape == (6,)
    assert not np.allclose(z_next, z)  # a real cell update, not a no-op


def test_forward_trajectory_length_matches_input():
    model = LSTMSurrogate(hidden_size=2)
    ys = torch.tensor([0.9, 0.8, 0.7, 0.85], dtype=torch.float32)
    vs = torch.tensor([0.0, 1.0, 1.0, 0.0], dtype=torch.float32)
    preds = model.forward_trajectory(ys, vs)
    assert preds.shape == (4,)


def test_teacher_forced_and_rollout_predictions_shapes():
    model = LSTMSurrogate(hidden_size=2)
    rows = _toy_trajectory(1.0, [0, 1], num_turns=5)

    tf_preds = teacher_forced_predictions(model, rows, y_col="y_safety")
    rollout_preds = rollout_predictions(model, rows, y_col="y_safety")

    assert len(tf_preds) == 5
    assert len(rollout_preds) == 5
    for turn_index, y_true, y_pred in tf_preds:
        assert isinstance(turn_index, int)
        assert isinstance(y_true, float)
        assert isinstance(y_pred, float)


def test_mse_from_predictions_respects_min_turn_index():
    predictions_by_traj = [[(0, 1.0, 0.0), (1, 1.0, 1.0), (2, 1.0, 0.5)]]
    full = mse_from_predictions(predictions_by_traj, min_turn_index=0)
    matched = mse_from_predictions(predictions_by_traj, min_turn_index=1)

    assert full == np.mean([1.0, 0.0, 0.25])
    assert matched == np.mean([0.0, 0.25])


def test_mse_from_predictions_empty_returns_nan():
    assert np.isnan(mse_from_predictions([], min_turn_index=0))


def test_train_lstm_surrogate_reduces_train_loss_on_easy_synthetic_data():
    train_rows = [_toy_trajectory(y0, [0, 1, 1, 0], num_turns=5) for y0 in (1.0, 0.6, -0.4)]
    held_out_rows = [_toy_trajectory(0.8, [1, 0, 0, 1], num_turns=5)]

    model, info = train_lstm_surrogate(
        hidden_size=4,
        train_rows_by_traj=train_rows,
        held_out_rows_by_traj=held_out_rows,
        y_col="y_safety",
        epochs=60,
        lr=5e-2,
        patience=60,
        seed=0,
        min_turn_index=0,
    )

    assert info["history"][-1]["train_loss"] < info["history"][0]["train_loss"]

    held_out_preds = [rollout_predictions(model, rows, y_col="y_safety") for rows in held_out_rows]
    rollout_mse = mse_from_predictions(held_out_preds, min_turn_index=0)
    assert np.isfinite(rollout_mse)
