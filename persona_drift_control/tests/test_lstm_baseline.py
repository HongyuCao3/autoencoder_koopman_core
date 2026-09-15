import numpy as np
import pytest
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


def test_contemporaneous_v_feeds_the_next_turns_action():
    """The v-alignment fix (docs/experiments/lstm_baseline_plan.md, mirroring
    dataset.ReducedStateConfig.contemporaneous_v): predicting y_(t+1) must
    read v_(t+1) -- the reminder inserted before that reply -- not v_t. Both
    modes are checked against an explicit hand-rolled reference so a silent
    off-by-one in either direction fails here."""

    model = LSTMSurrogate(hidden_size=2)
    ys = torch.tensor([1.0, 0.75, 0.5, 0.25], dtype=torch.float32)
    vs = torch.tensor([0.0, 1.0, 0.0, 1.0], dtype=torch.float32)

    @torch.no_grad()
    def reference(v_indices: list[int]) -> list[float]:
        h = torch.zeros(1, 2)
        c = torch.zeros(1, 2)
        out = [float(model.readout_layer(h).reshape(()))]
        for t, vi in enumerate(v_indices):
            h, c = model.cell(torch.stack([ys[t], vs[vi]]).reshape(1, 2), (h, c))
            out.append(float(model.readout_layer(h).reshape(())))
        return out

    with torch.no_grad():
        unaligned = model.forward_trajectory(ys, vs).numpy().tolist()
        aligned = model.forward_trajectory(ys, vs, contemporaneous_v=True).numpy().tolist()

    assert unaligned == pytest.approx(reference([0, 1, 2]))
    assert aligned == pytest.approx(reference([1, 2, 3]))
    # vs alternates, so the two pairings genuinely differ on this trajectory
    # (they would coincide for a constant v and prove nothing).
    assert unaligned != pytest.approx(aligned)


def test_rollout_predictions_shift_matches_forward_trajectory_shift():
    """`rollout_predictions` feeds the same v slot as `forward_trajectory`;
    if only one of the two were shifted, training and held-out scoring would
    silently disagree about which action caused which turn."""

    model = LSTMSurrogate(hidden_size=2)
    rows = [
        {"trajectory_id": "t0", "turn": t + 1, "y_safety": y, "u_remind": v}
        for t, (y, v) in enumerate([(1.0, 0), (0.75, 1), (0.5, 0), (0.25, 1)])
    ]
    vs = [float(row["u_remind"]) for row in rows]

    aligned = [pred for _, _, pred in rollout_predictions(model, rows, y_col="y_safety", contemporaneous_v=True)]

    z = model.init_state()
    expected = [model.readout(z)]
    for t in range(len(rows) - 1):
        z = model.step(z, np.array([vs[t + 1]]))
        expected.append(model.readout(z))

    assert aligned == pytest.approx(expected)


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
        early_stop_rows_by_traj=held_out_rows,
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


def test_warm_start_with_an_empty_prefix_is_exactly_init_state():
    """The regression anchor: the new method's zero-length case must be the
    old behaviour bit for bit, so no existing caller changes."""
    import numpy as np

    from persona_drift.modeling.lstm_baseline import LSTMSurrogate

    model = LSTMSurrogate(hidden_size=3)
    np.testing.assert_array_equal(model.warm_start([], []), model.init_state())


def test_warm_start_carries_the_observed_prefix_into_the_rollout():
    """A prefix the model has actually seen must move its state, and the
    resulting rollout must differ from the cold one -- otherwise the fairness
    fix is cosmetic."""
    import numpy as np
    import pytest

    from persona_drift.modeling.lstm_baseline import LSTMSurrogate

    model = LSTMSurrogate(hidden_size=4)
    warm = model.warm_start([0.9, 0.8, 0.7], [1.0, 0.0, 1.0])
    assert not np.allclose(warm, model.init_state())
    assert model.readout(warm) != pytest.approx(model.readout(model.init_state()))
    with pytest.raises(ValueError):
        model.warm_start([0.1, 0.2], [1.0])


def test_v_dim_one_matches_scalar():
    """The default width is bit-identical to the pre-`v_dim` code path.

    Three published Table 1 columns (`constraint` / `gsm8k_sharded` /
    `defense`) were produced before the action became a vector, so the
    scalar path has to stay exactly where it was: same `input_size`, same
    parameter init under the same seed, same tensor build order.
    """
    torch.manual_seed(0)
    model = LSTMSurrogate(hidden_size=3)
    assert model.v_dim == 1 and model.cell.input_size == 2

    ys = torch.tensor([0.9, 0.8, 0.7, 0.85], dtype=torch.float32)
    flat = torch.tensor([0.0, 1.0, 1.0, 0.0], dtype=torch.float32)
    assert torch.equal(model.forward_trajectory(ys, flat),
                       model.forward_trajectory(ys, flat.reshape(-1, 1)))

    rows = _toy_trajectory(0.9, [0, 1], num_turns=5)
    assert (rollout_predictions(model, rows, u_col="u_remind")
            == rollout_predictions(model, rows, u_col=["u_remind"]))
    assert (teacher_forced_predictions(model, rows, u_col="u_remind")
            == teacher_forced_predictions(model, rows, u_col=["u_remind"]))
    assert np.array_equal(model.warm_start([0.9, 0.8], [0.0, 1.0]),
                          model.warm_start([0.9, 0.8], [[0.0], [1.0]]))


def test_vector_action_channels_are_distinguishable():
    """`tsar_cefr` carries a 3-dof one-hot; the cell must see all three."""
    torch.manual_seed(0)
    model = LSTMSurrogate(hidden_size=3, v_dim=3)
    assert model.cell.input_size == 4

    z = model.init_state()
    nexts = [model.step(z, np.eye(3)[i]) for i in range(3)]
    for i in range(3):
        for j in range(i + 1, 3):
            assert not np.allclose(nexts[i], nexts[j]), f"channels {i},{j} collapsed"


def test_vector_action_trains_and_uses_the_channels():
    rows_by_traj = []
    for k, onehot in enumerate([(1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 0, 0)]):
        rows, y = [], 3.5
        for t in range(6):
            rows.append({"trajectory_id": f"t{k}", "turn": t + 1, "ell": y,
                         "a0": onehot[0], "a1": onehot[1], "a2": onehot[2]})
            y = 0.9 * y - 0.5 * onehot[0] - 0.25 * onehot[1]
        rows_by_traj.append(rows)
    cols = ["a0", "a1", "a2"]
    model, info = train_lstm_surrogate(4, rows_by_traj[:3], rows_by_traj[3:],
                                       y_col="ell", u_col=cols, epochs=40, seed=0)
    assert model.v_dim == 3
    assert info["history"][-1]["train_loss"] < info["history"][0]["train_loss"]


def test_wrong_action_width_raises():
    model = LSTMSurrogate(hidden_size=2, v_dim=3)
    with pytest.raises(ValueError, match="expected v_dim=3"):
        model.step(model.init_state(), np.array([1.0]))
    with pytest.raises(ValueError, match="at least one action channel"):
        LSTMSurrogate(hidden_size=2, v_dim=0)
