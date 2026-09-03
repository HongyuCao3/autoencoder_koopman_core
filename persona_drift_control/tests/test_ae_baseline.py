import numpy as np

from persona_drift.modeling.ae_baseline import AEKoopmanConfig, AEKoopmanSurrogate
from persona_drift.modeling.dataset import ReducedStateConfig, build_identification_dataset
from persona_drift.modeling.evaluate import one_step_error, rollout_output_error


def _toy_trajectory(traj_id: str, y0: float, v_pattern: list[int], num_turns: int, decay: float = 0.8, gain: float = 0.15) -> list[dict]:
    rows = []
    y = y0
    for t in range(num_turns):
        v = v_pattern[t % len(v_pattern)]
        rows.append({"trajectory_id": traj_id, "turn": t + 1, "y_safety": y, "u_remind": v})
        y = decay * y + gain * v
    return rows


def test_encode_decode_roundtrip_shapes():
    model = AEKoopmanSurrogate(state_dim=3, config=AEKoopmanConfig(latent_dim=2, hidden_dim=4, num_epochs=1))
    z = np.array([0.5, 1.0, 0.0])
    xi = model.encode(z)
    assert xi.shape == (2,)
    z_hat = model.decode(xi)
    assert z_hat.shape == (3,)


def test_fit_then_satisfies_predictor_protocol():
    config = ReducedStateConfig(nu=1, mu=2)
    rows = [row for i in range(3) for row in _toy_trajectory(f"t{i}", 1.0 - 0.2 * i, [0, 1, 1, 0], num_turns=6)]
    dataset = build_identification_dataset(rows, config, y_col="y_safety")

    model = AEKoopmanSurrogate(
        state_dim=config.state_dim, config=AEKoopmanConfig(latent_dim=2, hidden_dim=4, num_epochs=30)
    ).fit(dataset)

    z0 = dataset["Z"][0]
    y0 = model.readout(z0)
    assert isinstance(y0, float)
    assert y0 == z0[0]  # readout is a verbatim slice of z[0] under nu=1, matching core.py's predict_y

    z1 = model.step(z0, dataset["V"][0])
    assert z1.shape == z0.shape
    assert not np.allclose(z1, z0)  # a real dynamics update, not a no-op


def test_fit_reduces_reconstruction_loss():
    config = ReducedStateConfig(nu=1, mu=2)
    rows = [row for i in range(3) for row in _toy_trajectory(f"t{i}", 1.0 - 0.2 * i, [0, 1, 1, 0], num_turns=6)]
    dataset = build_identification_dataset(rows, config, y_col="y_safety")

    model = AEKoopmanSurrogate(
        state_dim=config.state_dim, config=AEKoopmanConfig(latent_dim=2, hidden_dim=4, num_epochs=100)
    ).fit(dataset)

    losses = [h["reconstruction_loss"] for h in model.training_history_]
    assert losses[-1] < losses[0]


def test_evaluate_helpers_accept_ae_surrogate_directly():
    """Unlike the LSTM baseline, this model's step/readout operate on the same
    raw z the Koopman surrogate uses, so it should plug into modeling.evaluate's
    Predictor-protocol functions unmodified -- see ae_baseline_plan.md."""

    config = ReducedStateConfig(nu=1, mu=2)
    train_rows = [row for i in range(3) for row in _toy_trajectory(f"train{i}", 1.0 - 0.2 * i, [0, 1, 1, 0], num_turns=6)]
    held_out_rows = _toy_trajectory("held_out0", 0.7, [1, 0, 0, 1], num_turns=6)
    dataset = build_identification_dataset(train_rows, config, y_col="y_safety")

    model = AEKoopmanSurrogate(
        state_dim=config.state_dim, config=AEKoopmanConfig(latent_dim=2, hidden_dim=4, num_epochs=30)
    ).fit(dataset)

    one_step_mse = one_step_error(model, dataset)
    rollout_mse = rollout_output_error(model, held_out_rows, config, y_col="y_safety")
    assert np.isfinite(one_step_mse)
    assert np.isfinite(rollout_mse)


def test_n_params_counts_encoder_decoder_and_dynamics():
    config = ReducedStateConfig(nu=1, mu=2)
    rows = [row for i in range(2) for row in _toy_trajectory(f"t{i}", 1.0, [0, 1], num_turns=5)]
    dataset = build_identification_dataset(rows, config, y_col="y_safety")

    model = AEKoopmanSurrogate(
        state_dim=config.state_dim, config=AEKoopmanConfig(latent_dim=1, hidden_dim=4, num_epochs=1)
    ).fit(dataset)

    assert model.n_params() > 0
