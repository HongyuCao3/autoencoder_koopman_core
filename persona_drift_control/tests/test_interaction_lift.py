import numpy as np

from persona_drift.modeling.dataset import ReducedStateConfig, build_identification_dataset
from persona_drift.modeling.interaction_lift import InteractionLiftedSurrogate, augment_with_interaction
from persona_drift.modeling.koopman import KoopmanSurrogate, no_extra_features, surrogate_from_arrays


def test_augment_with_interaction_appends_product_column():
    V = np.array([[1.0], [0.0], [1.0]])
    Z = np.array([[0.5, 0.0], [0.9, 1.0], [-0.2, 0.0]])
    V_aug = augment_with_interaction(V, Z, state_index=0)

    assert V_aug.shape == (3, 2)
    assert np.allclose(V_aug[:, 0], V[:, 0])
    assert np.allclose(V_aug[:, 1], [0.5 * 1.0, 0.9 * 0.0, -0.2 * 1.0])


def test_interaction_lifted_surrogate_matches_manual_augmented_step():
    state_dim = 1
    surrogate = surrogate_from_arrays(
        A=[[0.9]], B=[[0.1, 0.3]], b=[0.0], C=[[1.0]], state_dim=state_dim, extra_features_fn=no_extra_features
    )
    wrapped = InteractionLiftedSurrogate(surrogate=surrogate, state_index=0)

    z = np.array([0.4])
    v = np.array([1.0])

    z_next = wrapped.step(z, v)
    v_aug = np.array([1.0, 1.0 * z[0]])
    expected = surrogate.A @ z + surrogate.B @ v_aug + surrogate.b
    assert np.allclose(z_next, expected[:state_dim])
    assert wrapped.readout(z) == surrogate.readout(z)


def _simulate_interaction_only_trajectories(a, g_interaction, y0s, v_pattern, num_turns):
    """y_(t+1) = a*y_t + g_interaction*v_t*y_t, zero MAIN effect of v -- only
    reachable by fitting with an augmented control input; a plain
    KoopmanSurrogate (d_v=1, B a scalar) cannot represent this at all."""

    rows = []
    for i, y0 in enumerate(y0s):
        y = y0
        for t in range(num_turns):
            v = v_pattern[t % len(v_pattern)]
            rows.append({"trajectory_id": f"traj{i}", "turn": t, "y_probe": y, "u_remind": v})
            y = a * y + g_interaction * v * y
    return rows


def test_fitting_with_interaction_recovers_state_dependent_marginal_value():
    a, g_interaction = 0.85, 0.4
    v_pattern = [0, 1, 0, 0, 1, 1, 0, 1]
    rows = _simulate_interaction_only_trajectories(
        a, g_interaction, y0s=[1.0, 0.5, 0.2, 0.8, 0.3], v_pattern=v_pattern, num_turns=20
    )

    config = ReducedStateConfig(nu=1, mu=0)
    dataset = build_identification_dataset(rows, config)
    V_aug = augment_with_interaction(dataset["V"], dataset["Z"], state_index=0)

    model = KoopmanSurrogate(extra_features_fn=no_extra_features, ridge=1e-10).fit(
        {**dataset, "V": V_aug}
    )

    assert np.allclose(model.A[0, 0], a, atol=1e-6)
    assert np.allclose(model.B[0, 0], 0.0, atol=1e-6)  # no main effect in the ground truth
    assert np.allclose(model.B[0, 1], g_interaction, atol=1e-6)

    wrapped = InteractionLiftedSurrogate(surrogate=model, state_index=0)
    margin_high_y = wrapped.readout(wrapped.step(np.array([0.9]), np.array([1.0]))) - wrapped.readout(
        wrapped.step(np.array([0.9]), np.array([0.0]))
    )
    margin_low_y = wrapped.readout(wrapped.step(np.array([0.1]), np.array([1.0]))) - wrapped.readout(
        wrapped.step(np.array([0.1]), np.array([0.0]))
    )
    assert not np.isclose(margin_high_y, margin_low_y)
    assert margin_high_y > margin_low_y  # ground truth: bigger interaction effect at higher y
