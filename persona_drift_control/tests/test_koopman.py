import numpy as np
import pytest

from persona_drift.modeling.dataset import ReducedStateConfig, build_identification_dataset
from persona_drift.modeling.evaluate import one_step_error, rollout_output_error
from persona_drift.modeling.koopman import (
    KoopmanSurrogate,
    abs_sign_extra_features,
    controllability_diagnostics,
    no_extra_features,
)


def _simulate_linear_trajectories(a, g, c, y0s, v_pattern, num_turns):
    """Noiseless synthetic y_(t+1) = a*y_t + g*v_t + c, one trajectory per
    y0, so KoopmanSurrogate.fit's recovered A/B/b can be checked against
    known ground truth -- a fit on real self-chat data has no such ground
    truth to check against, so this is the only place correctness of the
    fitting code itself can be verified directly."""

    rows = []
    for i, y0 in enumerate(y0s):
        y = y0
        for t in range(num_turns):
            v = v_pattern[t % len(v_pattern)]
            rows.append(
                {
                    "trajectory_id": f"traj{i}",
                    "system_prompt_id": f"prompt{i}",
                    "turn": t,
                    "y_probe": y,
                    "u_remind": v,
                }
            )
            y = a * y + g * v + c
    return rows


def test_koopman_arx_recovers_known_linear_system():
    a, g, c = 0.85, 0.3, 0.05
    v_pattern = [0, 1, 0, 0, 1, 1, 0, 1]
    rows = _simulate_linear_trajectories(
        a, g, c, y0s=[1.0, 0.5, -0.3, 0.2, 0.9], v_pattern=v_pattern, num_turns=20
    )

    config = ReducedStateConfig(nu=1, mu=0)
    dataset = build_identification_dataset(rows, config)
    model = KoopmanSurrogate(extra_features_fn=no_extra_features, ridge=1e-10).fit(dataset)

    assert model.A.shape == (1, 1)
    assert model.B.shape == (1, 1)
    assert np.allclose(model.A[0, 0], a, atol=1e-6)
    assert np.allclose(model.B[0, 0], g, atol=1e-6)
    assert np.allclose(model.b[0], c, atol=1e-6)
    assert np.allclose(model.C[0, 0], 1.0, atol=1e-6)  # y_t == z_t exactly when nu=1, mu=0

    assert one_step_error(model, dataset) < 1e-9


def test_koopman_rollout_matches_ground_truth_on_held_out_trajectory():
    a, g, c = 0.85, 0.3, 0.05
    v_pattern = [0, 1, 0, 0, 1, 1, 0, 1]
    train_rows = _simulate_linear_trajectories(
        a, g, c, y0s=[1.0, 0.5, -0.3], v_pattern=v_pattern, num_turns=20
    )
    held_out_rows = _simulate_linear_trajectories(
        a, g, c, y0s=[0.2], v_pattern=v_pattern, num_turns=20
    )
    for row in held_out_rows:  # disjoint ids, mimicking a real held-out split
        row["trajectory_id"] = "held_out"
        row["system_prompt_id"] = "held_out_prompt"

    config = ReducedStateConfig(nu=1, mu=0)
    model = KoopmanSurrogate(ridge=1e-10).fit(build_identification_dataset(train_rows, config))

    assert rollout_output_error(model, held_out_rows, config) < 1e-6


def test_arx_and_richer_lifting_share_the_same_fit_and_eval_path():
    a, g, c = 0.6, 0.4, 0.0
    v_pattern = [0, 1, 1, 0]
    rows = _simulate_linear_trajectories(a, g, c, y0s=[0.4, -0.4, 0.9], v_pattern=v_pattern, num_turns=16)
    config = ReducedStateConfig(nu=1, mu=0)
    dataset = build_identification_dataset(rows, config)

    arx = KoopmanSurrogate(extra_features_fn=no_extra_features).fit(dataset)
    richer = KoopmanSurrogate(extra_features_fn=abs_sign_extra_features).fit(dataset)

    # Both go through the identical fit()/one_step_error() code path; the
    # richer lifting strictly contains ARX's features (extra weights can
    # always be driven to ~0), so it should never fit worse on training data.
    assert one_step_error(richer, dataset) <= one_step_error(arx, dataset) + 1e-6


def test_controllability_diagnostics_rank_matches_actuation():
    uncontrollable = controllability_diagnostics(A=np.eye(2), B=np.zeros((2, 1)), horizon=5)
    assert uncontrollable["controllability_rank"] == 0

    controllable = controllability_diagnostics(
        A=np.array([[1.0, 1.0], [0.0, 1.0]]), B=np.array([[0.0], [1.0]]), horizon=5
    )
    assert controllable["controllability_rank"] == 2


def test_fit_raises_on_empty_dataset():
    empty = {"Z": np.zeros((0, 1)), "V": np.zeros((0, 1)), "Z_next": np.zeros((0, 1)), "Y": np.zeros(0)}
    with pytest.raises(ValueError):
        KoopmanSurrogate().fit(empty)
