import importlib.util
import pathlib

import numpy as np
import pandas as pd
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "_eval_surrogate_rows", pathlib.Path(__file__).resolve().parents[1] / "scripts" / "eval_surrogate_rows.py"
)
ev = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ev)

from koopman_ae import AugmentedKoopmanModel, AugmentedStateConfig, build_augmented_state_dataset


def _planted_frame(n_traj=12, n_turns=10, seed=0):
    """A SECOND-order system: y_(t+1) = 1.2 y_t - 0.45 y_(t-1) + 0.25 r.

    Second order on purpose. A first-order plant is Markov, so the
    delay-embedded row and the Markov row fit it equally well and a contrast
    between them is legitimately zero -- which is what the first version of
    this fixture asserted the opposite of. The characteristic roots are complex
    with modulus sqrt(0.45) = 0.67, so the plant is stable and oscillatory, and
    memory is genuinely required to predict it.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_traj):
        r = float(rng.uniform(0.2, 0.9))
        y_prev = y = float(rng.uniform(0.0, 1.0))
        split = ("train", "validation", "test")[i % 3]
        for turn in range(1, n_turns + 1):
            rows.append({
                "trajectory_id": f"traj_{i:02d}", "topic": f"t{i}", "topic_split": split,
                "turn": turn, "normalized_output": y, "effective_norm": r,
            })
            y, y_prev = 1.2 * y - 0.45 * y_prev + 0.25 * r, y
    return pd.DataFrame(rows)


def test_lag_is_the_deepest_one_that_leaves_three_rollout_steps():
    assert ev.choose_lag(10) == 3
    assert ev.choose_lag(5) == 1
    with pytest.raises(ValueError, match="no lag leaves"):
        ev.choose_lag(4)


def test_stacked_handles_scalar_and_vector_rows_the_same_way():
    scalar = pd.DataFrame({"y": [0.1, 0.2]})
    vector = pd.DataFrame({"y": [[0.1, 0.9], [0.2, 0.8]]})
    assert ev._stacked(scalar, "y").shape == (2, 1)
    assert ev._stacked(vector, "y").shape == (2, 2)


def test_no_control_model_cannot_see_the_reference():
    frame = _planted_frame()
    cfg = AugmentedStateConfig(output_memory=2, input_memory=0, control_mode="error",
                               output_columns=("normalized_output",), target_columns=("effective_norm",))
    dataset = build_augmented_state_dataset(frame[frame.topic_split.eq("train")], cfg)
    inner = AugmentedKoopmanModel(output_dim=1, alpha=1e-6)
    inner.fit(dataset.Z_t, np.zeros_like(dataset.R), dataset.Z_next)
    blind = ev._NoControlModel(inner)
    z = dataset.Z_t[0]
    # Two very different references must give the identical next state.
    np.testing.assert_allclose(
        blind.predict_next_z(z, np.array([0.0])), blind.predict_next_z(z, np.array([9.9])))


def test_rows_are_aligned_by_key_not_by_position():
    # Regression for the bug the smoke run caught: the Koopman rollout sorts
    # with pandas' default (unstable) quicksort, so its trajectory order is not
    # the frame's. Scoring by position silently paired each prediction with
    # another trajectory's truth.
    frame = _planted_frame()
    result_a = ev.run_task_frame(frame, ("normalized_output",), ("effective_norm",), "planted", families=())
    shuffled = pd.concat([
        frame[frame.trajectory_id.eq(t)] for t in sorted(frame.trajectory_id.unique(), reverse=True)
    ]).reset_index(drop=True)
    result_b = ev.run_task_frame(shuffled, ("normalized_output",), ("effective_norm",), "planted", families=())
    for row in ("delay_linear_control", "markov_linear_control"):
        assert result_a["rows"][row]["skill_h"] == pytest.approx(result_b["rows"][row]["skill_h"], abs=1e-9)


def test_planted_linear_system_is_recovered_by_the_controlled_row():
    result = ev.run_task_frame(_planted_frame(), ("normalized_output",), ("effective_norm",), "planted", families=())
    assert result["lag"] == 3 and result["horizon"] == 6
    assert result["rows"]["delay_linear_control"]["skill_h"] > 0.99
    assert result["rows"]["best_null"]["skill_h"] == 0.0


def test_a_flat_readout_is_reported_as_degenerate_not_scored():
    frame = _planted_frame()
    frame["normalized_output"] = 0.5
    frame["effective_norm"] = 0.5
    result = ev.run_task_frame(frame, ("normalized_output",), ("effective_norm",), "flat", families=())
    assert result["rows"]["delay_linear_control"]["skill_h"] is None
    assert "degenerate" in result["rows"]["delay_linear_control"]


def test_nulls_are_given_turn_and_the_reference_but_never_the_control():
    frame = _planted_frame()
    cfg = AugmentedStateConfig(output_memory=4, input_memory=0, control_mode="error",
                               output_columns=("normalized_output",), target_columns=("effective_norm",))
    from koopman_ae import rollout_augmented_from_trajectories
    train = frame[frame.topic_split.eq("train")]
    test = frame[frame.topic_split.eq("test")]
    dataset = build_augmented_state_dataset(train, cfg)
    model = AugmentedKoopmanModel(output_dim=1, alpha=1e-6).fit(dataset.Z_t, dataset.R, dataset.Z_next)
    rollout = rollout_augmented_from_trajectories(model, test, cfg, observed_seed_turns=4)
    rollout = rollout[~rollout["uses_observed_seed"]].reset_index(drop=True)
    nulls = ev._null_panels(train, rollout, ("normalized_output",), ("effective_norm",), 4)
    assert sorted(nulls) == ["const", "stateless", "turn_mean"]
    # The stateless null is exogenous-only: two rows with the same (turn, r)
    # must get the same prediction no matter what y did.
    keys = list(zip(rollout["turn"], [float(v) for v in rollout["target_effective_norm"]]))
    for key in set(keys):
        idx = [i for i, k in enumerate(keys) if k == key]
        assert len(set(np.round(nulls["stateless"][idx, 0], 12))) == 1


def test_every_row_gets_the_same_kind_of_interval_and_a_paired_contrast():
    result = ev.run_task_frame(
        _planted_frame(n_traj=15), ("normalized_output",), ("effective_norm",), "planted",
        families=("lstm",), )
    for name in ("markov_linear_control", "delay_linear_control", "lstm"):
        assert "bootstrap" in result["rows"][name], f"{name} has no bootstrap CI"
    # The seeded row still reports its spread across seeds, but not as the bar.
    assert "seed_spread_ddof1" in result["rows"]["lstm"]
    assert "per_seed" in result["rows"]["lstm"]
    key = "delay_linear_control_minus_markov_linear_control"
    contrast = result["contrasts"][key]
    # On a planted linear system the delay-embedded row beats the Markov one,
    # and the paired interval has to say so rather than merely not overlapping.
    assert contrast["point"] > 0 and contrast["excludes_zero"]


def test_contrast_of_a_row_against_itself_is_exactly_zero():
    # Negative control for the contrast statistic: paired against itself the
    # difference must vanish in every resample, not merely on average.
    rng = np.random.default_rng(2)
    se = rng.uniform(0.1, 1.0, 60)
    null = rng.uniform(0.5, 2.0, 60)
    groups = np.repeat(np.arange(20), 3)
    out = ev._contrast(se, se, null, groups, seed=0)
    assert out["point"] == pytest.approx(0.0)
    assert out["ci_low"] == pytest.approx(0.0) and out["ci_high"] == pytest.approx(0.0)


def test_lstm_rollout_emits_one_prediction_per_future_turn():
    frame = _planted_frame(n_traj=6, n_turns=8)
    seqs = ev._sequences(frame, ("normalized_output",), ("effective_norm",))
    model, _ = ev.fit_lstm(seqs, seqs, out_dim=1, hidden=4, seed=0, max_epochs=5, patience=5)
    preds, index = ev.lstm_rollout(model, seqs, seed_turn=4)
    assert preds.shape == (6 * 4, 1)
    assert index[:4] == [("traj_00", 5), ("traj_00", 6), ("traj_00", 7), ("traj_00", 8)]


def test_provenance_records_the_harness_version_and_never_fakes_it():
    prov = ev.provenance()
    assert len(prov["git_sha"]) == 40 and prov["git_sha"].isalnum()
    assert isinstance(prov["git_dirty"], bool)
    assert prov["seeds"] == [0, 1, 2]
    # A missing fingerprint must fail loudly, not be written as null.
    import subprocess
    with pytest.raises(subprocess.CalledProcessError):
        original, ev.PACKAGE_ROOT = ev.PACKAGE_ROOT, pathlib.Path("/")
        try:
            ev.provenance()
        finally:
            ev.PACKAGE_ROOT = original
