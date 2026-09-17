import importlib.util
import json
import pathlib

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "_paper_side_experiments", ROOT / "scripts" / "paper_side_experiments.py")
se = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(se)

LANDED = ROOT / "results" / "surrogate_rows_paired"
COLUMNS = ("normalized_output",), ("effective_norm",)


def _second_order_frame(n_traj=12, n_turns=10, seed=0, noise=0.0):
    """y_(t+1) = 1.2 y_t - 0.45 y_(t-1) + 0.25 r -- memory is genuinely needed.

    Same plant as `tests/test_eval_surrogate_rows.py`, on purpose: the two
    files assert different things about one system, so a change in the fixture
    cannot make one of them pass by making the plant easier.

    `noise` matters more than it looks. With none, three consecutive outputs
    invert the recursion exactly, so r is a linear function of the state and
    the M1 rank check correctly refuses to report B. Any real readout has
    noise; the identified case has to be built with some.
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
            nxt = 1.2 * y - 0.45 * y_prev + 0.25 * r + noise * float(rng.standard_normal())
            y, y_prev = nxt, y
    return pd.DataFrame(rows)


def _frozen_frame(n_traj=9, n_turns=10):
    """y never moves and r equals it, so R is an exact affine function of [Z|1].

    This is `core`'s candidate-A failure in miniature: the control channel
    carries no variation of its own, so B is undefined rather than small.
    """
    rows = []
    for i in range(n_traj):
        y = 0.1 * i
        split = ("train", "validation", "test")[i % 3]
        for turn in range(1, n_turns + 1):
            rows.append({
                "trajectory_id": f"traj_{i:02d}", "topic": f"t{i}", "topic_split": split,
                "turn": turn, "normalized_output": y, "effective_norm": y,
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- A1

def test_deep_lag_spends_the_whole_trajectory_on_depth():
    assert se.deep_lag(10) == 6
    assert se.deep_lag(5) == 1
    assert se.ev.choose_lag(10) == 3


def test_every_lag_is_scored_on_the_same_rows_and_horizon():
    frame = _second_order_frame()
    result = se.run_a1_frame(frame, *COLUMNS, "planted", "deep")
    assert result["observed_seed_turns"] == se.deep_lag(10) + 1
    assert result["horizon"] == 3
    n_rows = {cell["n_rows"] for cell in result["rows"].values()}
    assert len(n_rows) == 1, "a lag was scored on a different row set"


def test_memory_shows_up_on_a_plant_that_needs_it():
    frame = _second_order_frame()
    result = se.run_a1_frame(frame, *COLUMNS, "planted", "primary")
    assert result["rows"]["lag_1"]["skill_h"] > result["rows"]["lag_0"]["skill_h"]
    assert result["contrasts"]["lag_1_minus_lag_0"]["ci_low"] > 0.0, \
        "second-order plant must separate lag 1 from the Markov row"


@pytest.mark.skipif(not (LANDED / "sentence_length_t10.json").exists(),
                    reason="Table 1's landed artifact is not present")
def test_primary_anchor_reproduces_table_1s_landed_cells():
    """The drift guard, signed 2026-09-16.

    A sweep that quietly changes the fit, the split, the null or the resampling
    unit would still draw a curve; it just would not be a curve through Table 1.
    The deepest grid point is Table 1's `ours` and the shallowest is its Markov
    row, so if either drifts the whole panel is disconnected from the table it
    is supposed to explain.
    """
    landed = json.loads((LANDED / "sentence_length_t10.json").read_text())
    frame, cfg = se.ev._load("sentence_length_t10")
    result = se.run_a1_frame(
        frame, tuple(cfg["output_columns"]), tuple(cfg["target_columns"]),
        "sentence_length_t10", "primary")
    assert result["lag_max"] == landed["lag"]
    assert result["observed_seed_turns"] == landed["observed_seed_turns"]
    assert result["horizon"] == landed["horizon"]
    assert result["best_null_name"] == landed["best_null_name"]
    for grid, row in (("lag_3", "delay_linear_control"), ("lag_0", "markov_linear_control")):
        np.testing.assert_allclose(
            result["rows"][grid]["rollout_mse"], landed["rows"][row]["rollout_mse"], rtol=1e-12)
        np.testing.assert_allclose(
            result["rows"][grid]["skill_h"], landed["rows"][row]["skill_h"], rtol=1e-12)


# --------------------------------------------------------------------------- M1

def test_rank_deficient_control_channel_emits_no_response_numbers():
    """P1-a: undefined must not be reported as a number.

    Candidate A's withdrawn Limitations sentence is what this guard costs if it
    is missing -- a question the data cannot ask, answered with a zero.
    """
    result = se.run_m1_frame(_frozen_frame(), *COLUMNS, "frozen",
                             action_channel="constant_reference_per_trajectory")
    assert result["identifiability"]["identified"] is False
    assert result["response"]["identified"] is False
    assert result["response"]["impulse_response_norm"] is None
    assert result["response"]["response_length"] is None
    assert result["response"]["undefined_reason"]


def test_a_noiseless_delay_embedding_also_makes_the_input_undefined():
    """Not a corner case -- the general form of the candidate-A failure.

    With enough delay taps and no noise, the recursion inverts and r is a
    linear function of the state, so B is unidentified even though the control
    channel looks perfectly well behaved in the data file.
    """
    result = se.run_m1_frame(_second_order_frame(), *COLUMNS, "planted",
                             action_channel="constant_reference_per_trajectory")
    assert result["identifiability"]["identified"] is False
    assert result["response"]["impulse_response_norm"] is None


def _ar1_frame(a, n_traj=36, n_turns=10, seed=0, noise=0.02):
    """y_(t+1) = a y_t + (1-a) r + noise: one knob, and it is the half-life."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_traj):
        r = float(rng.uniform(0.2, 0.9))
        y = float(rng.uniform(0.0, 1.0))
        split = ("train", "validation", "test")[i % 3]
        for turn in range(1, n_turns + 1):
            rows.append({
                "trajectory_id": f"traj_{i:02d}", "topic": f"t{i}", "topic_split": split,
                "turn": turn, "normalized_output": y, "effective_norm": r,
            })
            y = a * y + (1 - a) * r + noise * float(rng.standard_normal())
    return pd.DataFrame(rows)


def test_half_life_orders_two_plants_by_how_long_they_remember():
    """The mechanism reading is an ORDERING across columns, so that is what is
    asserted. A point estimate of the spectral radius off a ridge fit of a
    four-tap embedding is worth about one significant figure -- pinning the
    test to its absolute value would be testing the noise."""
    fast = se.run_m1_frame(_ar1_frame(0.5), *COLUMNS, "fast",
                           action_channel="constant_reference_per_trajectory")
    slow = se.run_m1_frame(_ar1_frame(0.95), *COLUMNS, "slow",
                           action_channel="constant_reference_per_trajectory")
    for result in (fast, slow):
        assert result["identifiability"]["identified"] is True
        assert result["spectrum"]["spectral_radius"] < 1.0
        assert result["response"]["caveat"], "a constant reference cannot be an observed impulse"
    assert slow["spectrum"]["half_life_steps"] > 5 * fast["spectrum"]["half_life_steps"]


# --------------------------------------------------------------------------- M2

def test_item_level_estimated_on_a_scored_turn_raises():
    """The leakage guard, signed 2026-09-16.

    Written as a raise rather than a note in the results file because a leak
    here does not fail: it makes the baseline stronger, the contrast smaller,
    and the paper's positive claim quietly weaker for the wrong reason.
    """
    frame = _second_order_frame()
    with pytest.raises(se.LeakageError, match="also scored"):
        se._with_item_level(frame, ("normalized_output",), seed_turn=4, scored_turns=[4, 5, 6])


def test_item_level_is_constant_along_a_trajectory_and_uses_only_the_prefix():
    frame = _second_order_frame()
    merged, columns = se._with_item_level(
        frame, ("normalized_output",), seed_turn=4, scored_turns=[5, 6, 7, 8, 9, 10])
    assert columns == ["item_level_0"]
    for _, group in merged.groupby("trajectory_id"):
        assert group["item_level_0"].nunique() == 1
        prefix = group[group["turn"] <= 4]["normalized_output"].mean()
        np.testing.assert_allclose(group["item_level_0"].iloc[0], prefix, rtol=1e-12)


def test_item_intercept_cannot_explain_a_second_order_plant():
    """On a plant whose memory is real, handing the Markov row item identity
    must not close the gap -- otherwise the probe cannot tell the two
    mechanisms apart on real data either."""
    result = se.run_m2_frame(_second_order_frame(n_traj=60, noise=0.02), *COLUMNS, "planted")
    assert result["contrasts"]["ours_minus_markov_plus_item"]["ci_low"] > 0.0
    assert result["rows"]["markov_plus_item"]["target_dim"] == 2
