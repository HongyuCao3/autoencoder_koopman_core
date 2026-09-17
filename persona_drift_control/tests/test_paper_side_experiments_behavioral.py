import importlib.util
import pathlib

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "_paper_side_experiments_behavioral", REPO / "scripts" / "paper_side_experiments_behavioral.py")
sb = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sb)


def test_readout_taps_come_from_the_transition_not_the_readout():
    """Regression for the first run, which reported [1, 0, 0, 0] on all three
    lines. C is a selection vector here -- the y-block holds y verbatim -- so a
    loading read off C is constant by construction and says nothing about
    memory. The loading has to come from row nu-1 of A, newest tap first, on a
    y-block stored oldest-first.
    """
    nu = 4
    A = np.zeros((nu, nu))
    # z_(t+1)'s newest y loads 0.5 on y_t, 0.3 on y_(t-1), nothing older.
    A[nu - 1, nu - 1] = 0.5
    A[nu - 1, nu - 2] = 0.3
    assert sb._readout_taps(A, nu) == [0.5, 0.3, 0.0, 0.0]


def test_a_markov_transition_reports_no_older_tap_weight():
    nu = 3
    A = np.zeros((nu, nu))
    A[nu - 1, nu - 1] = 0.9
    taps = sb._readout_taps(A, nu)
    assert taps[0] == 0.9 and taps[1:] == [0.0, 0.0]


def test_item_level_estimated_into_the_scored_window_raises():
    """The leakage guard, signed 2026-09-16."""
    by_traj = {"a": [{"turn": t, "y": 0.5} for t in range(1, 8)]}
    with pytest.raises(sb.LeakageError, match="item level"):
        sb._item_levels(by_traj, "y", prefix_turn=5, scored_min_turn=5)


def test_item_level_uses_only_the_prefix():
    by_traj = {"a": [{"turn": t, "y": float(t)} for t in range(1, 8)]}
    levels = sb._item_levels(by_traj, "y", prefix_turn=4, scored_min_turn=5)
    assert levels["a"] == pytest.approx(np.mean([1.0, 2.0, 3.0, 4.0]))


def test_identifiability_refuses_an_action_that_is_a_function_of_the_state():
    Z = np.random.default_rng(0).normal(size=(200, 3))
    identified = sb._identifiability(Z, (Z @ np.array([1.0, -0.5, 0.25])).reshape(-1, 1))
    assert identified["identified"] is False
    assert identified["undefined_reason"]
    free = sb._identifiability(Z, np.random.default_rng(1).normal(size=(200, 1)))
    assert free["identified"] is True
