"""`surrogate_eval.trivial_nulls` must stay value-identical to the copy that
G-K2-1 and G-S2-1 were computed with.

The shared harness took the authoritative copy so the paper's ten datasets all
score against the same nulls, but `fit_koopman_defense_model` keeps its own --
it is a committed-results reproduction path and does not get edited. Two
copies of a null definition drift apart; this test is what stops that, and it
fails loudly rather than letting a main-table cell quietly stop matching the
gate it came from.
"""
import importlib.util
import pathlib

import numpy as np
import pytest

from surrogate_eval import trivial_nulls

_DEFENSE_FIT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "fit_koopman_defense_model.py"


def _defense_module():
    spec = importlib.util.spec_from_file_location("_fit_koopman_defense_model", _DEFENSE_FIT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _random_panel(rng, n, n_turns):
    turn = rng.integers(1, n_turns + 1, size=n).astype(float)
    return {
        "y_next": rng.uniform(0.0, 1.0, size=n),
        "turn_next": turn,
        "v": rng.integers(0, 2, size=n).astype(float),
    }


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_shared_nulls_match_the_defense_line_copy(seed):
    rng = np.random.default_rng(seed)
    train = _random_panel(rng, 240, 5)
    test = _random_panel(rng, 80, 5)

    theirs = _defense_module()._null_predictions(train, test)
    ours = trivial_nulls(train, test, exogenous=("turn_next", "v"))

    assert sorted(theirs) == sorted(ours) == ["const", "stateless", "turn_mean"]
    for name in theirs:
        np.testing.assert_allclose(ours[name], theirs[name], rtol=0, atol=0,
                                   err_msg=f"null {name!r} drifted from the defense-line definition")


def test_equivalence_also_holds_when_a_test_turn_was_never_trained_on():
    # The fallback branch: `turn_mean` has no estimate for turn 5, both copies
    # must fall back to the same constant.
    rng = np.random.default_rng(9)
    train = _random_panel(rng, 120, 4)
    test = _random_panel(rng, 40, 5)
    assert 5.0 in set(test["turn_next"]) and 5.0 not in set(train["turn_next"])

    theirs = _defense_module()._null_predictions(train, test)
    ours = trivial_nulls(train, test, exogenous=("turn_next", "v"))
    np.testing.assert_allclose(ours["turn_mean"], theirs["turn_mean"], rtol=0, atol=0)
