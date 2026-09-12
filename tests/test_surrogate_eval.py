import numpy as np
import pytest

from surrogate_eval import (
    DegenerateNullError,
    best_null,
    bootstrap_ci,
    make_folds,
    seed_aggregate,
    skill_h,
    trivial_nulls,
)


def _panel(y_next, turn, v):
    return {"y_next": np.asarray(y_next, float), "turn_next": np.asarray(turn, float), "v": np.asarray(v, float)}


# ---------------------------------------------------------------- nulls

def test_stateless_null_must_be_given_turn():
    p = _panel([1.0, 2.0], [1, 2], [0, 1])
    with pytest.raises(ValueError, match="every null gets `turn`"):
        trivial_nulls(p, p, exogenous=("v",))


def test_missing_column_is_refused():
    p = _panel([1.0, 2.0], [1, 2], [0, 1])
    with pytest.raises(KeyError):
        trivial_nulls(p, {"y_next": p["y_next"]}, exogenous=("turn_next", "v"))


def test_stateless_null_recovers_a_planted_exogenous_law():
    # Positive control: y_next = 0.5 + 0.25*turn - 0.75*v exactly. A null that
    # is handed turn and v must fit it with zero error -- if it cannot, any
    # "surrogate beats the null" verdict downstream is unearned.
    turn = np.tile(np.arange(1, 6, dtype=float), 4)
    v = np.resize([0.0, 1.0], turn.size).astype(float)
    y = 0.5 + 0.25 * turn - 0.75 * v
    p = _panel(y, turn, v)
    preds = trivial_nulls(p, p, exogenous=("turn_next", "v"))
    assert np.allclose(preds["stateless"], y, atol=1e-10)


def test_turn_mean_falls_back_to_const_on_an_unseen_turn():
    train = _panel([1.0, 3.0], [1, 1], [0, 0])
    test = _panel([9.0], [7], [0])
    preds = trivial_nulls(train, test, exogenous=("turn_next", "v"))
    assert preds["turn_mean"][0] == pytest.approx(2.0)


def test_best_null_reports_which_one_won():
    y = np.array([0.0, 1.0, 2.0])
    name, pred = best_null({"a": np.array([0.0, 1.0, 2.0]), "b": np.zeros(3)}, y)
    assert name == "a" and np.allclose(pred, y)


# ---------------------------------------------------------------- skill

def test_skill_is_one_when_exact_and_zero_when_it_matches_the_null():
    y = np.array([0.1, 0.4, 0.9, 0.3])
    null = np.full(4, 0.425)
    assert skill_h(y, y, null, horizon=4) == pytest.approx(1.0)
    assert skill_h(y, null, null, horizon=4) == pytest.approx(0.0)


def test_one_step_skill_is_refused():
    y = np.array([0.1, 0.4])
    with pytest.raises(ValueError, match="one-step error stays banned"):
        skill_h(y, y, np.zeros(2), horizon=1)


def test_degenerate_readout_raises_rather_than_scoring():
    y = np.array([0.5, 0.5, 0.5])
    with pytest.raises(DegenerateNullError):
        skill_h(y, y, y.copy(), horizon=4)


def test_shape_mismatch_is_refused():
    with pytest.raises(ValueError, match="shape mismatch"):
        skill_h(np.zeros(3), np.zeros(2), np.ones(3), horizon=4)


# ---------------------------------------------------------------- bootstrap

def test_bootstrap_separates_a_planted_effect_from_a_planted_zero():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(30), 4)
    null_effect = rng.normal(0.0, 0.05, size=groups.size)
    real_effect = null_effect + 0.5
    assert bootstrap_ci(null_effect, groups, seed=1)["excludes_zero"] is False
    assert bootstrap_ci(real_effect, groups, seed=1)["excludes_zero"] is True


def test_bootstrap_resamples_groups_not_rows():
    # Two groups, constant within each: resampling groups can only ever produce
    # the means {0, 0.5, 1}. A row bootstrap would produce many more.
    groups = np.array(["a", "a", "a", "b", "b", "b"])
    values = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    out = bootstrap_ci(values, groups, seed=3, n_resamples=200)
    assert out["ci_low"] == pytest.approx(0.0) and out["ci_high"] == pytest.approx(1.0)
    assert out["n_groups"] == 2 and out["n_rows"] == 6


def test_bootstrap_is_reproducible_and_checks_its_inputs():
    groups = np.repeat(np.arange(5), 2)
    values = np.arange(10, dtype=float)
    a = bootstrap_ci(values, groups, seed=7, n_resamples=100)
    b = bootstrap_ci(values, groups, seed=7, n_resamples=100)
    assert a == b
    with pytest.raises(ValueError, match="rows"):
        bootstrap_ci(values, groups[:-1], seed=7)
    with pytest.raises(ValueError, match=">= 2 groups"):
        bootstrap_ci(np.zeros(3), np.zeros(3), seed=7)


# ---------------------------------------------------------------- seeds

def test_two_seeds_are_refused_three_are_not():
    with pytest.raises(ValueError, match="requires >= 3"):
        seed_aggregate([0.1, 0.2], ddof=1)
    out = seed_aggregate([0.1, 0.2, 0.3], ddof=1)
    assert out["n_seeds"] == 3 and out["mean"] == pytest.approx(0.2)


def test_ddof_has_no_default_and_changes_the_bar():
    with pytest.raises(TypeError):
        seed_aggregate([0.1, 0.2, 0.3])
    sample = seed_aggregate([0.1, 0.2, 0.3], ddof=1)["std"]
    population = seed_aggregate([0.1, 0.2, 0.3], ddof=0)["std"]
    assert sample > population


# ---------------------------------------------------------------- folds

def test_folds_are_group_disjoint_and_cover_every_row():
    groups = np.repeat(np.arange(12), 3)
    folds = make_folds(groups, 4, purpose="report", seed=0)
    assert len(folds) == 4
    covered = np.concatenate([f["test_rows"] for f in folds])
    assert np.array_equal(np.sort(covered), np.arange(groups.size))
    for f in folds:
        assert not set(groups[f["train_rows"]]) & set(groups[f["test_rows"]])


def test_report_folds_differ_from_gate_folds_at_the_same_seed():
    groups = np.repeat(np.arange(20), 2)
    gate = make_folds(groups, 5, purpose="gate", seed=11)
    report = make_folds(groups, 5, purpose="report", seed=11)
    as_set = lambda folds: {frozenset(f["test_groups"].tolist()) for f in folds}
    assert as_set(gate) != as_set(report)


def test_folds_are_deterministic_and_check_their_inputs():
    groups = np.repeat(np.arange(6), 2)
    a = make_folds(groups, 3, purpose="gate", seed=5)
    b = make_folds(groups, 3, purpose="gate", seed=5)
    assert all(np.array_equal(x["test_rows"], y["test_rows"]) for x, y in zip(a, b))
    with pytest.raises(ValueError, match="purpose must be"):
        make_folds(groups, 3, purpose="headline", seed=5)
    with pytest.raises(ValueError, match="at least 2"):
        make_folds(groups, 1, purpose="gate", seed=5)
    with pytest.raises(ValueError, match="exceeds"):
        make_folds(groups, 99, purpose="gate", seed=5)


def test_skill_from_squared_errors_is_the_same_definition_as_skill_h():
    from surrogate_eval import skill_from_squared_errors

    rng = np.random.default_rng(4)
    y = rng.normal(size=50)
    pred, null = y + rng.normal(0, 0.3, 50), y + rng.normal(0, 0.9, 50)
    assert skill_from_squared_errors((pred - y) ** 2, (null - y) ** 2, horizon=4) == pytest.approx(
        skill_h(y, pred, null, horizon=4)
    )


def test_skill_from_squared_errors_keeps_both_guards():
    from surrogate_eval import skill_from_squared_errors

    with pytest.raises(ValueError, match="one-step error stays banned"):
        skill_from_squared_errors(np.ones(3), np.ones(3), horizon=1)
    with pytest.raises(DegenerateNullError):
        skill_from_squared_errors(np.ones(3), np.zeros(3), horizon=4)
