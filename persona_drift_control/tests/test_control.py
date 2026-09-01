import numpy as np

from persona_drift.control import (
    ConstantRemindController,
    KoopmanMPCController,
    PeriodicController,
    RandomExciteController,
    ThresholdController,
    ZeroControlController,
)
from persona_drift.modeling.dataset import ReducedStateConfig
from persona_drift.modeling.koopman import KoopmanSurrogate, no_extra_features


def _rows(*y_probes):
    return [{"y_probe": y} for y in y_probes]


def _remind_rows(*y_and_u):
    """y_and_u: (y_probe, u_remind) pairs, oldest first."""
    return [{"y_probe": y, "u_remind": u} for y, u in y_and_u]


def _known_surrogate(a=0.9, g=0.2, c=0.0):
    """A KoopmanSurrogate with hand-set A/B/b/C (bypassing fit()) so
    KoopmanMPCController's action choice can be checked against a known,
    hand-computable model: y_(t+1) = a*y_t + g*u_t + c, readout y_t = z_t
    (nu=1, mu=0, no_extra_features -- z IS y, no lifting)."""

    model = KoopmanSurrogate(extra_features_fn=no_extra_features)
    model.state_dim = 1
    model.A = np.array([[a]])
    model.B = np.array([[g]])
    model.b = np.array([c])
    model.C = np.array([[1.0]])
    return model


def test_zero_control_always_zero():
    controller = ZeroControlController()
    assert controller.name == "zero_control"
    for turn in range(1, 17):
        assert controller.next_u_remind(turn, _rows(*([0.5] * (turn - 1)))) == 0


def test_constant_remind_always_one():
    controller = ConstantRemindController()
    assert controller.name == "constant_remind"
    assert controller.next_u_remind(1, []) == 1
    assert controller.next_u_remind(16, _rows(*([1.0] * 15))) == 1


def test_random_excite_is_bernoulli_and_seed_deterministic():
    a = RandomExciteController(p=0.5, seed=42)
    b = RandomExciteController(p=0.5, seed=42)
    seq_a = [a.next_u_remind(t, []) for t in range(1, 17)]
    seq_b = [b.next_u_remind(t, []) for t in range(1, 17)]
    assert seq_a == seq_b
    assert set(seq_a) <= {0, 1}


def test_random_excite_p_zero_and_one_are_deterministic():
    always_off = RandomExciteController(p=0.0, seed=0)
    always_on = RandomExciteController(p=1.0, seed=0)
    assert all(always_off.next_u_remind(t, []) == 0 for t in range(1, 17))
    assert all(always_on.next_u_remind(t, []) == 1 for t in range(1, 17))


def test_periodic_fires_only_on_multiples_of_period():
    controller = PeriodicController(period=4)
    assert controller.name == "periodic"
    fired_turns = [t for t in range(1, 17) if controller.next_u_remind(t, []) == 1]
    assert fired_turns == [4, 8, 12, 16]


def test_threshold_reminds_only_after_a_low_reading():
    controller = ThresholdController(y_min=0.7)
    assert controller.name == "threshold"
    assert controller.next_u_remind(1, []) == 0  # no history yet: don't act
    assert controller.next_u_remind(2, _rows(0.9)) == 0  # last reading above y_min
    assert controller.next_u_remind(2, _rows(0.5)) == 1  # last reading below y_min
    assert controller.next_u_remind(2, _rows(float("nan"))) == 0  # scorer failure: don't act


def test_threshold_only_looks_at_most_recent_reading():
    controller = ThresholdController(y_min=0.7)
    assert controller.next_u_remind(3, _rows(0.5, 0.9)) == 0
    assert controller.next_u_remind(3, _rows(0.9, 0.5)) == 1


def test_koopman_mpc_reminds_when_it_improves_predicted_safety_and_penalty_is_zero():
    # g=0.2 > 0: u_remind=1 always predicted to raise next-turn y_safety
    # relative to u_remind=0, all else equal -- with no cost for reminding,
    # the controller should always choose to remind.
    surrogate = _known_surrogate(a=0.9, g=0.2, c=0.0)
    controller = KoopmanMPCController(
        surrogate=surrogate, state_config=ReducedStateConfig(nu=1, mu=0), horizon=2, repeat_penalty=0.0
    )
    assert controller.name == "koopman_mpc"
    assert controller.next_u_remind(2, _remind_rows((0.5, 0))) == 1
    assert controller.next_u_remind(2, _remind_rows((0.9, 1))) == 1


def test_koopman_mpc_stops_reminding_when_the_repeat_penalty_dominates():
    # Same model as above (reminding still helps y_safety a little), but the
    # per-turn cost of reminding is set far larger than any possible safety
    # gain -- the optimal action must flip to 0.
    surrogate = _known_surrogate(a=0.9, g=0.2, c=0.0)
    controller = KoopmanMPCController(
        surrogate=surrogate, state_config=ReducedStateConfig(nu=1, mu=0), horizon=2, repeat_penalty=100.0
    )
    assert controller.next_u_remind(2, _remind_rows((0.5, 0))) == 0


def test_koopman_mpc_falls_back_to_zero_with_insufficient_history():
    surrogate = _known_surrogate()
    controller = KoopmanMPCController(surrogate=surrogate, state_config=ReducedStateConfig(nu=1, mu=1))
    assert controller.next_u_remind(1, []) == 0  # mu=1 needs >=1 prior row


def test_koopman_mpc_does_not_act_on_a_scorer_failure():
    surrogate = _known_surrogate()
    controller = KoopmanMPCController(surrogate=surrogate, state_config=ReducedStateConfig(nu=1, mu=0))
    assert controller.next_u_remind(2, _remind_rows((float("nan"), 0))) == 0
