from persona_drift.control import (
    ConstantRemindController,
    PeriodicController,
    RandomExciteController,
    ThresholdController,
    ZeroControlController,
)


def _rows(*y_probes):
    return [{"y_probe": y} for y in y_probes]


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
