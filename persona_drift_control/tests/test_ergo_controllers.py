import pytest

from persona_drift.ergo_controllers import (
    FixedTAndLastController,
    RandSchedTAndLastController,
)


def _resets(controller, num_shards):
    """u_reset value for every turn 1..num_shards, mirroring the loop in
    ergo_math_trajectory.run_ergo_math_trajectory (history is unused by
    these controllers, so an empty list is fine)."""
    return [controller.next_u_remind(turn, []) for turn in range(1, num_shards + 1)]


# ---------------------------------------------------------------------------
# FixedTAndLastController
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_shards,t", [(4, 1), (4, 2), (6, 3), (8, 1), (12, 7)])
def test_fixed_t_and_last_exactly_two_resets(num_shards, t):
    controller = FixedTAndLastController(t=t, num_shards=num_shards)
    resets = _resets(controller, num_shards)
    assert sum(resets) == 2
    assert not controller.degenerate


@pytest.mark.parametrize("num_shards,t", [(4, 1), (6, 3), (8, 5)])
def test_fixed_t_and_last_last_turn_always_reset(num_shards, t):
    controller = FixedTAndLastController(t=t, num_shards=num_shards)
    assert controller.next_u_remind(num_shards, []) == 1


def test_fixed_t_and_last_degenerate_when_t_equals_num_shards():
    with pytest.warns(UserWarning):
        controller = FixedTAndLastController(t=4, num_shards=4)
    assert controller.degenerate is True
    assert controller.degenerate_reason == "t_equals_num_shards"
    resets = _resets(controller, 4)
    assert sum(resets) == 1
    assert controller.next_u_remind(4, []) == 1


def test_fixed_t_and_last_degenerate_when_t_exceeds_num_shards():
    with pytest.warns(UserWarning):
        controller = FixedTAndLastController(t=10, num_shards=4)
    assert controller.degenerate is True
    assert controller.degenerate_reason == "t_exceeds_num_shards"
    resets = _resets(controller, 4)
    assert sum(resets) == 1
    assert controller.next_u_remind(4, []) == 1


def test_fixed_t_and_last_name_and_protocol_shape():
    controller = FixedTAndLastController(t=2, num_shards=5)
    assert controller.name == "fixed_t2_and_last"
    assert hasattr(controller, "name")
    assert isinstance(controller.next_u_remind(1, []), int)


# ---------------------------------------------------------------------------
# RandSchedTAndLastController
# ---------------------------------------------------------------------------


def test_randsched_t_and_last_reproducible_same_seed():
    a = RandSchedTAndLastController(num_shards=6, seed=42)
    b = RandSchedTAndLastController(num_shards=6, seed=42)
    assert a.t == b.t
    assert a.turns == b.turns
    assert _resets(a, 6) == _resets(b, 6)


def test_randsched_t_and_last_last_turn_always_reset_and_total_two():
    for seed in range(20):
        controller = RandSchedTAndLastController(num_shards=6, seed=seed)
        assert controller.next_u_remind(6, []) == 1
        assert sum(_resets(controller, 6)) == 2


def test_randsched_t_and_last_first_reset_in_range():
    num_shards = 6
    for seed in range(200):
        controller = RandSchedTAndLastController(num_shards=num_shards, seed=seed)
        assert 1 <= controller.t <= num_shards - 1


def test_randsched_t_and_last_degenerate_when_num_shards_equals_one():
    with pytest.warns(UserWarning):
        controller = RandSchedTAndLastController(num_shards=1, seed=0)
    assert controller.degenerate is True
    assert controller.degenerate_reason == "num_shards_equals_1"
    assert controller.t is None
    assert controller.turns == (1,)
    assert controller.next_u_remind(1, []) == 1


def test_randsched_t_and_last_different_seeds_can_differ():
    num_shards = 8
    ts = {RandSchedTAndLastController(num_shards=num_shards, seed=seed).t for seed in range(30)}
    assert len(ts) > 1


def test_randsched_t_and_last_name_default():
    controller = RandSchedTAndLastController(num_shards=5, seed=1)
    assert controller.name == "randsched_t_and_last"
