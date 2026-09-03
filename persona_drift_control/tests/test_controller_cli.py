import json

import pytest

from persona_drift.control import (
    BudgetLimitedController,
    FixedScheduleController,
    KoopmanMPCController,
    PeriodicController,
    RandomExciteController,
)
from persona_drift.controller_cli import load_koopman_mpc_interaction_controller, make_controller_factory
from persona_drift.modeling.interaction_lift import InteractionLiftedSurrogate


def test_make_controller_factory_periodic_builds_periodic_controller():
    factory = make_controller_factory("periodic", threshold_y_min=0.7, koopman_mpc_controller=None, periodic_period=3)
    controller = factory(seed=0)
    assert isinstance(controller, PeriodicController)
    assert controller.period == 3


def test_make_controller_factory_periodic_without_period_raises():
    with pytest.raises(ValueError):
        make_controller_factory("periodic", threshold_y_min=0.7, koopman_mpc_controller=None)


def test_make_controller_factory_random_excite_seeds_differently_per_entry():
    # Regression guard for the bug found while executing docs/next_step_diagnosis.md
    # (2026-09-02): passing only `seed` (not `entry_id`) into
    # RandomExciteController gave every entry sharing a `seed` the exact
    # same draw sequence -- confirmed live in Phase B's collected data.
    factory = make_controller_factory("random_excite", threshold_y_min=0.7, koopman_mpc_controller=None, random_excite_p=0.5)
    draws_a = [factory(seed=0, entry_id="attack_a").next_u_remind(t, []) for t in range(1, 6)]
    draws_b = [factory(seed=0, entry_id="attack_b").next_u_remind(t, []) for t in range(1, 6)]
    assert draws_a != draws_b  # same numeric seed, different entry_id -> different draws
    # reproducibility: same (seed, entry_id) -> same draws every time.
    draws_a_again = [factory(seed=0, entry_id="attack_a").next_u_remind(t, []) for t in range(1, 6)]
    assert draws_a == draws_a_again


def test_make_controller_factory_random_excite_ignores_entry_id_by_default():
    # `factory(seed)` (no entry_id) must still work -- every non-random_excite
    # caller relies on this.
    factory = make_controller_factory("random_excite", threshold_y_min=0.7, koopman_mpc_controller=None, random_excite_p=0.5)
    controller = factory(seed=0)
    assert isinstance(controller, RandomExciteController)


def test_make_controller_factory_koopman_mpc_interaction_returns_passed_controller():
    sentinel = object()
    factory = make_controller_factory(
        "koopman_mpc_interaction", threshold_y_min=0.7, koopman_mpc_controller=None, koopman_mpc_interaction_controller=sentinel
    )
    assert factory(seed=0) is sentinel


def test_load_koopman_mpc_interaction_controller_wraps_surrogate(tmp_path):
    report = {"model": {"A": [[0.9, 0.0, 0.0]] * 1 + [[0, 1, 0], [0, 0, 1]], "B": [[0.1, 0.2], [0.0, 0.0], [0.0, 0.0]], "b": [0.0, 0.0, 0.0], "C": [[1.0, 0.0, 0.0]]}}
    path = tmp_path / "interaction_model_report.json"
    path.write_text(json.dumps(report))

    controller = load_koopman_mpc_interaction_controller(path, nu=1, mu=2, horizon=2, repeat_penalty=0.15)

    assert isinstance(controller, KoopmanMPCController)
    assert isinstance(controller.surrogate, InteractionLiftedSurrogate)
    assert controller.horizon == 2
    assert controller.repeat_penalty == 0.15
    assert controller.state_config.nu == 1
    assert controller.state_config.mu == 2
    assert controller.state_config.contemporaneous_v is False  # default preserved


def test_load_koopman_mpc_interaction_controller_threads_contemporaneous_v(tmp_path):
    report = {"model": {"A": [[0.9, 0.0, 0.0]] * 1 + [[0, 1, 0], [0, 0, 1]], "B": [[0.1, 0.2], [0.0, 0.0], [0.0, 0.0]], "b": [0.0, 0.0, 0.0], "C": [[1.0, 0.0, 0.0]]}}
    path = tmp_path / "interaction_model_report.json"
    path.write_text(json.dumps(report))

    controller = load_koopman_mpc_interaction_controller(path, nu=1, mu=2, horizon=2, repeat_penalty=0.15, contemporaneous_v=True)

    assert controller.state_config.contemporaneous_v is True


def test_make_controller_factory_fixed_schedule_builds_the_listed_turns():
    factory = make_controller_factory(
        "fixed_schedule", threshold_y_min=0.7, koopman_mpc_controller=None, fixed_schedule_turns=(3,), remind_budget=1
    )
    controller = factory(seed=0)
    assert isinstance(controller, FixedScheduleController)
    assert controller.name == "fixed_schedule_t3"
    assert [t for t in range(1, 6) if controller.next_u_remind(t, [])] == [3]


def test_make_controller_factory_fixed_schedule_rejects_a_schedule_over_budget():
    # The whole point of this arm is being budget-matched to the adaptive one,
    # so an over-budget schedule is a config error, not something to truncate.
    with pytest.raises(ValueError, match="budget-matched"):
        make_controller_factory(
            "fixed_schedule",
            threshold_y_min=0.7,
            koopman_mpc_controller=None,
            fixed_schedule_turns=(2, 4),
            remind_budget=1,
        )


def test_make_controller_factory_wraps_model_free_controllers_in_the_budget():
    factory = make_controller_factory(
        "threshold", threshold_y_min=0.7, koopman_mpc_controller=None, remind_budget=1
    )
    controller = factory(seed=0)
    assert isinstance(controller, BudgetLimitedController)
    assert controller.name == "threshold_budget1"
    low = [{"y_probe": 0.1, "u_remind": 0}]
    assert controller.next_u_remind(2, low) == 1  # first dip: spend it
    assert controller.next_u_remind(3, low + [{"y_probe": 0.1, "u_remind": 1}]) == 0  # spent


def test_make_controller_factory_without_budget_is_unwrapped():
    controller = make_controller_factory("threshold", threshold_y_min=0.7, koopman_mpc_controller=None)(seed=0)
    assert not isinstance(controller, BudgetLimitedController)
    assert controller.name == "threshold"


def test_make_controller_factory_random_excite_rejects_a_budget():
    # Capping the draws would break the i.i.d. Bernoulli design the
    # identification fit assumes -- fail loudly rather than quietly biasing
    # a future Phase-B-style collection.
    with pytest.raises(ValueError, match="i.i.d."):
        make_controller_factory(
            "random_excite", threshold_y_min=0.7, koopman_mpc_controller=None, random_excite_p=0.5, remind_budget=1
        )


def test_load_koopman_mpc_interaction_controller_passes_budget_knobs_through(tmp_path):
    model_path = tmp_path / "interaction_model_report.json"
    model_path.write_text(
        json.dumps({"model": {"A": [[0.9]], "B": [[0.1, -0.2]], "b": [0.0], "C": [[1.0]]}})
    )
    controller = load_koopman_mpc_interaction_controller(
        model_path,
        nu=1,
        mu=0,
        horizon=5,
        repeat_penalty=0.0,
        contemporaneous_v=True,
        pad_short_history=True,
        remind_budget=1,
        episode_length=5,
        name="koopman_mpc_interaction",
    )
    assert controller.remind_budget == 1
    assert controller.episode_length == 5
    assert controller.pad_short_history is True
    assert controller.name == "koopman_mpc_interaction_budget1"


def test_load_koopman_mpc_interaction_controller_defaults_match_phase_a_to_i(tmp_path):
    # Unbudgeted loads must keep recording themselves as plain "koopman_mpc"
    # so nothing already collected has to be renamed to stay comparable.
    model_path = tmp_path / "interaction_model_report.json"
    model_path.write_text(
        json.dumps({"model": {"A": [[0.9]], "B": [[0.1, -0.2]], "b": [0.0], "C": [[1.0]]}})
    )
    controller = load_koopman_mpc_interaction_controller(
        model_path, nu=1, mu=0, horizon=2, repeat_penalty=0.0, contemporaneous_v=True
    )
    assert controller.name == "koopman_mpc"
    assert controller.remind_budget is None
    assert controller.episode_length is None
    assert controller.pad_short_history is False
