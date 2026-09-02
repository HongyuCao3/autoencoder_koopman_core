import json

import pytest

from persona_drift.control import KoopmanMPCController, PeriodicController, RandomExciteController
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
    # Regression guard for the bug found while executing docs/next step.md
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
