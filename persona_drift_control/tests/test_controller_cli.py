import json

import pytest

from persona_drift.control import KoopmanMPCController, PeriodicController
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
