import pytest

from persona_drift.control import PeriodicController
from persona_drift.controller_cli import make_controller_factory


def test_make_controller_factory_periodic_builds_periodic_controller():
    factory = make_controller_factory("periodic", threshold_y_min=0.7, koopman_mpc_controller=None, periodic_period=3)
    controller = factory(seed=0)
    assert isinstance(controller, PeriodicController)
    assert controller.period == 3


def test_make_controller_factory_periodic_without_period_raises():
    with pytest.raises(ValueError):
        make_controller_factory("periodic", threshold_y_min=0.7, koopman_mpc_controller=None)
