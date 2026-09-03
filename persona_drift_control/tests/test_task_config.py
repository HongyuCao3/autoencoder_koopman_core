"""Regression guard for conf/task/*.yaml + the three task-scoped Hydra
entrypoints' top-level configs (conf/fit_koopman.yaml, conf/fit_koopman_ae.yaml,
conf/screening.yaml). Composes configs the same way Hydra would at startup,
no GPU/torch needed -- mirrors tests/test_hydra_config.py's approach.

The main thing worth guarding here: overriding one task's parameters must
never change another task's -- that's the whole point of splitting the
config into a `task` group (see docs/experiments/koopman_defense_pilot.md's
r/E discussion for why this was introduced)."""

from __future__ import annotations

import pathlib

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import MissingMandatoryValue

CONFIG_DIR = str(pathlib.Path(__file__).resolve().parents[1] / "conf")


def test_fit_koopman_default_task_is_defense_with_validated_values():
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(config_name="fit_koopman")
    assert cfg.task.name == "defense"
    assert cfg.task.fit.nu == 1
    assert cfg.task.fit.mu == 2
    assert cfg.task.fit.contemporaneous_v is True
    assert cfg.task.fit.y_col == "y_safety"
    assert cfg.task.fit.split_col == "attack_id"


def test_fit_koopman_task_override_switches_the_whole_bundle():
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(config_name="fit_koopman", overrides=["task=persona_drift"])
    assert cfg.task.name == "persona_drift"
    assert cfg.task.fit.mu == 1
    assert cfg.task.fit.contemporaneous_v is False
    assert cfg.task.fit.y_col == "y_probe"
    assert cfg.task.fit.split_col == "system_prompt_id"


def test_fit_koopman_param_override_does_not_leak_across_tasks():
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        persona_cfg = compose(
            config_name="fit_koopman", overrides=["task=persona_drift", "task.fit.learning_rate=5e-4"]
        )
        defense_cfg = compose(config_name="fit_koopman")  # separate compose call, no override
    assert persona_cfg.task.fit.learning_rate == 5e-4
    assert defense_cfg.task.fit.learning_rate == 1e-3  # unaffected by the other compose call


def test_sycophancy_task_composes_and_matches_persona_drift_defaults():
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(config_name="fit_koopman", overrides=["task=sycophancy"])
    assert cfg.task.name == "sycophancy"
    assert cfg.task.fit.nu == 1
    assert cfg.task.fit.mu == 1
    assert cfg.task.screening is None


def test_fit_koopman_ae_default_task_carries_ae_hyperparameters():
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(config_name="fit_koopman_ae")
    assert cfg.task.fit.hidden_dim == 4
    assert cfg.task.fit.num_layers == 1
    assert cfg.task.fit.learning_rate == 1e-3
    assert list(cfg.task.fit.latent_dims) == [1, 2, 4]


def test_screening_default_task_is_defense_with_koopman_mpc_defaults():
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(config_name="screening", overrides=["output_dir=outputs/test_run"])
    assert cfg.task.name == "defense"
    assert cfg.task.screening.controller == "threshold"
    assert cfg.task.screening.koopman_nu == 1
    assert cfg.task.screening.koopman_mu == 2
    assert cfg.task.screening.koopman_contemporaneous_v is True


def test_screening_requires_output_dir():
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(config_name="screening")
    with pytest.raises(MissingMandatoryValue):
        str(cfg.output_dir)


def test_screening_persona_drift_task_has_no_screening_schema():
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(config_name="screening", overrides=["task=persona_drift", "output_dir=outputs/test_run"])
    assert cfg.task.screening is None
