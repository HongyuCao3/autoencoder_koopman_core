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


# --- conf/experiment/* (Phase J, the budget-constrained setting) -----------
# These guard the property the whole group exists for: one experiment file =
# one arm = one output_dir, and composing an experiment must not change what
# `task=defense` alone means for every earlier phase's re-run.

PHASE_J_EXPERIMENTS = [
    "phaseJ_budget1_koopman",
    "phaseJ_budget1_threshold",
    *[f"phaseJ_budget1_fixed_t{turn}" for turn in range(1, 6)],
]


def test_screening_without_experiment_keeps_the_unbudgeted_defaults():
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(config_name="screening", overrides=["output_dir=outputs/tmp"])
    assert cfg.task.screening.remind_budget is None
    assert cfg.task.screening.episode_length is None
    assert cfg.task.screening.fixed_schedule_turns is None
    assert cfg.task.screening.koopman_pad_short_history is False
    assert cfg.task.screening.koopman_horizon == 2
    assert cfg.allow_config_change is False


def test_phase_j_experiments_pin_distinct_output_dirs():
    dirs = {}
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        for name in PHASE_J_EXPERIMENTS:
            cfg = compose(config_name="screening", overrides=[f"experiment={name}"])
            dirs[name] = cfg.output_dir
    assert len(set(dirs.values())) == len(PHASE_J_EXPERIMENTS), dirs
    # ...and none of them can collide with an already-collected phase.
    assert all("phaseJ" in output_dir for output_dir in dirs.values())


def test_phase_j_experiments_share_one_attack_set_budget_and_seeds():
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        configs = {
            name: compose(config_name="screening", overrides=[f"experiment={name}"])
            for name in PHASE_J_EXPERIMENTS
        }
    reference = configs["phaseJ_budget1_koopman"].task.screening
    for name, cfg in configs.items():
        assert list(cfg.task.screening.attack_ids) == list(reference.attack_ids), name
        assert cfg.task.screening.remind_budget == 1, name
        assert cfg.task.screening.episode_length == 5, name
        # One shared seed list, not a literal [0, 1]: what makes the arms
        # comparable is that they agree, and the list itself grew on
        # 2026-09-03 (docs/next_step_diagnosis.md section 4 step 3 -- 2 seeds
        # could not separate adjacent arms). The >= 5 floor is that step's
        # requirement, so shrinking the sweep back is a test failure rather
        # than a silent loss of power.
        assert list(cfg.task.screening.seeds) == list(reference.seeds), name
        assert len(reference.seeds) >= 5


def test_phase_j_koopman_arm_plans_to_the_end_of_the_episode():
    # horizon=2 (Phase I's value) cannot represent "save it for a worse turn"
    # at all, so this is the one setting the budgeted arm must not inherit.
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(config_name="screening", overrides=["experiment=phaseJ_budget1_koopman"])
    s = cfg.task.screening
    assert s.controller == "koopman_mpc_interaction"
    assert s.koopman_horizon >= s.episode_length
    assert s.koopman_pad_short_history is True
    assert s.koopman_contemporaneous_v is True
    assert s.koopman_interaction_model_path.endswith("interaction_model_report_valigned.json")


def test_phase_j_fixed_arms_spend_exactly_their_budget():
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        for turn in range(1, 6):
            cfg = compose(config_name="screening", overrides=[f"experiment=phaseJ_budget1_fixed_t{turn}"])
            assert cfg.task.screening.controller == "fixed_schedule"
            assert list(cfg.task.screening.fixed_schedule_turns) == [turn]


def test_experiment_override_does_not_leak_into_a_later_plain_compose():
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        compose(config_name="screening", overrides=["experiment=phaseJ_budget1_koopman"])
        plain = compose(config_name="screening", overrides=["output_dir=outputs/tmp"])
    assert plain.task.screening.remind_budget is None
    assert plain.task.screening.koopman_horizon == 2
