"""Regression guard for conf/adversarial_screening.yaml + conf/generation/*
(docs/experiments/adversarial_screening_thinking_pilot.md): catches YAML
typos or a broken `generation` group reference without needing a GPU or
even scripts/run_adversarial_screening_hydra.py's @hydra.main entrypoint --
just composes the config the same way Hydra would at startup."""

from __future__ import annotations

import pathlib

from hydra import compose, initialize_config_dir

CONFIG_DIR = str(pathlib.Path(__file__).resolve().parents[1] / "conf")


def test_default_generation_group_is_no_thinking():
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(config_name="adversarial_screening")
    assert cfg.generation.name == "no_thinking"
    assert cfg.generation.enable_thinking is False
    assert cfg.generation.agent_max_new_tokens == 256


def test_generation_thinking_override_changes_enable_thinking_and_token_budget():
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(config_name="adversarial_screening", overrides=["generation=thinking"])
    assert cfg.generation.name == "thinking"
    assert cfg.generation.enable_thinking is True
    # Must be raised above the no_thinking baseline's 256: a </think> block
    # can eat the whole budget before the model reaches its final answer at
    # a token budget sized only for a non-thinking reply.
    assert cfg.generation.agent_max_new_tokens > 256


def test_top_level_defaults_are_unaffected_by_generation_override():
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(config_name="adversarial_screening", overrides=["generation=thinking"])
    assert cfg.agent_model == "Qwen/Qwen3-4B"
    assert cfg.num_attacks == 20
    assert list(cfg.seeds) == [0, 1]
