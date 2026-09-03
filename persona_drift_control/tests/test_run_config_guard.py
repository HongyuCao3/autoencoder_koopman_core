"""Guard for the guard: `run_config_guard` is what stops a Phase J arm from
resuming into another arm's output directory (conf/experiment/*.yaml,
docs/experiments/budget_constrained_defense_plan.md). If it silently passed,
the failure mode it exists to prevent -- one report averaging trajectories
produced by two different controllers -- would be invisible in the report
itself, so it is worth testing directly."""

from __future__ import annotations

import json

import pytest

from persona_drift.run_config_guard import RUN_CONFIG_FILENAME, describe_config_changes, guard_run_config

CONFIG = {"output_dir": "outputs/arm_a", "task": {"screening": {"controller": "fixed_schedule", "remind_budget": 1}}}


def test_first_run_records_the_config_and_creates_the_directory(tmp_path):
    output_dir = tmp_path / "arm_a"
    assert guard_run_config(output_dir, CONFIG) == []
    assert json.loads((output_dir / RUN_CONFIG_FILENAME).read_text()) == CONFIG


def test_identical_rerun_is_allowed(tmp_path):
    # Resuming a crashed/timed-out run of the SAME arm must keep working --
    # that's the behavior screening_common's resumability depends on.
    guard_run_config(tmp_path, CONFIG)
    assert guard_run_config(tmp_path, CONFIG) == []


def test_different_config_in_the_same_directory_raises_and_names_the_keys(tmp_path):
    guard_run_config(tmp_path, CONFIG)
    other = {**CONFIG, "task": {"screening": {"controller": "koopman_mpc_interaction", "remind_budget": 1}}}
    with pytest.raises(SystemExit) as excinfo:
        guard_run_config(tmp_path, other)
    message = str(excinfo.value)
    assert "task.screening.controller" in message
    assert "fixed_schedule" in message and "koopman_mpc_interaction" in message
    # The recorded config is left untouched, so the original run stays resumable.
    assert json.loads((tmp_path / RUN_CONFIG_FILENAME).read_text()) == CONFIG


def test_allow_config_change_overwrites_and_reports_the_changes(tmp_path):
    guard_run_config(tmp_path, CONFIG)
    other = {**CONFIG, "task": {"screening": {"controller": "fixed_schedule", "remind_budget": 2}}}
    changes = guard_run_config(tmp_path, other, allow_config_change=True)
    assert any("remind_budget" in line for line in changes)
    assert json.loads((tmp_path / RUN_CONFIG_FILENAME).read_text()) == other


def test_describe_config_changes_covers_added_and_removed_keys():
    changes = describe_config_changes({"a": 1, "b": {"c": 2}}, {"a": 1, "b": {"d": 3}})
    joined = "\n".join(changes)
    assert "b.c" in joined and "<absent>" in joined and "b.d" in joined
    assert "a" not in [line.split(":")[0].strip() for line in changes]
