"""Tests for scripts/aggregate_table2_defense.py and
scripts/check_phaseE_5seed_drift.py.

The guards get the weight here. Every failure these tests plant produces a
table that RENDERS FINE and means something else -- a column that silently
averages a self-judged arm with an independently judged one, two rows reading
the same controller directory, or an arm quietly missing a seed. A guard that
passes silently leaves its own failure mode invisible in the report, which is
the reasoning `run_config_guard`'s docstring already makes for the run side.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib

import numpy as np
import pytest

SCRIPTS = pathlib.Path(__file__).resolve().parents[1] / "scripts"
ATTACKS = [f"attack_{i}" for i in range(8)]
SEEDS = [0, 1, 2, 3, 4]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"_{name}", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rows(design: str, *, y: float = 0.8, judge: str = "Qwen/Qwen3-4B", model: str = "Qwen/Qwen3-4B",
          seeds=tuple(SEEDS), attacks=tuple(ATTACKS), u: int = 0, tokens: int = 0) -> list[dict]:
    return [
        {"attack_id": attack, "seed": seed, "turn": turn, "y_safety": y, "u_remind": u,
         "inserted_tokens": tokens, "excitation_design": design, "model": model,
         "judge_model": judge, "judge_parse_failure": False, "refusal_flag": False,
         "attacker_query": f"q{turn}", "agent_message": f"a{turn}",
         "decoding_config": {"temperature": 0.0}}
        for attack in attacks for seed in seeds for turn in (1, 2, 3, 4, 5)
    ]


def _write_arm(root: pathlib.Path, name: str, rows: list[dict]) -> pathlib.Path:
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "trajectories.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return directory


# ---------------------------------------------------------------- aggregation

def test_seed_std_is_the_sample_sd_signed_on_2026_09_13():
    """The table prints `ddof=1`. At n=5 the two conventions differ by
    sqrt(5/4) = 1.118, which is a visible width change and nothing else, so a
    silent flip back to ddof=0 would look like a tighter measurement."""

    module = _load("aggregate_table2_defense")
    assert module.REPORT_DDOF == 1
    arm = {"per_trajectory": {(attack, seed): {"terminal_y": float(seed)} for attack in ATTACKS for seed in SEEDS}}
    shared = sorted(arm["per_trajectory"])
    values = module.per_seed_means(arm, "terminal_y", SEEDS, shared)
    assert values == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert np.std(values, ddof=1) == pytest.approx(1.5811, abs=1e-4)
    assert np.std(values, ddof=0) == pytest.approx(1.4142, abs=1e-4)


def test_per_seed_means_collapse_attacks_before_the_std():
    """`n` in `mean +/- std (n)` is the seed count. Aggregating the 40
    trajectories directly would report a spread over attacks with n=40 on the
    label, which is the banned 'n is the sample count' caliber."""

    module = _load("aggregate_table2_defense")
    per_trajectory = {(attack, seed): {"terminal_y": 10.0 if attack == ATTACKS[0] else 0.0}
                      for attack in ATTACKS for seed in SEEDS}
    values = module.per_seed_means({"per_trajectory": per_trajectory}, "terminal_y", SEEDS, sorted(per_trajectory))
    # Every seed sees the same one loud attack, so the across-seed spread is 0
    # even though the across-trajectory spread is large.
    assert values == [1.25] * 5
    assert np.std(values, ddof=1) == 0.0


def test_guard_rejects_a_column_that_mixes_judges():
    module = _load("aggregate_table2_defense")
    arms = {
        "self": {"design": ["a"], "model": ["Qwen/Qwen3-4B"], "judge_model": ["Qwen/Qwen3-4B"],
                 "per_trajectory": {(a, s): {} for a in ATTACKS for s in SEEDS}},
        "indep": {"design": ["b"], "model": ["Qwen/Qwen3-4B"], "judge_model": ["Qwen/Qwen3-4B-Instruct-2507"],
                  "per_trajectory": {(a, s): {} for a in ATTACKS for s in SEEDS}},
    }
    with pytest.raises(SystemExit, match="do not share a judge"):
        module.guard(arms)


def test_guard_rejects_two_rows_reading_the_same_controller():
    """A stale `output_dir` reused for a second arm is a one-token mistake
    that produces two identical rows rather than an error."""

    module = _load("aggregate_table2_defense")
    arms = {
        name: {"design": ["fixed_schedule_t5"], "model": ["m"], "judge_model": ["m"],
               "per_trajectory": {(a, s): {} for a in ATTACKS for s in SEEDS}}
        for name in ("fixed_t5", "ours")
    }
    with pytest.raises(SystemExit, match="appears in 2 arms"):
        module.guard(arms)


def test_guard_rejects_a_ragged_attack_seed_grid():
    module = _load("aggregate_table2_defense")
    arms = {
        "full": {"design": ["a"], "model": ["m"], "judge_model": ["m"],
                 "per_trajectory": {(a, s): {} for a in ATTACKS for s in SEEDS}},
        "short": {"design": ["b"], "model": ["m"], "judge_model": ["m"],
                  "per_trajectory": {(a, s): {} for a in ATTACKS for s in SEEDS if s != 4}},
    }
    with pytest.raises(SystemExit, match="not on the same"):
        module.guard(arms)


def test_guard_rejects_fewer_than_three_seeds():
    """The caliber `.claude/global.md` bans outright -- and the exact hole G1
    was run to close on the two endpoint arms."""

    module = _load("aggregate_table2_defense")
    arms = {
        name: {"design": [name], "model": ["m"], "judge_model": ["m"],
               "per_trajectory": {(a, s): {} for a in ATTACKS for s in (0, 1)}}
        for name in ("a", "b")
    }
    with pytest.raises(SystemExit, match="requires >= 3"):
        module.guard(arms)


def test_guard_names_the_judge_kind_so_the_exception_cannot_travel_silently():
    module = _load("aggregate_table2_defense")
    same = {"design": ["a"], "model": ["Qwen/Qwen3-4B"], "judge_model": ["Qwen/Qwen3-4B"],
            "per_trajectory": {(a, s): {} for a in ATTACKS for s in SEEDS}}
    other = dict(same, design=["b"])
    assert module.guard({"x": same, "y": other})["judge_kind"] == "self"

    indep = {"design": ["a"], "model": ["Qwen/Qwen3-4B"], "judge_model": ["Qwen/Qwen3-4B-Instruct-2507"],
             "per_trajectory": {(a, s): {} for a in ATTACKS for s in SEEDS}}
    assert module.guard({"x": indep, "y": dict(indep, design=["b"])})["judge_kind"] == "independent"


def test_load_arm_reads_terminal_and_late_window_off_the_same_trajectory(tmp_path):
    module = _load("aggregate_table2_defense")
    rows = _rows("some_arm", u=1, tokens=36)
    for row in rows:
        row["y_safety"] = float(row["turn"])  # turn-shaped so the two windows must differ
    arm = module.load_arm("some_arm", _write_arm(tmp_path, "some_arm", rows))
    metrics = arm["per_trajectory"][(ATTACKS[0], 0)]
    assert metrics["terminal_y"] == 5.0
    assert metrics["late_y"] == pytest.approx(4.0)  # mean of turns 3, 4, 5
    assert metrics["n_reminders"] == 5.0
    assert metrics["inserted_tokens"] == 180.0


# ----------------------------------------------------------------- drift check

def test_drift_check_passes_on_an_identical_overlap(tmp_path):
    module = _load("check_phaseE_5seed_drift")
    archived = _write_arm(tmp_path, "archived", _rows("zero_control", seeds=(0, 1)))
    rerun = _write_arm(tmp_path, "rerun", _rows("zero_control"))
    result = module.compare(archived, rerun)
    assert result["identical"]
    assert result["n_shared_rows"] == len(ATTACKS) * 2 * 5
    assert result["rerun_only_seeds"] == [2, 3, 4]


def test_drift_check_catches_a_single_changed_message(tmp_path):
    """One flipped token out of 80 rows has to be enough -- a refactor that
    is behaviour-preserving on 79 of 80 rows is not behaviour-preserving."""

    module = _load("check_phaseE_5seed_drift")
    archived = _write_arm(tmp_path, "archived", _rows("zero_control", seeds=(0, 1)))
    drifted = _rows("zero_control")
    drifted[7]["agent_message"] = "something the refactor changed"
    rerun = _write_arm(tmp_path, "rerun", drifted)
    result = module.compare(archived, rerun)
    assert not result["identical"]
    assert result["n_mismatched"]["agent_message"] == 1
    assert result["examples"]["agent_message"][0]["rerun"] == "something the refactor changed"


def test_drift_check_refuses_two_directories_that_are_not_the_same_arm(tmp_path):
    module = _load("check_phaseE_5seed_drift")
    archived = _write_arm(tmp_path, "archived", _rows("zero_control", attacks=("only_here",), seeds=(0, 1)))
    rerun = _write_arm(tmp_path, "rerun", _rows("zero_control", attacks=("only_there",)))
    with pytest.raises(SystemExit, match="share no"):
        module.compare(archived, rerun)


def test_drift_check_refuses_a_non_unique_row_key(tmp_path):
    module = _load("check_phaseE_5seed_drift")
    rows = _rows("zero_control", seeds=(0, 1))
    duplicated = _write_arm(tmp_path, "dup", rows + rows[:1])
    with pytest.raises(SystemExit, match="not unique"):
        module.compare(duplicated, duplicated)
