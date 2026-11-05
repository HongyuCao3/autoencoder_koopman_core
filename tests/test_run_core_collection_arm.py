import importlib.util
import pathlib
import subprocess
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
_SPEC = importlib.util.spec_from_file_location("_run_core_collection_arm", ROOT / "scripts" / "run_core_collection_arm.py")
arm = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(arm)

TASK = "sentence_length_t10"


def _cli(*extra):
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_core_collection_arm.py"), "--task", TASK,
         "--agent-model", "none", "--out-dir", "/nonexistent", "--dry-run", *extra],
        capture_output=True, text=True)


def test_schedule_is_in_range_reproducible_and_seed_dependent():
    a = arm.draw_commands(252, 10, TASK, seed=0)
    assert len(a) == 252 and all(len(c) == 9 for c in a)
    flat = np.concatenate(a)
    assert flat.min() >= 5 and flat.max() <= 100
    assert a == arm.draw_commands(252, 10, TASK, seed=0)
    assert a != arm.draw_commands(252, 10, TASK, seed=1)


def test_the_command_carries_no_information_about_the_error():
    """The point of 4b: under `error` the command is r - y_t exactly (R^2 = 1);
    under `random` the turn-2 ask must be unpredictable from the turn-1 error."""
    seeds = arm.load_seed_rows(TASK)
    err = np.array([r["target_raw_count"] - arm.score_row(TASK, r, r["raw_generation"])["measured_raw_count"]
                    for r in seeds], dtype=float)
    ask = np.array([c[0] for c in arm.draw_commands(len(seeds), 10, TASK, seed=0)], dtype=float)
    r2 = np.corrcoef(err, ask)[0, 1] ** 2
    assert r2 < 0.05


def test_random_feedback_reports_the_previous_ask_and_sets_the_new_one():
    row = {"measured_raw_count": 12, "topic": "rivers"}
    text = arm.random_feedback_sentence_length(row, asked=30, command=71)
    assert "too short" in text and "exactly 30 words, and it had 12" in text and "exactly 71 words" in text
    near = arm.random_feedback_sentence_length({"measured_raw_count": 31, "topic": "rivers"}, asked=30, command=8)
    assert "within 1 word of the 30 words" in near and "exactly 8 words" in near


def test_error_policy_feedback_is_the_landed_builder():
    seed_row = arm.load_seed_rows(TASK)[0]
    row = arm.score_row(TASK, seed_row, seed_row["raw_generation"])
    assert arm.BUILDERS[TASK]({**seed_row, **row}) == seed_row["feedback_text"]


def test_cli_refuses_inconsistent_policy_flags():
    assert _cli("--command-policy", "random").returncode != 0
    assert _cli("--command-seed", "3").returncode != 0
    bad_task = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_core_collection_arm.py"), "--task", "even_odd_t5",
         "--agent-model", "none", "--out-dir", "/x", "--dry-run", "--command-policy", "random",
         "--command-seed", "0"], capture_output=True, text=True)
    assert bad_task.returncode != 0 and "no command range" in bad_task.stderr


def test_dry_run_exercises_the_random_schedule():
    out = _cli("--command-policy", "random", "--command-seed", "0", "--disable-thinking")
    assert out.returncode == 0, out.stderr
    assert "command schedule: 2268 draws in [5, 100]" in out.stdout
    assert "0 hit an unwitnessed branch" in out.stdout
