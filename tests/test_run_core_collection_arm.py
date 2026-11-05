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


def test_branch_schedule_is_in_range_reproducible_and_blind_to_the_target():
    seeds = arm.load_seed_rows(TASK)
    pts = arm.draw_branch_points(len(seeds), TASK, seed=0)
    assert {k for k, _ in pts} == set(arm.BRANCH_TURNS)
    asks = np.array([c for _, c in pts])
    assert asks.shape == (252, arm.N_BRANCHES) and asks.min() >= 5 and asks.max() <= 100
    assert pts == arm.draw_branch_points(len(seeds), TASK, seed=0)
    assert pts != arm.draw_branch_points(len(seeds), TASK, seed=1)
    target = np.array([r["target_raw_count"] for r in seeds], dtype=float)
    assert np.corrcoef(target, asks[:, 0])[0, 1] ** 2 < 0.05


def test_siblings_share_the_main_prefix_and_differ_only_in_the_number_asked():
    seed_row = arm.load_seed_rows(TASK)[0]
    prev = {**arm.score_row(TASK, seed_row, "a b c"), "topic": seed_row["topic"]}
    history = [(seed_row["raw_generation"], "fb1"), ("a b c", "main feedback asks for r")]
    main = arm.build_messages(seed_row, history)
    a, fa = arm.branch_messages(TASK, seed_row, history, prev, 17)
    b, fb = arm.branch_messages(TASK, seed_row, history, prev, 83)
    assert a[:-1] == b[:-1] == main[:-1]
    r = seed_row["target_raw_count"]
    assert f"exactly {r} words, and it had 3" in fa and "exactly 17 words" in fa
    assert fa.replace("17", "83") == fb


def test_cli_branch_requires_seed_and_thinking_off():
    assert _cli("--command-policy", "branch", "--disable-thinking").returncode != 0
    no_think = _cli("--command-policy", "branch", "--command-seed", "0")
    assert no_think.returncode != 0 and "requires --disable-thinking" in no_think.stderr


def test_dry_run_forks_on_the_landed_prefix():
    out = _cli("--command-policy", "branch", "--command-seed", "0", "--disable-thinking")
    assert out.returncode == 0, out.stderr
    assert "+756 generations" in out.stdout
    assert "252 fork prefixes equal the landed turn-k prompt" in out.stdout
    assert "0 hit an unwitnessed branch" in out.stdout


class _FakeLLM:
    """Stands in for vllm.LLM: the reply has ~0.7x the words the last user turn asks for."""

    def __init__(self, **_):
        pass

    def chat(self, prompts, sampling, **_):
        import re
        import types
        outs = []
        for msgs in prompts:
            ask = re.findall(r"exactly (\d+) words", msgs[-1]["content"])
            n = int(0.7 * int(ask[-1])) + 2 if ask else 9
            text = " ".join(["word"] * n)
            outs.append(types.SimpleNamespace(outputs=[types.SimpleNamespace(text=text, token_ids=[0] * n)]))
        return outs


def _run_stubbed(monkeypatch, out_dir, *extra):
    import json
    import types
    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(LLM=_FakeLLM, SamplingParams=lambda **k: k))
    monkeypatch.setattr(sys, "argv", ["x", "--task", TASK, "--agent-model", "stub", "--out-dir", str(out_dir),
                                      "--limit-trajectories", "12", "--disable-thinking", *extra])
    arm.main()
    rows = [json.loads(l) for l in (out_dir / "trajectories.jsonl").open()]
    bp = out_dir / "branch_points.jsonl"
    return rows, [json.loads(l) for l in bp.open()] if bp.exists() else None


def test_branch_run_keeps_the_error_corpus_and_writes_one_fork_per_trajectory(tmp_path, monkeypatch):
    err_rows, none = _run_stubbed(monkeypatch, tmp_path / "err")
    rows, points = _run_stubbed(monkeypatch, tmp_path / "br", "--command-policy", "branch", "--command-seed", "0")
    assert none is None
    strip = ("command_policy", "command_seed", "command_raw_count", "command_norm")
    assert [{k: v for k, v in r.items() if k not in strip} for r in rows] == err_rows
    assert len(points) == 12 and all(len(p["branches"]) == arm.N_BRANCHES for p in points)
    main = {(r["trajectory_id"], r["turn"]): r for r in rows}
    for p in points:
        k, tid = p["k"], p["trajectory_id"]
        assert k in arm.BRANCH_TURNS and p["command_policy"] == "branch" and p["disable_thinking"]
        assert len(p["git_sha"]) == 40
        assert p["prefix_state"]["y_k_minus_1"]["raw_count"] == main[(tid, k - 1)]["measured_raw_count"]
        assert (p["prefix_state"]["y_k_minus_2"] is None) == (k == 2)
        assert p["main_reply_k"]["raw_count"] == main[(tid, k)]["measured_raw_count"]
        for b in p["branches"]:
            assert b["prompt_messages"][:-1] == main[(tid, k)]["prompt_messages"][:-1]
            assert b["measured_raw_count"] == int(0.7 * b["command_raw_count"]) + 2
    cfg = __import__("json").loads((tmp_path / "br" / "run_config.json").read_text())
    assert cfg["command_policy"] == "branch" and cfg["n_branch_points"] == 12
    with pytest.raises(FileExistsError):
        _run_stubbed(monkeypatch, tmp_path / "br", "--command-policy", "branch", "--command-seed", "0")
