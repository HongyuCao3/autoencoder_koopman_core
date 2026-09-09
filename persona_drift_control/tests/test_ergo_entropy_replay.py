"""Tests for the replay in scripts/analyze_ergo_entropy_readout.py.

The entropies this script reports are only meaningful if the replayed context
is the one that actually generated the recorded text. Both ways it can be
wrong are silent -- a missing system message or an append-mode reset just
produces different numbers, not an error -- so each gets a guard and a test.

test_legacy_replay_matches_the_pre_2026_09_08_inline_logic is the
regression check the code rules require: the already-published
outputs/ergo_math_phaseB_random_excite/entropy_readout.json must stay
reproducible from the default command line.
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "analyze_ergo_entropy_readout",
    pathlib.Path(__file__).resolve().parents[1] / "scripts" / "analyze_ergo_entropy_readout.py",
)
mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mod)

SYSTEM_ROLE = "system"


def _rows(profile: str | None, reset_mode: str | None, resets: tuple[int, ...] = ()) -> list[dict]:
    out = []
    for turn in (1, 2, 3):
        row = {"trajectory_id": "t0", "turn": turn, "u_reset": int(turn in resets),
               "user_message": f"u{turn}", "agent_message": f"a{turn}"}
        if profile is not None:
            row["prompt_profile"] = profile
        if reset_mode is not None:
            row["reset_mode"] = reset_mode
        out.append(row)
    return out


def _old_inline_replay(traj_rows: list[dict]) -> list[list[dict]]:
    """The logic as it stood before 2026-09-08, verbatim."""
    seen, agent_history = [], []
    for row in traj_rows:
        if row["u_reset"]:
            agent_history = [{"role": "user", "content": row["user_message"]}]
        else:
            agent_history.append({"role": "user", "content": row["user_message"]})
        seen.append([dict(m) for m in agent_history])
        agent_history = [*agent_history, {"role": "assistant", "content": row["agent_message"]}]
    return seen


def test_legacy_replay_matches_the_pre_2026_09_08_inline_logic():
    rows = _rows(profile=None, reset_mode=None, resets=(2,))
    new = [list(msgs) for _, msgs in mod.replay_histories(rows, "legacy")]
    assert new == _old_inline_replay(rows)


def test_upstream_replay_starts_from_the_system_message():
    rows = _rows(profile="upstream", reset_mode="append")
    histories = [list(msgs) for _, msgs in mod.replay_histories(rows, "upstream")]
    for msgs in histories:
        assert msgs[0]["role"] == SYSTEM_ROLE
    assert histories[0] == [histories[0][0], {"role": "user", "content": "u1"}]
    assert [m["content"] for m in histories[2][1:]] == ["u1", "a1", "u2", "a2", "u3"]


def test_legacy_replay_has_no_system_message():
    histories = [list(msgs) for _, msgs in mod.replay_histories(_rows("legacy", "overwrite"), "legacy")]
    assert all(m["role"] != SYSTEM_ROLE for msgs in histories for m in msgs)


def test_an_upstream_reset_keeps_the_system_message():
    rows = _rows(profile="upstream", reset_mode="overwrite", resets=(3,))
    histories = [list(msgs) for _, msgs in mod.replay_histories(rows, "upstream")]
    assert histories[2][0]["role"] == SYSTEM_ROLE
    assert [m["content"] for m in histories[2][1:]] == ["u3"]


def test_profile_mismatch_is_refused():
    with pytest.raises(SystemExit, match="prompt_profile"):
        list(mod.replay_histories(_rows("upstream", "append"), "legacy"))


def test_append_mode_reset_is_refused():
    with pytest.raises(SystemExit, match="append"):
        list(mod.replay_histories(_rows("upstream", "append", resets=(2,)), "upstream"))


def test_group_trajectories_sorts_by_turn():
    rows = [{"trajectory_id": "t0", "turn": 3, "u_reset": 0, "user_message": "u3", "agent_message": "a3"},
            {"trajectory_id": "t0", "turn": 1, "u_reset": 0, "user_message": "u1", "agent_message": "a1"},
            {"trajectory_id": "t1", "turn": 1, "u_reset": 0, "user_message": "x", "agent_message": "y"}]
    groups = mod.group_trajectories(rows)
    assert [len(g) for g in groups] == [2, 1]
    assert [r["turn"] for r in groups[0]] == [1, 3]
