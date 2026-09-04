import json

import pytest

from persona_drift.attack_bank import AttackEntry
from persona_drift.attack_trajectory import run_attack_trajectory
from persona_drift.rejudge import (
    judge_seed_for_row,
    pending_rows,
    rejudge_dirs,
    rejudge_file,
    rejudge_row,
)


class FakeJudge:
    """Returns a fixed digit per call, recording the seeds it was asked for."""

    def __init__(self, replies=("2",), model_id="fake-independent-judge"):
        self.model_id = model_id
        self._replies = list(replies)
        self.seeds = []
        self.prompts = []

    def generate(self, messages, seed, config=None, enable_thinking=None, return_thinking=False):
        self.seeds.append(seed)
        self.prompts.append(messages[0]["content"])
        return self._replies[(len(self.seeds) - 1) % len(self._replies)]


class FakeAgent:
    def __init__(self, model_id="fake-agent"):
        self.model_id = model_id
        self.tokenizer = None
        self.seeds = []

    def generate(self, messages, seed, config=None, enable_thinking=None, return_thinking=False):
        self.seeds.append(seed)
        text = f"reply to turn {len(self.seeds)}"
        return (text, "") if return_thinking else text


ENTRY = AttackEntry(
    attack_id="a1",
    category="illegal",
    actor_name="somebody",
    plain_query="the harmful goal",
    multi_turn_queries=("q1", "q2", "q3"),
)


def _self_judged_rows(judge_reply="5"):
    agent = FakeAgent()
    judge = FakeJudge((judge_reply,), model_id="fake-agent")  # self-judging: same id as the agent
    return run_attack_trajectory(agent, judge, ENTRY, seed=0, trajectory_id="a1__seed0"), judge


def test_rejudge_reuses_the_original_judge_seed():
    rows, self_judge = _self_judged_rows()
    independent = FakeJudge(("3",))
    for row in rows:
        rejudge_row(independent, row)
    # Not just "some seed": the same seeds the live run derived, so the two
    # score sets differ in the weights and nothing else.
    assert independent.seeds == self_judge.seeds
    assert independent.seeds == [judge_seed_for_row(row) for row in rows]


def test_rejudge_row_replaces_the_score_and_keeps_the_original():
    rows, _ = _self_judged_rows(judge_reply="5")
    rejudged = rejudge_row(FakeJudge(("1",)), rows[0])

    assert rejudged["y_safety"] == 0.0  # 1 -> fully complied
    assert rejudged["y_probe"] == 0.0  # the alias controllers read stays in sync
    assert rejudged["judge_model"] == "fake-independent-judge"
    assert rejudged["y_safety_self_judge"] == 1.0
    assert rejudged["judge_model_self_judge"] == "fake-agent"
    assert rejudged["judge_raw_output_self_judge"] == "5"
    assert rows[0]["y_safety"] == 1.0  # source row not mutated


def test_rejudge_row_copies_the_controller_decision_through_untouched():
    # The rejudge re-measures a trajectory; it does not re-run the
    # controller, and a rejudged row that quietly lost u_remind would make
    # the reminder-interaction analysis impossible.
    rows, _ = _self_judged_rows()
    rows[0]["u_remind"] = 1
    rows[0]["excitation_design"] = "threshold_budget1"
    rejudged = rejudge_row(FakeJudge(("4",)), rows[0])
    assert rejudged["u_remind"] == 1
    assert rejudged["excitation_design"] == "threshold_budget1"
    assert rejudged["attacker_query"] == rows[0]["attacker_query"]
    assert rejudged["agent_message"] == rows[0]["agent_message"]


def test_rejudge_row_records_parse_failures_like_the_live_judge():
    rows, _ = _self_judged_rows()
    rejudged = rejudge_row(FakeJudge(("no digit here",)), rows[0])
    assert rejudged["judge_parse_failure"] is True
    assert rejudged["y_safety"] != rejudged["y_safety"]  # nan
    assert rejudged["judge_raw_output"] == "no digit here"


def test_pending_rows_skips_done_and_redoes_changed_replies():
    rows, _ = _self_judged_rows()
    done = [dict(rows[0]), dict(rows[1])]
    done[1]["agent_message"] = "a different reply than the source now has"
    pending = pending_rows(rows, done)
    assert [(r["trajectory_id"], r["turn"]) for r in pending] == [("a1__seed0", 2), ("a1__seed0", 3)]


def test_rejudge_file_is_resumable_and_drops_stale_rows(tmp_path):
    rows, _ = _self_judged_rows()
    source = tmp_path / "trajectories.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    dest = tmp_path / "rejudged" / "trajectories.jsonl"

    first = FakeJudge(("1",))
    rejudge_file(first, source, dest, log_every=1)
    assert len(first.seeds) == 3

    # Second pass: nothing to do, no judge call at all.
    second = FakeJudge(("2",))
    carried = rejudge_file(second, source, dest, log_every=1)
    assert second.seeds == []
    assert [row["y_safety"] for row in carried] == [0.0, 0.0, 0.0]

    # Source row 2 gets a new reply (the arm was extended/re-run): only that
    # row is judged again, and the stale score does not survive.
    rows[1]["agent_message"] = "a new reply"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    third = FakeJudge(("5",))
    updated = rejudge_file(third, source, dest, log_every=1)
    assert len(third.seeds) == 1
    by_turn = {row["turn"]: row["y_safety"] for row in updated}
    assert by_turn == {1: 0.0, 2: 1.0, 3: 0.0}
    assert len(dest.read_text().strip().splitlines()) == 3


def test_rejudge_file_drops_rows_whose_source_disappeared(tmp_path):
    rows, _ = _self_judged_rows()
    source = tmp_path / "trajectories.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    dest = tmp_path / "rejudged" / "trajectories.jsonl"
    rejudge_file(FakeJudge(("1",)), source, dest, log_every=1)

    # prepare_resumable_trajectories_file truncates an incomplete trajectory
    # out of the source; the rejudged copy must follow, not keep orphans.
    source.write_text(json.dumps(rows[0]) + "\n")
    kept = rejudge_file(FakeJudge(("1",)), source, dest, log_every=1)
    assert len(kept) == 1
    assert len(dest.read_text().strip().splitlines()) == 1


def test_rejudge_dirs_skips_missing_arms_and_reports_a_manifest(tmp_path):
    rows, _ = _self_judged_rows()
    arm = tmp_path / "arm_a"
    arm.mkdir()
    (arm / "trajectories.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    missing = tmp_path / "arm_not_run_yet"
    missing.mkdir()

    judge = FakeJudge(("3",))
    manifest = rejudge_dirs(
        [arm, missing],
        judge_model_id="fake-independent-judge",
        out_subdir="rejudge_x",
        chat_model_cls=lambda model_id, device, enable_thinking: judge,
    )
    assert list(manifest) == [str(arm)]
    entry = manifest[str(arm)]
    assert entry["n_rows"] == 3
    assert entry["n_changed"] == 3  # 5 -> 3 on every row
    assert entry["self_judge_model"] == "fake-agent"
    assert (arm / "rejudge_x" / "trajectories.jsonl").exists()


def test_rejudge_dirs_raises_when_no_arm_has_data(tmp_path):
    with pytest.raises(SystemExit, match="no arm directory"):
        rejudge_dirs([tmp_path / "nothing"], judge_model_id="j", out_subdir="rejudge_x")


def test_load_jsonl_tolerates_a_torn_final_line_only(tmp_path):
    from persona_drift.rejudge import load_jsonl

    path = tmp_path / "trajectories.jsonl"
    path.write_text('{"trajectory_id": "t", "turn": 1}\n{"trajectory_id": "t", "tur')
    assert len(load_jsonl(path)) == 1

    path.write_text('{"trajectory_id": "t", "tur\n{"trajectory_id": "t", "turn": 2}\n')
    with pytest.raises(json.JSONDecodeError):
        load_jsonl(path)
