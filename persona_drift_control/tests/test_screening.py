import json

from persona_drift.screening import _prepare_resumable_trajectories_file


def _row(trajectory_id, turn, prompt_id="p1", topic="weekend hiking"):
    return {
        "trajectory_id": trajectory_id,
        "system_prompt_id": prompt_id,
        "topic": topic,
        "turn": turn,
        "y_probe": 0.5,
    }


def test_no_existing_file_returns_empty_and_creates_nothing(tmp_path):
    path = tmp_path / "trajectories.jsonl"
    completed, topics = _prepare_resumable_trajectories_file(path, expected_rows_per_trajectory=3)
    assert completed == {}
    assert topics == {}
    assert path.read_text() == ""


def test_complete_trajectories_are_kept_and_rewritten_in_turn_order(tmp_path):
    path = tmp_path / "trajectories.jsonl"
    # deliberately out of turn order, to check the rewrite sorts them
    rows = [_row("t1", 2), _row("t1", 1), _row("t1", 3)]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    completed, topics = _prepare_resumable_trajectories_file(path, expected_rows_per_trajectory=3)

    assert list(completed.keys()) == ["t1"]
    assert [r["turn"] for r in completed["t1"]] == [1, 2, 3]
    assert topics == {"p1": "weekend hiking"}
    rewritten = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["turn"] for r in rewritten] == [1, 2, 3]


def test_partial_trajectory_is_dropped_not_kept(tmp_path):
    path = tmp_path / "trajectories.jsonl"
    rows = [_row("t1", 1), _row("t1", 2)]  # only 2 of 3 expected turns: killed mid-trajectory
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    completed, topics = _prepare_resumable_trajectories_file(path, expected_rows_per_trajectory=3)

    assert completed == {}
    assert topics == {}
    assert path.read_text() == ""  # partial rows are dropped, not left dangling


def test_mixed_complete_and_partial_trajectories(tmp_path):
    path = tmp_path / "trajectories.jsonl"
    rows = (
        [_row("t1", turn, prompt_id="p1") for turn in (1, 2, 3)]  # complete
        + [_row("t2", 1, prompt_id="p2")]  # partial
    )
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    completed, topics = _prepare_resumable_trajectories_file(path, expected_rows_per_trajectory=3)

    assert set(completed.keys()) == {"t1"}
    assert topics == {"p1": "weekend hiking"}
    rewritten_ids = {json.loads(line)["trajectory_id"] for line in path.read_text().splitlines()}
    assert rewritten_ids == {"t1"}
