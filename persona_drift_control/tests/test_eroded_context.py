import json

from persona_drift.eroded_context import load_eroded_contexts


def _row(trajectory_id, attack_id, plain_query, turn, attacker_query, agent_message, y_safety, seed=0):
    return {
        "trajectory_id": trajectory_id,
        "attack_id": attack_id,
        "plain_query": plain_query,
        "turn": turn,
        "attacker_query": attacker_query,
        "agent_message": agent_message,
        "y_safety": y_safety,
        "seed": seed,
    }


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_builds_context_up_to_the_final_turn_and_isolates_the_final_query(tmp_path):
    path = tmp_path / "trajectories.jsonl"
    _write_jsonl(
        path,
        [
            _row("t0", "a0", "goal0", 1, "q1", "reply1", 1.0),
            _row("t0", "a0", "goal0", 2, "q2", "reply2", 0.8),
            _row("t0", "a0", "goal0", 3, "q3", "reply3", 0.5),
        ],
    )

    contexts = load_eroded_contexts(path, seed=0, max_final_turn_y_safety=0.8)

    assert len(contexts) == 1
    ctx = contexts[0]
    assert ctx["query_id"] == "t0"
    assert ctx["attack_id"] == "a0"
    assert ctx["plain_query"] == "goal0"
    assert ctx["final_turn_query"] == "q3"
    assert ctx["recorded_final_y_safety"] == 0.5
    assert ctx["context_turns"] == 2
    assert ctx["context_messages"] == [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "reply1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "reply2"},
    ]


def test_filters_out_trajectories_already_at_ceiling(tmp_path):
    path = tmp_path / "trajectories.jsonl"
    _write_jsonl(
        path,
        [
            _row("t0", "a0", "goal0", 1, "q1", "reply1", 1.0),
            _row("t0", "a0", "goal0", 2, "q2", "reply2", 1.0),  # final turn still at ceiling
            _row("t1", "a1", "goal1", 1, "q1", "reply1", 1.0),
            _row("t1", "a1", "goal1", 2, "q2", "reply2", 0.3),  # eroded, kept
        ],
    )

    contexts = load_eroded_contexts(path, seed=0, max_final_turn_y_safety=0.8)

    assert [c["query_id"] for c in contexts] == ["t1"]


def test_filters_out_nan_final_turn_y_safety(tmp_path):
    path = tmp_path / "trajectories.jsonl"
    rows = [
        _row("t0", "a0", "goal0", 1, "q1", "reply1", 1.0),
        _row("t0", "a0", "goal0", 2, "q2", "reply2", float("nan")),
    ]
    _write_jsonl(path, rows)

    contexts = load_eroded_contexts(path, seed=0, max_final_turn_y_safety=0.8)

    assert contexts == []


def test_filters_to_the_requested_seed(tmp_path):
    path = tmp_path / "trajectories.jsonl"
    _write_jsonl(
        path,
        [
            _row("t0__seed0", "a0", "goal0", 1, "q1", "reply1", 1.0, seed=0),
            _row("t0__seed0", "a0", "goal0", 2, "q2", "reply2", 0.4, seed=0),
            _row("t0__seed1", "a0", "goal0", 1, "q1", "reply1", 1.0, seed=1),
            _row("t0__seed1", "a0", "goal0", 2, "q2", "reply2", 0.4, seed=1),
        ],
    )

    contexts = load_eroded_contexts(path, seed=0, max_final_turn_y_safety=0.8)

    assert [c["query_id"] for c in contexts] == ["t0__seed0"]


def test_single_turn_trajectories_are_skipped_no_context_available(tmp_path):
    path = tmp_path / "trajectories.jsonl"
    _write_jsonl(path, [_row("t0", "a0", "goal0", 1, "q1", "reply1", 0.2)])

    contexts = load_eroded_contexts(path, seed=0, max_final_turn_y_safety=0.8)

    assert contexts == []
