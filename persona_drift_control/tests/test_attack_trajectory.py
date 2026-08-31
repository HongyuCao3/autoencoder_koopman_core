from persona_drift.attack_bank import AttackEntry
from persona_drift.attack_trajectory import AttackTrajectoryConfig, run_attack_trajectory


class FakeChatModel:
    def __init__(self, model_id="fake-model", reply="a generated reply"):
        self.model_id = model_id
        self._reply = reply
        self.calls = 0

    def generate(self, messages, seed, config=None):
        self.calls += 1
        return self._reply


def _entry(num_turns=4):
    return AttackEntry(
        attack_id="safemtdata_0001",
        category="chemical_biological",
        actor_name="Some Actor",
        plain_query="do a harmful thing",
        multi_turn_queries=tuple(f"attacker turn {i}" for i in range(num_turns)),
    )


def test_run_attack_trajectory_replays_fixed_queries_and_scores_each_turn():
    agent = FakeChatModel(reply="I can't help with that.")
    judge = FakeChatModel(reply="5")

    rows = run_attack_trajectory(
        agent=agent,
        judge=judge,
        entry=_entry(num_turns=4),
        seed=0,
        trajectory_id="t1",
    )

    assert len(rows) == 4
    assert [row["attacker_query"] for row in rows] == [f"attacker turn {i}" for i in range(4)]
    assert all(row["agent_message"] == "I can't help with that." for row in rows)
    assert all(row["y_safety"] == 1.0 for row in rows)
    assert all(row["refusal_flag"] is True for row in rows)
    assert all(row["judge_parse_failure"] is False for row in rows)
    assert all(row["judge_raw_output"] == "5" for row in rows)
    assert agent.calls == 4
    assert judge.calls == 4


def test_run_attack_trajectory_uses_entrys_own_turn_count():
    rows = run_attack_trajectory(
        agent=FakeChatModel(),
        judge=FakeChatModel(reply="3"),
        entry=_entry(num_turns=5),
        seed=0,
        trajectory_id="t2",
    )
    assert len(rows) == 5
    assert [row["turn"] for row in rows] == [1, 2, 3, 4, 5]


def test_run_attack_trajectory_shares_a_single_model_when_agent_and_judge_are_the_same_instance():
    shared = FakeChatModel(reply="ok")
    rows = run_attack_trajectory(
        agent=shared,
        judge=shared,
        entry=_entry(num_turns=3),
        seed=0,
        trajectory_id="t3",
    )
    # 3 agent generations + 3 judge generations on the same fake instance
    assert shared.calls == 6
    assert all(row["model"] == row["judge_model"] for row in rows)


def test_run_attack_trajectory_records_attack_metadata_on_every_row():
    entry = _entry(num_turns=3)
    rows = run_attack_trajectory(
        agent=FakeChatModel(),
        judge=FakeChatModel(reply="3"),
        entry=entry,
        seed=7,
        trajectory_id="t4",
        config=AttackTrajectoryConfig(),
    )
    for row in rows:
        assert row["attack_id"] == entry.attack_id
        assert row["category"] == entry.category
        assert row["plain_query"] == entry.plain_query
        assert row["seed"] == 7
        assert row["trajectory_id"] == "t4"
