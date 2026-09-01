from persona_drift.benign_bank import BenignEntry
from persona_drift.benign_trajectory import BENIGN_PLACEHOLDER_GOAL, run_benign_trajectory
from persona_drift.control import ConstantRemindController, ZeroControlController


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()


class FakeAgent:
    def __init__(self, reply="a generated reply", thinking="fake reasoning trace"):
        self.model_id = "fake-agent"
        self._reply = reply
        self._thinking = thinking
        self.calls = 0
        self.messages_seen: list[list[dict[str, str]]] = []
        self.tokenizer = FakeTokenizer()

    def generate(self, messages, seed, config=None, enable_thinking=None, return_thinking=False):
        self.calls += 1
        self.messages_seen.append([dict(m) for m in messages])
        if return_thinking:
            return self._reply, self._thinking
        return self._reply


class FakeJudge:
    """Distinguishes the two judge prompts by their distinctive rubric
    wording, so tests can assert y_safety and y_help are scored
    independently rather than both accidentally reading the same reply."""

    def __init__(self, safety_reply="5", help_reply="4"):
        self.model_id = "fake-judge"
        self._safety_reply = safety_reply
        self._help_reply = help_reply
        self.calls = 0
        self.prompts_seen: list[str] = []

    def generate(self, messages, seed, config=None, enable_thinking=None, return_thinking=False):
        self.calls += 1
        prompt = messages[0]["content"]
        self.prompts_seen.append(prompt)
        if "safety evaluator" in prompt:
            return self._safety_reply
        assert "response-quality evaluator" in prompt
        return self._help_reply


def _entry(num_turns=6):
    return BenignEntry(
        benign_id="mtbench_writing",
        category="writing",
        multi_turn_queries=tuple(f"benign turn {i}" for i in range(num_turns)),
    )


def test_run_benign_trajectory_replays_fixed_queries_and_scores_both_judges():
    agent = FakeAgent(reply="here is a complete answer")
    judge = FakeJudge(safety_reply="5", help_reply="4")

    rows = run_benign_trajectory(agent=agent, judge=judge, entry=_entry(num_turns=6), seed=0, trajectory_id="t1")

    assert len(rows) == 6
    assert [row["question"] for row in rows] == [f"benign turn {i}" for i in range(6)]
    assert all(row["y_safety"] == 1.0 for row in rows)  # "5" -> 1.0
    assert all(row["y_help"] == 0.75 for row in rows)  # "4" -> 0.75
    assert all(row["y_probe"] == row["y_safety"] for row in rows)
    assert agent.calls == 6
    assert judge.calls == 12  # safety + helpfulness per turn


def test_safety_judge_call_uses_the_benign_placeholder_goal():
    judge = FakeJudge()
    run_benign_trajectory(
        agent=FakeAgent(), judge=judge, entry=_entry(num_turns=1), seed=0, trajectory_id="t2"
    )
    safety_prompt = next(p for p in judge.prompts_seen if "safety evaluator" in p)
    assert BENIGN_PLACEHOLDER_GOAL in safety_prompt


def test_default_controller_is_zero_control_no_reminder_ever_inserted():
    agent = FakeAgent()
    rows = run_benign_trajectory(
        agent=agent, judge=FakeJudge(), entry=_entry(num_turns=6), seed=0, trajectory_id="t3"
    )
    assert all(row["u_remind"] == 0 for row in rows)
    assert all(row["excitation_design"] == "zero_control" for row in rows)
    assert all(row["inserted_reminder_text"] is None for row in rows)
    assert all(row["inserted_tokens"] == 0 for row in rows)
    assert agent.messages_seen[0][-1]["content"] == "benign turn 0"

    explicit_zero_rows = run_benign_trajectory(
        agent=FakeAgent(),
        judge=FakeJudge(),
        entry=_entry(num_turns=6),
        seed=0,
        trajectory_id="t3b",
        controller=ZeroControlController(),
    )
    assert [row["u_remind"] for row in rows] == [row["u_remind"] for row in explicit_zero_rows]


def test_constant_remind_controller_prepends_safety_reminder_every_turn():
    agent = FakeAgent()
    rows = run_benign_trajectory(
        agent=agent,
        judge=FakeJudge(),
        entry=_entry(num_turns=3),
        seed=0,
        trajectory_id="t4",
        controller=ConstantRemindController(),
    )
    assert all(row["u_remind"] == 1 for row in rows)
    assert all(row["inserted_reminder_text"] is not None for row in rows)
    assert all(row["inserted_tokens"] > 0 for row in rows)
    assert rows[0]["question"] == "benign turn 0"
    last_user_message = agent.messages_seen[0][-1]["content"]
    assert last_user_message.startswith(rows[0]["inserted_reminder_text"])
    assert last_user_message.endswith("benign turn 0")


def test_records_benign_metadata_on_every_row():
    entry = _entry(num_turns=3)
    rows = run_benign_trajectory(
        agent=FakeAgent(), judge=FakeJudge(), entry=entry, seed=7, trajectory_id="t5"
    )
    for row in rows:
        assert row["benign_id"] == entry.benign_id
        assert row["category"] == entry.category
        assert row["seed"] == 7
        assert row["trajectory_id"] == "t5"
