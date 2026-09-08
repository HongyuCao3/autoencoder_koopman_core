import pytest

from persona_drift.control import ConstantRemindController, ZeroControlController
from persona_drift.ergo_math_bank import GSM8KShardedItem
from persona_drift.ergo_math_trajectory import ErgoMathTrajectoryConfig, run_ergo_math_trajectory


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()


class FakeChatModel:
    def __init__(self, model_id="fake-model", reply="Current answer: 3360", thinking="fake reasoning"):
        self.model_id = model_id
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


def _entry(gold="3360"):
    return GSM8KShardedItem(
        item_id="ergo_GSM8K_1246",
        gold_answer=gold,
        shards=("shard one", "shard two", "shard three", "shard four"),
    )


def test_run_ergo_math_trajectory_produces_one_row_per_shard():
    agent = FakeChatModel()
    rows = run_ergo_math_trajectory(agent=agent, judge=agent, entry=_entry(), seed=0, trajectory_id="t1")
    assert len(rows) == 4
    assert [row["turn"] for row in rows] == [1, 2, 3, 4]
    assert all(row["num_shards"] == 4 for row in rows)
    assert all(row["y_task_success"] == 1.0 for row in rows)
    assert all(row["y_probe"] == row["y_task_success"] for row in rows)
    assert agent.calls == 4


def test_zero_control_never_resets_and_accumulates_history():
    agent = FakeChatModel()
    rows = run_ergo_math_trajectory(
        agent=agent, judge=agent, entry=_entry(), seed=0, trajectory_id="t2", controller=ZeroControlController()
    )
    assert all(row["u_reset"] == 0 for row in rows)
    assert all(row["excitation_design"] == "zero_control" for row in rows)
    # Turn 4's agent-facing history should still contain turn 1's shard text --
    # history was never wiped.
    last_call_messages = agent.messages_seen[-1]
    assert any("shard one" in m["content"] for m in last_call_messages if m["role"] == "user")
    # 4 user turns + 3 prior assistant replies + this turn's user message = 7
    assert len(last_call_messages) == 7


def test_constant_remind_controller_resets_every_turn_and_wipes_history():
    agent = FakeChatModel()
    rows = run_ergo_math_trajectory(
        agent=agent, judge=agent, entry=_entry(), seed=0, trajectory_id="t3", controller=ConstantRemindController()
    )
    assert all(row["u_reset"] == 1 for row in rows)
    assert all(row["excitation_design"] == "constant_remind" for row in rows)
    # Every reset turn should see exactly one user message (the consolidated
    # stimulus), never any prior assistant turns.
    for call_messages in agent.messages_seen:
        assert len(call_messages) == 1
        assert call_messages[0]["role"] == "user"
    # The 3rd (index 2) reset turn's consolidated message should mention every
    # shard revealed so far (turns 1-3), not just the newest one.
    third_turn_text = agent.messages_seen[2][0]["content"]
    assert "shard one" in third_turn_text
    assert "shard two" in third_turn_text
    assert "shard three" in third_turn_text
    assert "shard four" not in third_turn_text


def test_incorrect_answer_scores_zero_and_flags_no_parse_failure():
    agent = FakeChatModel(reply="Current answer: 1")
    rows = run_ergo_math_trajectory(agent=agent, judge=agent, entry=_entry(gold="3360"), seed=0, trajectory_id="t4")
    assert all(row["y_task_success"] == 0.0 for row in rows)
    assert all(row["judge_parse_failure"] is False for row in rows)


def test_unparseable_reply_flags_parse_failure():
    agent = FakeChatModel(reply="I need more information before I can answer.")
    rows = run_ergo_math_trajectory(agent=agent, judge=agent, entry=_entry(), seed=0, trajectory_id="t5")
    assert all(row["judge_parse_failure"] is True for row in rows)
    assert all(row["y_task_success"] == 0.0 for row in rows)


def test_records_item_and_run_metadata_on_every_row():
    entry = _entry()
    rows = run_ergo_math_trajectory(agent=FakeChatModel(), judge=FakeChatModel(), entry=entry, seed=7, trajectory_id="t6")
    for row in rows:
        assert row["item_id"] == entry.item_id
        assert row["gold_answer"] == entry.gold_answer
        assert row["seed"] == 7
        assert row["trajectory_id"] == "t6"


# docs/experiments/two_task_success_plan.md section 2 E1: reset_mode tests.


def test_reset_mode_overwrite_wipes_history_to_length_one():
    agent = FakeChatModel()
    config = ErgoMathTrajectoryConfig(reset_mode="overwrite")
    run_ergo_math_trajectory(
        agent=agent,
        judge=agent,
        entry=_entry(),
        seed=0,
        trajectory_id="t7",
        controller=ConstantRemindController(),
        config=config,
    )
    # Existing (byte-for-byte-unchanged) behavior: every reset turn's
    # agent-facing history is exactly the one consolidated user message.
    for call_messages in agent.messages_seen:
        assert len(call_messages) == 1
        assert call_messages[0]["role"] == "user"


def test_reset_mode_append_grows_history_monotonically():
    agent = FakeChatModel()
    config = ErgoMathTrajectoryConfig(reset_mode="append")
    run_ergo_math_trajectory(
        agent=agent,
        judge=agent,
        entry=_entry(),
        seed=0,
        trajectory_id="t8",
        controller=ConstantRemindController(),
        config=config,
    )
    lengths = [len(call_messages) for call_messages in agent.messages_seen]
    # Every turn appends the consolidated message onto the prior history
    # (which already grew by the assistant reply), so per-turn history
    # length must strictly increase, never reset back down to 1.
    assert lengths == sorted(lengths)
    assert all(later > earlier for earlier, later in zip(lengths, lengths[1:]))
    assert lengths[0] == 1


def test_reset_mode_overwrite_and_append_produce_same_stimulus_text():
    agent_overwrite = FakeChatModel()
    agent_append = FakeChatModel()
    run_ergo_math_trajectory(
        agent=agent_overwrite,
        judge=agent_overwrite,
        entry=_entry(),
        seed=0,
        trajectory_id="t9a",
        controller=ConstantRemindController(),
        config=ErgoMathTrajectoryConfig(reset_mode="overwrite"),
    )
    run_ergo_math_trajectory(
        agent=agent_append,
        judge=agent_append,
        entry=_entry(),
        seed=0,
        trajectory_id="t9b",
        controller=ConstantRemindController(),
        config=ErgoMathTrajectoryConfig(reset_mode="append"),
    )
    # The consolidated stimulus text itself must not depend on reset_mode --
    # only where it lands in agent_history differs. Compare each turn's
    # newest user message (last user-role message on each call).
    for overwrite_messages, append_messages in zip(agent_overwrite.messages_seen, agent_append.messages_seen):
        overwrite_stimulus = [m for m in overwrite_messages if m["role"] == "user"][-1]["content"]
        append_stimulus = [m for m in append_messages if m["role"] == "user"][-1]["content"]
        assert overwrite_stimulus == append_stimulus


def test_invalid_reset_mode_raises_value_error():
    with pytest.raises(ValueError):
        ErgoMathTrajectoryConfig(reset_mode="clobber")
