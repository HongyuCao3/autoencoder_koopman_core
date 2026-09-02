from persona_drift.control import ConstantRemindController, ZeroControlController
from persona_drift.sycophancy_bank import SycophancyItem
from persona_drift.sycophancy_trajectory import SycophancyTrajectoryConfig, run_sycophancy_trajectory


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()


class FakeChatModel:
    def __init__(self, model_id="fake-model", reply="a generated reply", thinking="fake reasoning trace"):
        self.model_id = model_id
        self._reply = reply
        self._thinking = thinking
        self.calls = 0
        self.enable_thinking_calls: list[bool | None] = []
        self.messages_seen: list[list[dict[str, str]]] = []
        self.tokenizer = FakeTokenizer()

    def generate(self, messages, seed, config=None, enable_thinking=None, return_thinking=False):
        self.calls += 1
        self.enable_thinking_calls.append(enable_thinking)
        self.messages_seen.append([dict(m) for m in messages])
        if return_thinking:
            return self._reply, self._thinking
        return self._reply


def _entry():
    return SycophancyItem(
        item_id="sycon_fp_0000",
        category="false_presupposition",
        question="Why does X happen?",
        presupposition="X is caused by Y (false).",
        correction="X is actually caused by Z.",
        pushback_turns=("pushback 1", "pushback 2", "pushback 3", "pushback 4"),
    )


def test_run_sycophancy_trajectory_replays_question_then_pushback_turns():
    agent = FakeChatModel(reply="I'll stick with my answer.")
    judge = FakeChatModel(reply="MAINTAINS")

    rows = run_sycophancy_trajectory(agent=agent, judge=judge, entry=_entry(), seed=0, trajectory_id="t1")

    assert len(rows) == 5
    assert [row["user_message"] for row in rows] == [
        "Why does X happen?",
        "pushback 1",
        "pushback 2",
        "pushback 3",
        "pushback 4",
    ]
    assert all(row["y_consistency"] == 1.0 for row in rows)
    assert all(row["stance_label"] == "MAINTAINS" for row in rows)
    assert all(row["is_flip"] is False for row in rows)
    assert all(row["y_probe"] == row["y_consistency"] for row in rows)
    assert agent.calls == 5
    assert judge.calls == 5


def test_flips_label_sets_is_flip_true_and_y_consistency_zero():
    rows = run_sycophancy_trajectory(
        agent=FakeChatModel(),
        judge=FakeChatModel(reply="FLIPS"),
        entry=_entry(),
        seed=0,
        trajectory_id="t2",
    )
    assert all(row["y_consistency"] == 0.0 for row in rows)
    assert all(row["stance_label"] == "FLIPS" for row in rows)
    assert all(row["is_flip"] is True for row in rows)


def test_unparseable_judge_output_leaves_stance_label_none_and_is_flip_false():
    rows = run_sycophancy_trajectory(
        agent=FakeChatModel(),
        judge=FakeChatModel(reply="I decline to answer."),
        entry=_entry(),
        seed=0,
        trajectory_id="t3",
    )
    assert all(row["judge_parse_failure"] is True for row in rows)
    assert all(row["stance_label"] is None for row in rows)
    assert all(row["is_flip"] is False for row in rows)


def test_records_item_metadata_on_every_row():
    entry = _entry()
    rows = run_sycophancy_trajectory(
        agent=FakeChatModel(),
        judge=FakeChatModel(reply="MAINTAINS"),
        entry=entry,
        seed=7,
        trajectory_id="t4",
        config=SycophancyTrajectoryConfig(),
    )
    for row in rows:
        assert row["item_id"] == entry.item_id
        assert row["category"] == entry.category
        assert row["presupposition"] == entry.presupposition
        assert row["correction"] == entry.correction
        assert row["seed"] == 7
        assert row["trajectory_id"] == "t4"


def test_default_controller_is_zero_control_no_reminder_ever_inserted():
    agent = FakeChatModel()
    rows = run_sycophancy_trajectory(
        agent=agent, judge=FakeChatModel(reply="MAINTAINS"), entry=_entry(), seed=0, trajectory_id="t5"
    )
    assert all(row["u_remind"] == 0 for row in rows)
    assert all(row["excitation_design"] == "zero_control" for row in rows)
    assert all(row["inserted_reminder_text"] is None for row in rows)
    assert all(row["inserted_tokens"] == 0 for row in rows)
    assert agent.messages_seen[0][-1]["content"] == "Why does X happen?"

    explicit_zero_rows = run_sycophancy_trajectory(
        agent=FakeChatModel(),
        judge=FakeChatModel(reply="MAINTAINS"),
        entry=_entry(),
        seed=0,
        trajectory_id="t5b",
        controller=ZeroControlController(),
    )
    assert [row["u_remind"] for row in rows] == [row["u_remind"] for row in explicit_zero_rows]


def test_constant_remind_controller_prepends_consistency_reminder_every_turn():
    agent = FakeChatModel()
    rows = run_sycophancy_trajectory(
        agent=agent,
        judge=FakeChatModel(reply="MAINTAINS"),
        entry=_entry(),
        seed=0,
        trajectory_id="t6",
        controller=ConstantRemindController(),
    )
    assert all(row["u_remind"] == 1 for row in rows)
    assert all(row["inserted_reminder_text"] is not None for row in rows)
    assert all(row["inserted_tokens"] > 0 for row in rows)
    assert rows[0]["user_message"] == "Why does X happen?"
    last_user_message = agent.messages_seen[0][-1]["content"]
    assert last_user_message.startswith(rows[0]["inserted_reminder_text"])
    assert last_user_message.endswith("Why does X happen?")
