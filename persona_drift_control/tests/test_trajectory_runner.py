from persona_drift.control import ConstantRemindController, ZeroControlController
from persona_drift.trajectory_runner import JudgeCall, run_reminder_gated_trajectory


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()


class FakeChatModel:
    def __init__(self, model_id="fake-model", reply="a reply", thinking="thinking"):
        self.model_id = model_id
        self._reply = reply
        self._thinking = thinking
        self.calls = 0
        self.tokenizer = FakeTokenizer()

    def generate(self, messages, seed, config=None, enable_thinking=None, return_thinking=False):
        self.calls += 1
        if return_thinking:
            return self._reply, self._thinking
        return self._reply


def _fixed_score_call(score, parse_failure=False, raw="raw"):
    def call(judge, entry, rows, turn, stimulus, response, seed, config):
        return score, parse_failure, raw

    return call


def test_replays_all_stimuli_and_tags_y_probe_from_primary_score_field():
    rows = run_reminder_gated_trajectory(
        agent=FakeChatModel(),
        judge=FakeChatModel(),
        entry=object(),
        seed=0,
        trajectory_id="t1",
        multi_turn_stimuli=("s1", "s2", "s3"),
        stimulus_field="stimulus_text",
        reminder_fn=lambda level: None,
        judge_calls=[
            JudgeCall("score_a", "fail_a", "raw_a", _fixed_score_call(0.75), config=None, seed_offset=2)
        ],
        primary_score_field="score_a",
        extra_row_fields_fn=lambda entry: {},
        agent_gen=_gen_config(),
        run_id="test",
    )
    assert len(rows) == 3
    assert [r["stimulus_text"] for r in rows] == ["s1", "s2", "s3"]
    assert all(r["y_probe"] == 0.75 for r in rows)
    assert all(r["score_a"] == 0.75 for r in rows)


def test_multiple_judge_calls_all_land_on_the_row_with_distinct_seeds():
    seen_seeds = []

    def judge_a(judge, entry, rows, turn, stimulus, response, seed, config):
        seen_seeds.append(("a", seed))
        return 1.0, False, "a-raw"

    def judge_b(judge, entry, rows, turn, stimulus, response, seed, config):
        seen_seeds.append(("b", seed))
        return 0.0, False, "b-raw"

    rows = run_reminder_gated_trajectory(
        agent=FakeChatModel(),
        judge=FakeChatModel(),
        entry=object(),
        seed=5,
        trajectory_id="t2",
        multi_turn_stimuli=("only turn",),
        stimulus_field="q",
        reminder_fn=lambda level: None,
        judge_calls=[
            JudgeCall("score_a", "fail_a", "raw_a", judge_a, config=None, seed_offset=2),
            JudgeCall("score_b", "fail_b", "raw_b", judge_b, config=None, seed_offset=3),
        ],
        primary_score_field="score_a",
        extra_row_fields_fn=lambda entry: {},
        agent_gen=_gen_config(),
        run_id="test",
    )
    assert rows[0]["score_a"] == 1.0
    assert rows[0]["score_b"] == 0.0
    assert rows[0]["y_probe"] == 1.0  # primary_score_field, not score_b
    # turn 1, seed=5: judge_seed = seed*1_000_000 + turn*100 + seed_offset
    assert ("a", 5_000_000 + 100 + 2) in seen_seeds
    assert ("b", 5_000_000 + 100 + 3) in seen_seeds


def test_extra_row_fields_fn_is_merged_into_every_row():
    rows = run_reminder_gated_trajectory(
        agent=FakeChatModel(),
        judge=FakeChatModel(),
        entry="the-entry",
        seed=0,
        trajectory_id="t3",
        multi_turn_stimuli=("s1", "s2"),
        stimulus_field="q",
        reminder_fn=lambda level: None,
        judge_calls=[JudgeCall("score_a", "fail_a", "raw_a", _fixed_score_call(1.0), config=None, seed_offset=2)],
        primary_score_field="score_a",
        extra_row_fields_fn=lambda entry: {"entry_echo": entry, "constant": 42},
        agent_gen=_gen_config(),
        run_id="test",
    )
    assert all(r["entry_echo"] == "the-entry" for r in rows)
    assert all(r["constant"] == 42 for r in rows)


def test_default_controller_is_zero_control():
    rows = run_reminder_gated_trajectory(
        agent=FakeChatModel(),
        judge=FakeChatModel(),
        entry=object(),
        seed=0,
        trajectory_id="t4",
        multi_turn_stimuli=("s1", "s2"),
        stimulus_field="q",
        reminder_fn=lambda level: "[reminder]" if level else None,
        judge_calls=[JudgeCall("score_a", "fail_a", "raw_a", _fixed_score_call(1.0), config=None, seed_offset=2)],
        primary_score_field="score_a",
        extra_row_fields_fn=lambda entry: {},
        agent_gen=_gen_config(),
        run_id="test",
    )
    assert all(r["u_remind"] == 0 for r in rows)
    assert all(r["inserted_reminder_text"] is None for r in rows)

    explicit_rows = run_reminder_gated_trajectory(
        agent=FakeChatModel(),
        judge=FakeChatModel(),
        entry=object(),
        seed=0,
        trajectory_id="t4b",
        multi_turn_stimuli=("s1", "s2"),
        stimulus_field="q",
        reminder_fn=lambda level: "[reminder]" if level else None,
        judge_calls=[JudgeCall("score_a", "fail_a", "raw_a", _fixed_score_call(1.0), config=None, seed_offset=2)],
        primary_score_field="score_a",
        extra_row_fields_fn=lambda entry: {},
        agent_gen=_gen_config(),
        run_id="test",
        controller=ZeroControlController(),
    )
    assert [r["u_remind"] for r in rows] == [r["u_remind"] for r in explicit_rows]


def test_constant_remind_controller_inserts_reminder_and_counts_tokens():
    agent = FakeChatModel()
    rows = run_reminder_gated_trajectory(
        agent=agent,
        judge=FakeChatModel(),
        entry=object(),
        seed=0,
        trajectory_id="t5",
        multi_turn_stimuli=("s1", "s2"),
        stimulus_field="q",
        reminder_fn=lambda level: "[reminder text]" if level else None,
        judge_calls=[JudgeCall("score_a", "fail_a", "raw_a", _fixed_score_call(1.0), config=None, seed_offset=2)],
        primary_score_field="score_a",
        extra_row_fields_fn=lambda entry: {},
        agent_gen=_gen_config(),
        run_id="test",
        controller=ConstantRemindController(),
    )
    assert all(r["u_remind"] == 1 for r in rows)
    assert all(r["inserted_reminder_text"] == "[reminder text]" for r in rows)
    assert all(r["inserted_tokens"] > 0 for r in rows)


def _gen_config():
    from persona_drift.chat_model import GenerationConfig

    return GenerationConfig(max_new_tokens=256)
