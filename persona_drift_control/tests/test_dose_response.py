import numpy as np

from persona_drift.chat_model import SteeringConfig
from persona_drift.dose_response import ALPHA_GRID, DoseResponseConfig, run_dose_response_query


class FakeChatModel:
    def __init__(self, model_id="fake-model", reply="a generated reply"):
        self.model_id = model_id
        self._reply = reply
        self.calls: list[dict] = []

    def generate(self, messages, seed, config=None, steering=None, enable_thinking=None, return_thinking=False):
        self.calls.append(
            {
                "messages": messages,
                "seed": seed,
                "config": config,
                "steering": steering,
                "enable_thinking": enable_thinking,
            }
        )
        if return_thinking:
            return self._reply, ""
        return self._reply


def test_run_dose_response_query_sweeps_the_default_alpha_grid():
    agent = FakeChatModel(reply="I can't help with that.")
    judge = FakeChatModel(reply="5")
    direction = np.ones(4)

    rows = run_dose_response_query(
        agent=agent,
        judge=judge,
        query_id="q0",
        harmful_goal="do a harmful thing",
        layer=3,
        direction=direction,
        seed=0,
    )

    assert len(rows) == len(ALPHA_GRID)
    assert [row["alpha"] for row in rows] == list(ALPHA_GRID)
    assert all(row["query_id"] == "q0" for row in rows)
    assert all(row["harmful_goal"] == "do a harmful thing" for row in rows)
    assert all(row["layer"] == 3 for row in rows)
    assert all(row["y_safety"] == 1.0 for row in rows)
    assert agent.calls[0]["messages"] == [{"role": "user", "content": "do a harmful thing"}]


def test_alpha_zero_skips_the_steering_hook_entirely():
    agent = FakeChatModel()
    judge = FakeChatModel(reply="3")
    direction = np.ones(4)

    run_dose_response_query(
        agent=agent, judge=judge, query_id="q0", harmful_goal="x", layer=1, direction=direction, seed=0
    )

    zero_alpha_call = agent.calls[ALPHA_GRID.index(0.0)]
    assert zero_alpha_call["steering"] is None


def test_nonzero_alpha_passes_a_steering_config_with_the_scaled_direction():
    agent = FakeChatModel()
    judge = FakeChatModel(reply="3")
    direction = np.array([1.0, 2.0])

    run_dose_response_query(
        agent=agent, judge=judge, query_id="q0", harmful_goal="x", layer=9, direction=direction, seed=0
    )

    first_alpha_call = agent.calls[0]  # ALPHA_GRID[0] == -1.0
    steering = first_alpha_call["steering"]
    assert isinstance(steering, SteeringConfig)
    assert steering.layer == 9
    assert steering.alpha == -1.0
    assert np.array_equal(steering.direction, direction)


def test_custom_alpha_grid_is_respected():
    agent = FakeChatModel()
    judge = FakeChatModel(reply="4")
    direction = np.ones(2)
    config = DoseResponseConfig(alpha_grid=(0.0, 2.0))

    rows = run_dose_response_query(
        agent=agent, judge=judge, query_id="q0", harmful_goal="x", layer=1, direction=direction, seed=0, config=config
    )

    assert [row["alpha"] for row in rows] == [0.0, 2.0]
    assert agent.calls[1]["steering"].alpha == 2.0


def test_context_messages_are_prepended_before_the_steered_turn():
    agent = FakeChatModel()
    judge = FakeChatModel(reply="3")
    direction = np.ones(2)
    context = [
        {"role": "user", "content": "turn1 attacker query"},
        {"role": "assistant", "content": "turn1 real reply"},
    ]

    run_dose_response_query(
        agent=agent,
        judge=judge,
        query_id="q0",
        harmful_goal="the underlying goal",
        layer=1,
        direction=direction,
        seed=0,
        context_messages=context,
        question_text="turn2 attacker query",
    )

    for call in agent.calls:
        assert call["messages"] == [*context, {"role": "user", "content": "turn2 attacker query"}]


def test_question_text_defaults_to_harmful_goal_reproducing_the_bare_ask():
    agent = FakeChatModel()
    judge = FakeChatModel(reply="3")
    direction = np.ones(2)

    run_dose_response_query(
        agent=agent, judge=judge, query_id="q0", harmful_goal="the goal", layer=1, direction=direction, seed=0
    )

    for call in agent.calls:
        assert call["messages"] == [{"role": "user", "content": "the goal"}]


def test_row_records_question_text_and_context_turns():
    agent = FakeChatModel()
    judge = FakeChatModel(reply="3")
    direction = np.ones(2)
    context = [
        {"role": "user", "content": "t1 q"},
        {"role": "assistant", "content": "t1 a"},
        {"role": "user", "content": "t2 q"},
        {"role": "assistant", "content": "t2 a"},
    ]

    rows = run_dose_response_query(
        agent=agent,
        judge=judge,
        query_id="q0",
        harmful_goal="goal",
        layer=1,
        direction=direction,
        seed=0,
        context_messages=context,
        question_text="t3 q",
    )

    assert all(row["question_text"] == "t3 q" for row in rows)
    assert all(row["context_turns"] == 2 for row in rows)
    assert all(row["harmful_goal"] == "goal" for row in rows)


def test_distinct_seeds_per_alpha_level():
    agent = FakeChatModel()
    judge = FakeChatModel(reply="3")
    direction = np.ones(2)

    run_dose_response_query(
        agent=agent, judge=judge, query_id="q0", harmful_goal="x", layer=1, direction=direction, seed=5
    )

    seeds_used = [call["seed"] for call in agent.calls]
    assert len(seeds_used) == len(set(seeds_used))
