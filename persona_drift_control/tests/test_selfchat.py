import pytest

from persona_drift import selfchat
from persona_drift.control import ZeroControlController
from persona_drift.prompt_bank import PromptEntry
from persona_drift.selfchat import TrajectoryConfig, run_trajectory


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()


class FakeChatModel:
    """Duck-types the subset of ChatModel that run_trajectory calls, without
    needing torch/a real model loaded -- run_trajectory never checks
    isinstance."""

    def __init__(self, model_id="fake-model", reply="a generated reply"):
        self.model_id = model_id
        self.tokenizer = FakeTokenizer()
        self._reply = reply
        self.calls = 0

    def generate(self, messages, seed, config=None):
        self.calls += 1
        return self._reply


def _entry():
    return PromptEntry(
        prompt_id="character_traits_999",
        prompt_category="character_traits",
        system_prompt="You love tacos.",
        probe_question="What do you love?",
        score_fn=lambda x: 1.0 if "taco" in x.lower() else 0.0,
    )


def _config(user_mode, num_turns=3):
    return TrajectoryConfig(num_turns=num_turns, probe_repeats=1, user_mode=user_mode)


def test_live_mode_requires_a_user_sim():
    with pytest.raises(ValueError, match="user_mode='live'"):
        run_trajectory(
            agent=FakeChatModel(),
            user_sim=None,
            entry=_entry(),
            controller=ZeroControlController(),
            seed=0,
            topic="weekend hiking",
            trajectory_id="t1",
            topic_split="test",
            config=_config("live"),
        )


def test_scripted_mode_uses_the_script_instead_of_user_sim(monkeypatch):
    script = ["script turn one", "script turn two", "script turn three"]
    monkeypatch.setattr(selfchat, "load_user_script", lambda topic, seed: script)

    rows = run_trajectory(
        agent=FakeChatModel(),
        user_sim=None,
        entry=_entry(),
        controller=ZeroControlController(),
        seed=0,
        topic="weekend hiking",
        trajectory_id="t1",
        topic_split="test",
        config=_config("scripted"),
    )

    assert [row["user_message"] for row in rows] == script
    assert all(row["user_mode"] == "scripted" for row in rows)


def test_scripted_mode_rejects_a_too_short_script(monkeypatch):
    monkeypatch.setattr(selfchat, "load_user_script", lambda topic, seed: ["only one turn"])

    with pytest.raises(ValueError, match="only 1 turns"):
        run_trajectory(
            agent=FakeChatModel(),
            user_sim=None,
            entry=_entry(),
            controller=ZeroControlController(),
            seed=0,
            topic="weekend hiking",
            trajectory_id="t1",
            topic_split="test",
            config=_config("scripted", num_turns=3),
        )


def test_live_mode_still_calls_user_sim_each_turn():
    user_sim = FakeChatModel(reply="live user reply")
    rows = run_trajectory(
        agent=FakeChatModel(),
        user_sim=user_sim,
        entry=_entry(),
        controller=ZeroControlController(),
        seed=0,
        topic="weekend hiking",
        trajectory_id="t1",
        topic_split="test",
        config=_config("live"),
    )
    assert user_sim.calls == 3
    assert all(row["user_message"] == "live user reply" for row in rows)
    assert all(row["user_mode"] == "live" for row in rows)
