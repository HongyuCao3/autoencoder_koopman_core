"""CPU-only orchestration test for the enable_thinking ablation
(docs/experiments/adversarial_screening_thinking_pilot.md): confirms
run_adversarial_screening actually threads enable_thinking into the
ChatModel it constructs for the agent, without needing a real
torch/transformers model -- ChatModel itself is monkeypatched out, same
approach as the existing FakeChatModel tests in test_attack_trajectory.py."""

from __future__ import annotations

import json

from persona_drift import adversarial_screening
from persona_drift.attack_trajectory import AttackTrajectoryConfig
from persona_drift.chat_model import GenerationConfig
from persona_drift.control import ConstantRemindController, RandomExciteController


class _FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()


class _FakeChatModel:
    def __init__(self, model_id, device="cuda", dtype=None, enable_thinking=False):
        self.model_id = model_id
        self.enable_thinking = enable_thinking
        self.tokenizer = _FakeTokenizer()

    def generate(self, messages, seed, config=None, enable_thinking=None, return_thinking=False):
        if return_thinking:
            return "a safe-looking reply", "some reasoning trace"
        return "5"  # judge digit


def test_run_adversarial_screening_threads_enable_thinking_into_agent_chatmodel(tmp_path, monkeypatch):
    constructed: list[dict] = []

    class RecordingChatModel(_FakeChatModel):
        def __init__(self, model_id, device="cuda", dtype=None, enable_thinking=False):
            super().__init__(model_id, device=device, dtype=dtype, enable_thinking=enable_thinking)
            constructed.append({"model_id": model_id, "enable_thinking": enable_thinking})

    monkeypatch.setattr(adversarial_screening, "ChatModel", RecordingChatModel)

    report = adversarial_screening.run_adversarial_screening(
        agent_model_id="fake-model",
        judge_model_id="fake-model",
        output_dir=tmp_path / "out",
        num_attacks=1,
        seeds=(0,),
        attack_rng_seed=0,
        device="cpu",
        trajectory_config=AttackTrajectoryConfig(agent_gen=GenerationConfig(max_new_tokens=16)),
        enable_thinking=True,
    )

    # judge_model_id == agent_model_id: only one ChatModel is constructed
    # (self-judging, per adversarial_screening.py), and it's the agent's
    # enable_thinking that must be True -- the judge's own calls are pinned
    # to False per-call regardless (see test_safety_judge.py).
    assert constructed == [{"model_id": "fake-model", "enable_thinking": True}]
    assert report["config"]["enable_thinking"] is True


def test_run_adversarial_screening_defaults_enable_thinking_to_false(tmp_path, monkeypatch):
    constructed: list[dict] = []

    class RecordingChatModel(_FakeChatModel):
        def __init__(self, model_id, device="cuda", dtype=None, enable_thinking=False):
            super().__init__(model_id, device=device, dtype=dtype, enable_thinking=enable_thinking)
            constructed.append({"model_id": model_id, "enable_thinking": enable_thinking})

    monkeypatch.setattr(adversarial_screening, "ChatModel", RecordingChatModel)

    report = adversarial_screening.run_adversarial_screening(
        agent_model_id="fake-model",
        judge_model_id="fake-model",
        output_dir=tmp_path / "out",
        num_attacks=1,
        seeds=(0,),
        attack_rng_seed=0,
        device="cpu",
        trajectory_config=AttackTrajectoryConfig(agent_gen=GenerationConfig(max_new_tokens=16)),
    )

    assert constructed == [{"model_id": "fake-model", "enable_thinking": False}]
    assert report["config"]["enable_thinking"] is False


def test_run_adversarial_screening_run_id_and_output_dir_differ_by_enable_thinking(tmp_path, monkeypatch):
    monkeypatch.setattr(adversarial_screening, "ChatModel", _FakeChatModel)
    logged_run_ids: list[str] = []
    monkeypatch.setattr(
        adversarial_screening,
        "configure_run_logger",
        lambda run_id, config, logs_dir=None: logged_run_ids.append(run_id),
    )

    common_kwargs = dict(
        agent_model_id="fake-model",
        judge_model_id="fake-model",
        num_attacks=1,
        seeds=(0,),
        attack_rng_seed=0,
        device="cpu",
        trajectory_config=AttackTrajectoryConfig(agent_gen=GenerationConfig(max_new_tokens=16)),
    )
    adversarial_screening.run_adversarial_screening(
        output_dir=tmp_path / "no_thinking", enable_thinking=False, **common_kwargs
    )
    adversarial_screening.run_adversarial_screening(
        output_dir=tmp_path / "thinking", enable_thinking=True, **common_kwargs
    )

    assert len(logged_run_ids) == 2
    assert "think0" in logged_run_ids[0]
    assert "think1" in logged_run_ids[1]
    assert logged_run_ids[0] != logged_run_ids[1]


def test_default_controller_factory_is_zero_control(tmp_path, monkeypatch):
    monkeypatch.setattr(adversarial_screening, "ChatModel", _FakeChatModel)

    report = adversarial_screening.run_adversarial_screening(
        agent_model_id="fake-model",
        judge_model_id="fake-model",
        output_dir=tmp_path / "out",
        num_attacks=1,
        seeds=(0,),
        attack_rng_seed=0,
        device="cpu",
        trajectory_config=AttackTrajectoryConfig(agent_gen=GenerationConfig(max_new_tokens=16)),
    )
    assert report["config"]["controller"] == "zero_control"


def test_controller_factory_is_called_fresh_per_trajectory_with_its_own_seed(tmp_path, monkeypatch):
    monkeypatch.setattr(adversarial_screening, "ChatModel", _FakeChatModel)
    seen_seeds: list[int] = []

    def factory(seed: int):
        seen_seeds.append(seed)
        return ConstantRemindController()

    report = adversarial_screening.run_adversarial_screening(
        agent_model_id="fake-model",
        judge_model_id="fake-model",
        output_dir=tmp_path / "out",
        num_attacks=2,
        seeds=(0, 1),
        attack_rng_seed=0,
        device="cpu",
        trajectory_config=AttackTrajectoryConfig(agent_gen=GenerationConfig(max_new_tokens=16)),
        controller_factory=factory,
    )
    # 2 attacks x 2 seeds = 4 trajectories, one fresh controller call each,
    # plus one extra upfront call with seeds[0] that run_adversarial_screening
    # makes solely to read `.name` for the run_id/config before the loop
    # starts (harmless for every real Controller in control.py: constructing
    # one is side-effect-free, and RandomExciteController's per-trajectory
    # instances are still each freshly seeded inside the loop regardless).
    assert seen_seeds == [0, 0, 1, 0, 1]
    assert report["config"]["controller"] == "constant_remind"


def test_random_excite_controller_is_independently_seeded_per_trajectory(tmp_path, monkeypatch):
    # A regression guard for the bug a single shared RandomExciteController
    # instance would have: its RNG must not run continuously across
    # trajectories (that would break both per-trajectory reproducibility
    # from its own seed and resumability) -- each trajectory's sequence must
    # depend only on that trajectory's own seed.
    monkeypatch.setattr(adversarial_screening, "ChatModel", _FakeChatModel)

    def factory(seed: int):
        return RandomExciteController(p=0.5, seed=seed)

    adversarial_screening.run_adversarial_screening(
        agent_model_id="fake-model",
        judge_model_id="fake-model",
        output_dir=tmp_path / "a",
        num_attacks=3,
        seeds=(0,),
        attack_rng_seed=0,
        device="cpu",
        trajectory_config=AttackTrajectoryConfig(agent_gen=GenerationConfig(max_new_tokens=16)),
        controller_factory=factory,
    )
    # Same seed (0) reused for every one of the 3 attacks above -> every
    # trajectory must draw the identical u_remind sequence, since each gets
    # its own fresh RandomExciteController(seed=0).
    rows_path = tmp_path / "a" / "trajectories.jsonl"
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    by_trajectory: dict[str, list[int]] = {}
    for row in sorted(rows, key=lambda r: (r["trajectory_id"], r["turn"])):
        by_trajectory.setdefault(row["trajectory_id"], []).append(row["u_remind"])
    sequences = list(by_trajectory.values())
    assert len(sequences) == 3
    assert sequences[0] == sequences[1] == sequences[2]
