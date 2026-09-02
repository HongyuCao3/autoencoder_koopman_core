"""CPU-only orchestration tests for sycophancy_screening.run_sycophancy_screening,
mirroring test_adversarial_screening.py's approach (ChatModel monkeypatched
out, no real torch/transformers model needed)."""

from __future__ import annotations

import json

from persona_drift import sycophancy_screening
from persona_drift.chat_model import GenerationConfig
from persona_drift.control import ConstantRemindController
from persona_drift.sycophancy_trajectory import SycophancyTrajectoryConfig


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
            return "I'll stick with my original answer.", "some reasoning trace"
        return "MAINTAINS"  # judge label


def test_run_sycophancy_screening_threads_enable_thinking_into_agent_chatmodel(tmp_path, monkeypatch):
    constructed: list[dict] = []

    class RecordingChatModel(_FakeChatModel):
        def __init__(self, model_id, device="cuda", dtype=None, enable_thinking=False):
            super().__init__(model_id, device=device, dtype=dtype, enable_thinking=enable_thinking)
            constructed.append({"model_id": model_id, "enable_thinking": enable_thinking})

    monkeypatch.setattr(sycophancy_screening, "ChatModel", RecordingChatModel)

    report = sycophancy_screening.run_sycophancy_screening(
        agent_model_id="fake-model",
        judge_model_id="fake-model",
        output_dir=tmp_path / "out",
        num_items=1,
        seeds=(0,),
        item_rng_seed=0,
        device="cpu",
        trajectory_config=SycophancyTrajectoryConfig(agent_gen=GenerationConfig(max_new_tokens=16)),
        enable_thinking=True,
    )

    assert constructed == [{"model_id": "fake-model", "enable_thinking": True}]
    assert report["config"]["enable_thinking"] is True


def test_run_sycophancy_screening_defaults_enable_thinking_to_false(tmp_path, monkeypatch):
    constructed: list[dict] = []

    class RecordingChatModel(_FakeChatModel):
        def __init__(self, model_id, device="cuda", dtype=None, enable_thinking=False):
            super().__init__(model_id, device=device, dtype=dtype, enable_thinking=enable_thinking)
            constructed.append({"model_id": model_id, "enable_thinking": enable_thinking})

    monkeypatch.setattr(sycophancy_screening, "ChatModel", RecordingChatModel)

    report = sycophancy_screening.run_sycophancy_screening(
        agent_model_id="fake-model",
        judge_model_id="fake-model",
        output_dir=tmp_path / "out",
        num_items=1,
        seeds=(0,),
        item_rng_seed=0,
        device="cpu",
        trajectory_config=SycophancyTrajectoryConfig(agent_gen=GenerationConfig(max_new_tokens=16)),
    )

    assert constructed == [{"model_id": "fake-model", "enable_thinking": False}]
    assert report["config"]["enable_thinking"] is False


def test_default_controller_factory_is_zero_control(tmp_path, monkeypatch):
    monkeypatch.setattr(sycophancy_screening, "ChatModel", _FakeChatModel)

    report = sycophancy_screening.run_sycophancy_screening(
        agent_model_id="fake-model",
        judge_model_id="fake-model",
        output_dir=tmp_path / "out",
        num_items=1,
        seeds=(0,),
        item_rng_seed=0,
        device="cpu",
        trajectory_config=SycophancyTrajectoryConfig(agent_gen=GenerationConfig(max_new_tokens=16)),
    )
    assert report["config"]["controller"] == "zero_control"


def test_controller_factory_is_called_fresh_per_trajectory_with_its_own_seed(tmp_path, monkeypatch):
    monkeypatch.setattr(sycophancy_screening, "ChatModel", _FakeChatModel)
    seen_seeds: list[int] = []

    def factory(seed: int, entry_id: str = ""):
        seen_seeds.append(seed)
        return ConstantRemindController()

    report = sycophancy_screening.run_sycophancy_screening(
        agent_model_id="fake-model",
        judge_model_id="fake-model",
        output_dir=tmp_path / "out",
        num_items=2,
        seeds=(0, 1),
        item_rng_seed=0,
        device="cpu",
        trajectory_config=SycophancyTrajectoryConfig(agent_gen=GenerationConfig(max_new_tokens=16)),
        controller_factory=factory,
    )
    # 2 items x 2 seeds = 4 trajectories, one fresh controller call each,
    # plus one extra upfront call with seeds[0] read for .name before the
    # loop starts -- same accounting as
    # test_adversarial_screening.test_controller_factory_is_called_fresh_per_trajectory_with_its_own_seed.
    assert seen_seeds == [0, 0, 1, 0, 1]
    assert report["config"]["controller"] == "constant_remind"


def test_report_and_trajectories_written_to_output_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(sycophancy_screening, "ChatModel", _FakeChatModel)
    out_dir = tmp_path / "out"

    report = sycophancy_screening.run_sycophancy_screening(
        agent_model_id="fake-model",
        judge_model_id="fake-model",
        output_dir=out_dir,
        num_items=2,
        seeds=(0,),
        item_rng_seed=0,
        device="cpu",
        trajectory_config=SycophancyTrajectoryConfig(agent_gen=GenerationConfig(max_new_tokens=16)),
    )

    assert (out_dir / "trajectories.jsonl").exists()
    assert (out_dir / "sycophancy_screening_report.json").exists()
    assert (out_dir / "sycophancy_screening_report.md").exists()
    rows = [json.loads(line) for line in (out_dir / "trajectories.jsonl").read_text().splitlines()]
    assert len(rows) == 2 * 5  # 2 items x 5 turns each, 1 seed
    assert "new_q1_escalation" in report
    assert "discrete_flip_events" in report


def test_resumability_skips_already_completed_trajectories(tmp_path, monkeypatch):
    monkeypatch.setattr(sycophancy_screening, "ChatModel", _FakeChatModel)
    out_dir = tmp_path / "out"

    sycophancy_screening.run_sycophancy_screening(
        agent_model_id="fake-model",
        judge_model_id="fake-model",
        output_dir=out_dir,
        num_items=1,
        seeds=(0,),
        item_rng_seed=0,
        device="cpu",
        trajectory_config=SycophancyTrajectoryConfig(agent_gen=GenerationConfig(max_new_tokens=16)),
    )
    first_rows = (out_dir / "trajectories.jsonl").read_text()

    calls = []

    class CountingChatModel(_FakeChatModel):
        def generate(self, *a, **kw):
            calls.append(1)
            return super().generate(*a, **kw)

    monkeypatch.setattr(sycophancy_screening, "ChatModel", CountingChatModel)
    sycophancy_screening.run_sycophancy_screening(
        agent_model_id="fake-model",
        judge_model_id="fake-model",
        output_dir=out_dir,
        num_items=1,
        seeds=(0,),
        item_rng_seed=0,
        device="cpu",
        trajectory_config=SycophancyTrajectoryConfig(agent_gen=GenerationConfig(max_new_tokens=16)),
    )
    # already-completed trajectory -> no new agent/judge model even
    # constructed for the resumed run, let alone any generate() calls.
    assert calls == []
    assert (out_dir / "trajectories.jsonl").read_text() == first_rows
