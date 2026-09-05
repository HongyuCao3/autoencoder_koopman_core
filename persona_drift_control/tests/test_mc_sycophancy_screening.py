"""CPU-only orchestration tests for
mc_sycophancy_screening.run_mc_sycophancy_screening, mirroring
test_sycophancy_screening.py's approach (ChatModel monkeypatched out, no
real torch/transformers model needed)."""

from __future__ import annotations

import json

from persona_drift import mc_sycophancy_screening
from persona_drift.chat_model import GenerationConfig
from persona_drift.control import ConstantRemindController
from persona_drift.mc_sycophancy_trajectory import MCSycophancyTrajectoryConfig


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
            # Regex-extractable, so these tests never need a judge model
            # call -- same reasoning test_mc_sycophancy_trajectory.py uses.
            return "Final answer: A", "some reasoning trace"
        return "A"  # never reached by the regex-hit path above


def test_run_mc_sycophancy_screening_threads_enable_thinking_into_agent_chatmodel(tmp_path, monkeypatch):
    constructed: list[dict] = []

    class RecordingChatModel(_FakeChatModel):
        def __init__(self, model_id, device="cuda", dtype=None, enable_thinking=False):
            super().__init__(model_id, device=device, dtype=dtype, enable_thinking=enable_thinking)
            constructed.append({"model_id": model_id, "enable_thinking": enable_thinking})

    monkeypatch.setattr(mc_sycophancy_screening, "ChatModel", RecordingChatModel)

    report = mc_sycophancy_screening.run_mc_sycophancy_screening(
        agent_model_id="fake-model",
        judge_model_id="fake-model",
        output_dir=tmp_path / "out",
        num_items=1,
        seeds=(0,),
        item_rng_seed=0,
        device="cpu",
        trajectory_config=MCSycophancyTrajectoryConfig(agent_gen=GenerationConfig(max_new_tokens=16)),
        enable_thinking=True,
    )

    assert constructed == [{"model_id": "fake-model", "enable_thinking": True}]
    assert report["config"]["enable_thinking"] is True


def test_default_controller_factory_is_zero_control(tmp_path, monkeypatch):
    monkeypatch.setattr(mc_sycophancy_screening, "ChatModel", _FakeChatModel)

    report = mc_sycophancy_screening.run_mc_sycophancy_screening(
        agent_model_id="fake-model",
        judge_model_id="fake-model",
        output_dir=tmp_path / "out",
        num_items=1,
        seeds=(0,),
        item_rng_seed=0,
        device="cpu",
        trajectory_config=MCSycophancyTrajectoryConfig(agent_gen=GenerationConfig(max_new_tokens=16)),
    )
    assert report["config"]["controller"] == "zero_control"


def test_controller_factory_is_called_fresh_per_trajectory_with_its_own_seed(tmp_path, monkeypatch):
    monkeypatch.setattr(mc_sycophancy_screening, "ChatModel", _FakeChatModel)
    seen_seeds: list[int] = []

    def factory(seed: int, entry_id: str = ""):
        seen_seeds.append(seed)
        return ConstantRemindController()

    report = mc_sycophancy_screening.run_mc_sycophancy_screening(
        agent_model_id="fake-model",
        judge_model_id="fake-model",
        output_dir=tmp_path / "out",
        num_items=2,
        seeds=(0, 1),
        item_rng_seed=0,
        device="cpu",
        trajectory_config=MCSycophancyTrajectoryConfig(agent_gen=GenerationConfig(max_new_tokens=16)),
        controller_factory=factory,
    )
    assert seen_seeds == [0, 0, 1, 0, 1]
    assert report["config"]["controller"] == "constant_remind"


def test_report_and_trajectories_written_to_output_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(mc_sycophancy_screening, "ChatModel", _FakeChatModel)
    out_dir = tmp_path / "out"

    report = mc_sycophancy_screening.run_mc_sycophancy_screening(
        agent_model_id="fake-model",
        judge_model_id="fake-model",
        output_dir=out_dir,
        num_items=2,
        seeds=(0,),
        item_rng_seed=0,
        device="cpu",
        trajectory_config=MCSycophancyTrajectoryConfig(agent_gen=GenerationConfig(max_new_tokens=16)),
    )

    assert (out_dir / "trajectories.jsonl").exists()
    assert (out_dir / "mc_sycophancy_screening_report.json").exists()
    assert (out_dir / "mc_sycophancy_screening_report.md").exists()
    rows = [json.loads(line) for line in (out_dir / "trajectories.jsonl").read_text().splitlines()]
    assert len(rows) == 2 * 5  # 2 items x 5 turns each, 1 seed
    assert "new_q1_escalation" in report
    assert "discrete_flip_events" in report


def test_resumability_skips_already_completed_trajectories(tmp_path, monkeypatch):
    monkeypatch.setattr(mc_sycophancy_screening, "ChatModel", _FakeChatModel)
    out_dir = tmp_path / "out"

    mc_sycophancy_screening.run_mc_sycophancy_screening(
        agent_model_id="fake-model",
        judge_model_id="fake-model",
        output_dir=out_dir,
        num_items=1,
        seeds=(0,),
        item_rng_seed=0,
        device="cpu",
        trajectory_config=MCSycophancyTrajectoryConfig(agent_gen=GenerationConfig(max_new_tokens=16)),
    )
    first_rows = (out_dir / "trajectories.jsonl").read_text()

    calls = []

    class CountingChatModel(_FakeChatModel):
        def generate(self, *a, **kw):
            calls.append(1)
            return super().generate(*a, **kw)

    monkeypatch.setattr(mc_sycophancy_screening, "ChatModel", CountingChatModel)
    mc_sycophancy_screening.run_mc_sycophancy_screening(
        agent_model_id="fake-model",
        judge_model_id="fake-model",
        output_dir=out_dir,
        num_items=1,
        seeds=(0,),
        item_rng_seed=0,
        device="cpu",
        trajectory_config=MCSycophancyTrajectoryConfig(agent_gen=GenerationConfig(max_new_tokens=16)),
    )
    assert calls == []
    assert (out_dir / "trajectories.jsonl").read_text() == first_rows
