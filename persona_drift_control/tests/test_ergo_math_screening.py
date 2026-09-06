"""CPU-only orchestration tests for
ergo_math_screening.run_ergo_math_screening, mirroring
test_mc_sycophancy_screening.py's approach (ChatModel monkeypatched out, no
real torch/transformers model needed; uses the real vendored 103-item bank,
same as that file does for mc_sycophancy_bank.py)."""

from __future__ import annotations

import json

from persona_drift import ergo_math_screening
from persona_drift.chat_model import GenerationConfig
from persona_drift.control import ConstantRemindController
from persona_drift.ergo_math_bank import load_ergo_math_bank, select_screening_items
from persona_drift.ergo_math_trajectory import ErgoMathTrajectoryConfig


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
            return "Current answer: 42", "some reasoning trace"
        return "42"  # never reached by the regex-hit path above


def _expected_row_count(num_items: int, rng_seed: int, num_seeds: int) -> int:
    items = select_screening_items(load_ergo_math_bank(), num_items=num_items, rng_seed=rng_seed)
    return sum(len(item.shards) for item in items) * num_seeds


def test_run_ergo_math_screening_threads_enable_thinking_into_agent_chatmodel(tmp_path, monkeypatch):
    constructed: list[dict] = []

    class RecordingChatModel(_FakeChatModel):
        def __init__(self, model_id, device="cuda", dtype=None, enable_thinking=False):
            super().__init__(model_id, device=device, dtype=dtype, enable_thinking=enable_thinking)
            constructed.append({"model_id": model_id, "enable_thinking": enable_thinking})

    monkeypatch.setattr(ergo_math_screening, "ChatModel", RecordingChatModel)

    report = ergo_math_screening.run_ergo_math_screening(
        agent_model_id="fake-model",
        judge_model_id="fake-model",
        output_dir=tmp_path / "out",
        num_items=1,
        seeds=(0,),
        item_rng_seed=0,
        device="cpu",
        trajectory_config=ErgoMathTrajectoryConfig(agent_gen=GenerationConfig(max_new_tokens=16)),
        enable_thinking=True,
    )

    assert constructed == [{"model_id": "fake-model", "enable_thinking": True}]
    assert report["config"]["enable_thinking"] is True


def test_default_controller_factory_is_zero_control(tmp_path, monkeypatch):
    monkeypatch.setattr(ergo_math_screening, "ChatModel", _FakeChatModel)

    report = ergo_math_screening.run_ergo_math_screening(
        agent_model_id="fake-model",
        judge_model_id="fake-model",
        output_dir=tmp_path / "out",
        num_items=1,
        seeds=(0,),
        item_rng_seed=0,
        device="cpu",
        trajectory_config=ErgoMathTrajectoryConfig(agent_gen=GenerationConfig(max_new_tokens=16)),
    )
    assert report["config"]["controller"] == "zero_control"


def test_controller_factory_is_called_fresh_per_trajectory_with_its_own_seed(tmp_path, monkeypatch):
    monkeypatch.setattr(ergo_math_screening, "ChatModel", _FakeChatModel)
    seen_seeds: list[int] = []

    def factory(seed: int, entry_id: str = ""):
        seen_seeds.append(seed)
        return ConstantRemindController()

    report = ergo_math_screening.run_ergo_math_screening(
        agent_model_id="fake-model",
        judge_model_id="fake-model",
        output_dir=tmp_path / "out",
        num_items=2,
        seeds=(0, 1),
        item_rng_seed=0,
        device="cpu",
        trajectory_config=ErgoMathTrajectoryConfig(agent_gen=GenerationConfig(max_new_tokens=16)),
        controller_factory=factory,
    )
    assert seen_seeds == [0, 0, 1, 0, 1]
    assert report["config"]["controller"] == "constant_remind"


def test_report_and_trajectories_written_to_output_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(ergo_math_screening, "ChatModel", _FakeChatModel)
    out_dir = tmp_path / "out"

    report = ergo_math_screening.run_ergo_math_screening(
        agent_model_id="fake-model",
        judge_model_id="fake-model",
        output_dir=out_dir,
        num_items=2,
        seeds=(0,),
        item_rng_seed=0,
        device="cpu",
        trajectory_config=ErgoMathTrajectoryConfig(agent_gen=GenerationConfig(max_new_tokens=16)),
    )

    assert (out_dir / "trajectories.jsonl").exists()
    assert (out_dir / "ergo_math_screening_report.json").exists()
    assert (out_dir / "ergo_math_screening_report.md").exists()
    rows = [json.loads(line) for line in (out_dir / "trajectories.jsonl").read_text().splitlines()]
    assert len(rows) == _expected_row_count(num_items=2, rng_seed=0, num_seeds=1)
    assert "new_q1_escalation" in report
    assert "final_turn_success" in report


def test_resumability_skips_already_completed_trajectories(tmp_path, monkeypatch):
    monkeypatch.setattr(ergo_math_screening, "ChatModel", _FakeChatModel)
    out_dir = tmp_path / "out"

    ergo_math_screening.run_ergo_math_screening(
        agent_model_id="fake-model",
        judge_model_id="fake-model",
        output_dir=out_dir,
        num_items=1,
        seeds=(0,),
        item_rng_seed=0,
        device="cpu",
        trajectory_config=ErgoMathTrajectoryConfig(agent_gen=GenerationConfig(max_new_tokens=16)),
    )
    first_rows = (out_dir / "trajectories.jsonl").read_text()

    calls = []

    class CountingChatModel(_FakeChatModel):
        def generate(self, *a, **kw):
            calls.append(1)
            return super().generate(*a, **kw)

    monkeypatch.setattr(ergo_math_screening, "ChatModel", CountingChatModel)
    ergo_math_screening.run_ergo_math_screening(
        agent_model_id="fake-model",
        judge_model_id="fake-model",
        output_dir=out_dir,
        num_items=1,
        seeds=(0,),
        item_rng_seed=0,
        device="cpu",
        trajectory_config=ErgoMathTrajectoryConfig(agent_gen=GenerationConfig(max_new_tokens=16)),
    )
    assert calls == []
    assert (out_dir / "trajectories.jsonl").read_text() == first_rows
