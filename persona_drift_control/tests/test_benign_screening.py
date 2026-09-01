"""CPU-only orchestration test for run_benign_screening, mirroring
test_adversarial_screening.py's approach (ChatModel monkeypatched out, no
real torch/transformers model needed)."""

from __future__ import annotations

import json

from persona_drift import benign_screening
from persona_drift.benign_trajectory import BenignTrajectoryConfig
from persona_drift.chat_model import GenerationConfig
from persona_drift.control import ConstantRemindController


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
            return "a helpful-looking reply", "some reasoning trace"
        return "4"  # judge digit, works for either judge prompt


def test_default_controller_factory_is_zero_control(tmp_path, monkeypatch):
    monkeypatch.setattr(benign_screening, "ChatModel", _FakeChatModel)

    report = benign_screening.run_benign_screening(
        agent_model_id="fake-model",
        judge_model_id="fake-model",
        output_dir=tmp_path / "out",
        seeds=(0,),
        device="cpu",
        trajectory_config=BenignTrajectoryConfig(agent_gen=GenerationConfig(max_new_tokens=16)),
    )
    assert report["config"]["controller"] == "zero_control"
    # 8 categories x 1 seed = 8 trajectories
    assert len(report["config"]["benign_ids"]) == 8


def test_all_eight_categories_produce_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(benign_screening, "ChatModel", _FakeChatModel)

    benign_screening.run_benign_screening(
        agent_model_id="fake-model",
        judge_model_id="fake-model",
        output_dir=tmp_path / "out",
        seeds=(0,),
        device="cpu",
        trajectory_config=BenignTrajectoryConfig(agent_gen=GenerationConfig(max_new_tokens=16)),
        controller_factory=lambda seed: ConstantRemindController(),
    )
    rows = [json.loads(line) for line in (tmp_path / "out" / "trajectories.jsonl").read_text().splitlines()]
    categories = {row["category"] for row in rows}
    assert len(categories) == 8
    # each session is 6 turns (3 chained MT-Bench entries x 2 turns)
    assert len(rows) == 8 * 6
    assert all(row["u_remind"] == 1 for row in rows)


def test_resuming_skips_already_completed_trajectories(tmp_path, monkeypatch):
    monkeypatch.setattr(benign_screening, "ChatModel", _FakeChatModel)
    output_dir = tmp_path / "out"

    benign_screening.run_benign_screening(
        agent_model_id="fake-model",
        judge_model_id="fake-model",
        output_dir=output_dir,
        seeds=(0,),
        device="cpu",
        trajectory_config=BenignTrajectoryConfig(agent_gen=GenerationConfig(max_new_tokens=16)),
    )
    rows_after_first_run = (output_dir / "trajectories.jsonl").read_text().splitlines()

    calls: list[str] = []
    monkeypatch.setattr(
        benign_screening,
        "configure_run_logger",
        lambda run_id, config, logs_dir=None: calls.append(run_id),
    )
    report = benign_screening.run_benign_screening(
        agent_model_id="fake-model",
        judge_model_id="fake-model",
        output_dir=output_dir,
        seeds=(0,),
        device="cpu",
        trajectory_config=BenignTrajectoryConfig(agent_gen=GenerationConfig(max_new_tokens=16)),
    )
    rows_after_second_run = (output_dir / "trajectories.jsonl").read_text().splitlines()
    assert len(rows_after_first_run) == len(rows_after_second_run)
    assert report["diagnostics"]["n_rows"] == len(rows_after_second_run)
