import json

import numpy as np
import pytest

from persona_drift.continuous_readout import score_dirs, score_file, score_row
from persona_drift.sycophancy_judge import resolve_label_token_ids


class FakeTokenizer:
    def __init__(self, overrides):
        self._overrides = overrides

    def encode(self, text, add_special_tokens=False):
        return list(self._overrides[text])


LABEL_TOKEN_IDS = {"MAINTAINS": (10,), "HEDGES": (11,), "FLIPS": (12,)}


def _one_hot_logits(hot_id, vocab_size=200, magnitude=30.0):
    logits = np.full(vocab_size, -magnitude, dtype=np.float32)
    logits[hot_id] = magnitude
    return logits


class FakeJudge:
    """Always answers with the given label's one-hot logits, regardless of
    the prompt -- these tests are about score_row/score_file's bookkeeping,
    not about judge_sycophancy_distribution's math (covered separately in
    test_sycophancy_judge.py)."""

    def __init__(self, label="MAINTAINS", model_id="fake-judge"):
        self.model_id = model_id
        self.tokenizer = FakeTokenizer(
            {"MAINTAINS": [10], " MAINTAINS": [110], "HEDGES": [11], " HEDGES": [111], "FLIPS": [12], " FLIPS": [112]}
        )
        self._logits = _one_hot_logits(LABEL_TOKEN_IDS[label][0])
        self.calls = 0

    def next_token_logits(self, messages, enable_thinking=None):
        self.calls += 1
        return self._logits


def _row(trajectory_id="t1", turn=1, judge_model="fake-judge", agent_message="reply", stance_label="MAINTAINS"):
    return {
        "trajectory_id": trajectory_id,
        "turn": turn,
        "question": "q",
        "correction": "correct",
        "presupposition": "false",
        "agent_message": agent_message,
        "judge_model": judge_model,
        "stance_label": stance_label,
        "y_consistency": 1.0,
    }


def test_score_row_raises_on_judge_model_mismatch():
    row = _row(judge_model="some-other-model")
    with pytest.raises(ValueError):
        score_row(FakeJudge(), row, LABEL_TOKEN_IDS)


def test_score_row_does_not_overwrite_existing_fields():
    row = _row()
    scored = score_row(FakeJudge("MAINTAINS"), row, LABEL_TOKEN_IDS)
    assert scored["y_consistency"] == 1.0
    assert scored["stance_label"] == "MAINTAINS"
    assert scored["y_consistency_continuous"] == pytest.approx(1.0, abs=1e-6)
    assert scored["stance_label_argmax"] == "MAINTAINS"
    assert scored["continuous_readout_version"] == "v0.1"
    assert row == _row()  # source row not mutated


def test_score_row_new_fields_are_all_present():
    scored = score_row(FakeJudge("FLIPS"), _row(), LABEL_TOKEN_IDS)
    for key in ("p_maintains", "p_hedges", "p_flips", "label_mass_total", "stance_label_argmax", "continuous_readout_version"):
        assert key in scored


def test_score_file_skips_already_scored_rows_on_rerun(tmp_path):
    source_path = tmp_path / "trajectories.jsonl"
    dest_path = tmp_path / "continuous_readout" / "trajectories.jsonl"
    source_path.write_text(json.dumps(_row(turn=1)) + "\n" + json.dumps(_row(turn=2)) + "\n")

    judge = FakeJudge("MAINTAINS")
    score_file(judge, source_path, dest_path, LABEL_TOKEN_IDS)
    assert judge.calls == 2

    # Rerun with a fresh judge instance: nothing should be re-scored.
    judge2 = FakeJudge("MAINTAINS")
    rows = score_file(judge2, source_path, dest_path, LABEL_TOKEN_IDS)
    assert judge2.calls == 0
    assert len(rows) == 2


def test_score_file_rescores_a_row_whose_agent_message_changed(tmp_path):
    source_path = tmp_path / "trajectories.jsonl"
    dest_path = tmp_path / "continuous_readout" / "trajectories.jsonl"
    source_path.write_text(json.dumps(_row(turn=1, agent_message="original")) + "\n")

    score_file(FakeJudge("MAINTAINS"), source_path, dest_path, LABEL_TOKEN_IDS)

    source_path.write_text(json.dumps(_row(turn=1, agent_message="rewritten")) + "\n")
    judge2 = FakeJudge("FLIPS")
    rows = score_file(judge2, source_path, dest_path, LABEL_TOKEN_IDS)
    assert judge2.calls == 1
    assert rows[0]["agent_message"] == "rewritten"
    assert rows[0]["stance_label_argmax"] == "FLIPS"


def test_score_dirs_groups_by_judge_model_and_writes_per_dir_manifest(tmp_path):
    dir_a = tmp_path / "self_judge"
    dir_b = tmp_path / "independent_judge"
    for d in (dir_a, dir_b):
        d.mkdir()
    (dir_a / "trajectories.jsonl").write_text(json.dumps(_row(judge_model="judge-a")) + "\n")
    (dir_b / "trajectories.jsonl").write_text(json.dumps(_row(judge_model="judge-b", stance_label="FLIPS")) + "\n")

    def fake_chat_model_cls(model_id, device, enable_thinking):
        label = "MAINTAINS" if model_id == "judge-a" else "FLIPS"
        return FakeJudge(label=label, model_id=model_id)

    manifest = score_dirs([dir_a, dir_b], chat_model_cls=fake_chat_model_cls)
    assert manifest[str(dir_a)]["judge_model"] == "judge-a"
    assert manifest[str(dir_a)]["n_argmax_matches_stance_label"] == 1
    assert manifest[str(dir_b)]["judge_model"] == "judge-b"
    assert manifest[str(dir_b)]["n_argmax_matches_stance_label"] == 1
    assert (dir_a / "continuous_readout" / "trajectories.jsonl").exists()
    assert (dir_b / "continuous_readout" / "trajectories.jsonl").exists()


def test_score_dirs_skips_missing_directories(tmp_path):
    dir_a = tmp_path / "present"
    dir_a.mkdir()
    (dir_a / "trajectories.jsonl").write_text(json.dumps(_row(judge_model="judge-a")) + "\n")
    missing = tmp_path / "absent"

    manifest = score_dirs(
        [dir_a, missing], chat_model_cls=lambda model_id, device, enable_thinking: FakeJudge(model_id=model_id)
    )
    assert str(missing) not in manifest
    assert str(dir_a) in manifest


def test_resolve_label_token_ids_works_on_the_fake_tokenizer_used_here():
    resolved = resolve_label_token_ids(FakeJudge().tokenizer)
    assert set(resolved) == {"MAINTAINS", "HEDGES", "FLIPS"}
