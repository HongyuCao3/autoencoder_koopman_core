"""End-to-end test of scripts/run_sequor_s1_pilot_arm.py against a stub vLLM.

Job 15761810 lost 52 GPU-minutes to a NameError in the write path that a
`parse_args` dry run never reached, so the real `main()` runs here.

Beyond that, the pilot has two properties that no dry run and no eyeball can
confirm and that everything downstream depends on: the four arms must actually
differ in the actions they take, and the two excitation arms must leave
exactly one reminded side in every (item, turn, seed) cell. Both are asserted
on the ROWS, because a correct schedule does not prove the runner used it.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
import types

import pytest

_PATH = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "run_sequor_s1_pilot_arm.py"
TUPLES = pathlib.Path(__file__).resolve().parents[1] / "resources" / "sequor" / "sequor_tuples3.jsonl"
GOLD = pathlib.Path(__file__).resolve().parents[1] / "resources" / "sequor" / \
    "sequor_gold_judge_calibration.jsonl"


class _Out:
    def __init__(self, text: str, n_tokens: int, finish_reason: str):
        self.text = text
        self.token_ids = list(range(n_tokens))
        self.finish_reason = finish_reason


class _Completion:
    def __init__(self, out: _Out):
        self.outputs = [out]


class _Tokenizer:
    def encode(self, text: str) -> list[int]:
        return list(range(len(text.split())))


class _StubLLM:
    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs

    def get_tokenizer(self):
        return _Tokenizer()

    def chat(self, conversations, params, **_kwargs):
        return [_Completion(_Out(f"stub reply ({len(c)} deep)", 11, "stop"))
                for c in conversations]


class _SamplingParams:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.max_tokens = kwargs["max_tokens"]


@pytest.fixture
def runner(monkeypatch):
    stub = types.ModuleType("vllm")
    stub.LLM = _StubLLM
    stub.SamplingParams = _SamplingParams
    monkeypatch.setitem(sys.modules, "vllm", stub)
    spec = importlib.util.spec_from_file_location("run_sequor_s1_pilot_arm", _PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(runner, monkeypatch, out_dir: pathlib.Path, *extra: str) -> None:
    monkeypatch.setattr(sys, "argv", [
        "run_sequor_s1_pilot_arm.py",
        "--agent-model", "stub/model",
        "--out-dir", str(out_dir),
        "--tuples-path", str(TUPLES), "--gold-path", str(GOLD),
        "--n-items", "3", "--n-turns", "5", "--seeds", "0", "1",
        "--max-new-tokens", "64", "--mode", "debug", *extra,
    ])
    runner.main()


def _rows(out: pathlib.Path) -> list[dict]:
    return [json.loads(line) for line in (out / "trajectories.jsonl").open()]


def test_all_four_arms_run_for_every_seed(runner, monkeypatch, tmp_path):
    out = tmp_path / "pilot"
    _run(runner, monkeypatch, out)
    rows = _rows(out)
    assert len(rows) == 4 * 3 * 5 * 2
    assert {r["branch"] for r in rows} == {
        "zero_control", "constant_remind", "bernoulli", "antithetic"}
    assert {r["seed"] for r in rows} == {0, 1}
    report = json.loads((out / "arm_report.json").read_text())
    assert report["batches"] > 0          # the field whose name broke job 15761810
    assert report["n_rows"] == len(rows)


def test_the_arms_actually_differ_in_what_they_do(runner, monkeypatch, tmp_path):
    """Four labels over one behaviour would pass every structural check and
    produce a `B` of zero for reasons that have nothing to do with the model."""

    out = tmp_path / "pilot_diff"
    _run(runner, monkeypatch, out)
    per_arm = json.loads((out / "arm_report.json").read_text())["per_arm"]
    assert per_arm["zero_control"]["u_mean_from_turn2"] == 0.0
    assert per_arm["constant_remind"]["u_mean_from_turn2"] == 1.0
    assert 0.0 < per_arm["bernoulli"]["u_mean_from_turn2"] < 1.0
    assert per_arm["zero_control"]["inserted_tokens_total"] == 0
    assert per_arm["constant_remind"]["inserted_tokens_total"] > 0


def test_every_excitation_cell_has_exactly_one_reminded_side(runner, monkeypatch, tmp_path):
    out = tmp_path / "pilot_cells"
    _run(runner, monkeypatch, out)
    cells = json.loads((out / "arm_report.json").read_text())["cell_coverage"]
    assert cells["n_cells"] == 3 * 4 * 2          # items x turns>=2 x seeds
    assert cells["n_cells_with_exactly_one_reminder"] == cells["n_cells"]
    assert cells["cells_are_matched"] is True


def test_a_seed_changes_the_excitation_schedule(runner, monkeypatch, tmp_path):
    """Three seeds that draw the same coin are one arm reported as three, and
    the MDE computed from them would be too small."""

    out = tmp_path / "pilot_seeds"
    _run(runner, monkeypatch, out)
    rows = [r for r in _rows(out) if r["branch"] == "bernoulli"]
    by_seed = {}
    for row in rows:
        by_seed.setdefault(row["seed"], {})[(row["item_id"], row["turn"])] = row["u_remind"]
    assert by_seed[0] != by_seed[1]


def test_the_schedules_and_their_checks_are_in_the_artifact(runner, monkeypatch, tmp_path):
    """The actions taken are part of the run's provenance: an arm whose
    schedule cannot be reconstructed cannot be refit or re-analysed."""

    out = tmp_path / "pilot_prov"
    _run(runner, monkeypatch, out)
    config = json.loads((out / "run_config.json").read_text())
    assert set(config["schedules"]) == {"0", "1"}
    assert set(config["schedules"]["0"]) == {
        "zero_control", "constant_remind", "bernoulli", "antithetic"}
    assert config["schedule_checks"]["0"]["turn1_action_free"] is True
    assert config["provenance"]["git_sha"]
    assert 0.0 < config["gold_coverage"]["share_outside_calibration_set"] < 1.0 or \
        config["gold_coverage"]["n_items_fully_covered"] == 3


def test_the_report_says_it_is_not_a_result(runner, monkeypatch, tmp_path):
    out = tmp_path / "pilot_ne"
    _run(runner, monkeypatch, out)
    report = json.loads((out / "arm_report.json").read_text())
    assert "no K gate reads it" in report["not_a_result"]
    assert report["mode"] == "debug"


def test_it_refuses_to_write_into_an_existing_directory(runner, monkeypatch, tmp_path):
    out = tmp_path / "pilot_twice"
    out.mkdir()
    with pytest.raises(SystemExit, match="refusing to write into existing"):
        _run(runner, monkeypatch, out)


def test_the_default_item_selection_is_still_the_screening_set(runner, monkeypatch, tmp_path):
    """`--item-selection` was added for S1. The default must leave the pilot's
    item set untouched: the sizing report and the S0-0 gates were computed on
    the fully gold-covered dialogues, and a flag that silently widened them
    would change what those artifacts mean."""

    out = tmp_path / "default"
    _run(runner, monkeypatch, out)
    report = json.loads((out / "arm_report.json").read_text())
    from persona_drift.sequor_bank import gold_constraints, load_sequor_bank, select_screening_items
    expected = select_screening_items(load_sequor_bank(TUPLES, n_turns=5), gold_constraints(GOLD), 3)
    assert sorted(report["gold_coverage"]["gold_coverage_by_item"]) == \
        sorted(it.conversation_id for it in expected)
    assert report["gold_coverage"]["share_outside_calibration_set"] == 0.0


def test_bank_selection_admits_uncovered_dialogues_and_prices_them(runner, monkeypatch, tmp_path):
    out = tmp_path / "bank"
    _run(runner, monkeypatch, out, "--item-selection", "bank")
    report = json.loads((out / "arm_report.json").read_text())
    from persona_drift.sequor_bank import gold_constraints, load_sequor_bank, select_bank_items
    expected = select_bank_items(load_sequor_bank(TUPLES, n_turns=5), gold_constraints(GOLD), 3)
    assert sorted(report["gold_coverage"]["gold_coverage_by_item"]) == \
        sorted(it.conversation_id for it in expected)
    # the whole point of the ruling: the cost is a number in the artifact
    assert report["gold_coverage"]["share_outside_calibration_set"] > 0.0


def test_screening_selection_refuses_more_items_than_are_fully_covered(runner, monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", [
        "run_sequor_s1_pilot_arm.py", "--agent-model", "stub/model",
        "--out-dir", str(tmp_path / "too-many"),
        "--tuples-path", str(TUPLES), "--gold-path", str(GOLD),
        "--n-items", "40", "--n-turns", "3", "--seeds", "0",
        "--max-new-tokens", "64", "--mode", "debug",
    ])
    with pytest.raises(ValueError, match="fully gold-covered"):
        runner.main()
