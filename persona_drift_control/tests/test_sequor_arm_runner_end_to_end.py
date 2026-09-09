"""End-to-end test of scripts/run_sequor_s0_0_branch_arm.py against a stub vLLM.

WHY THIS EXISTS: job 15761810 generated 1404 rows over 52 GPU-minutes and lost
every one of them to `NameError: name 'turn_clock' is not defined`, raised
while building the summary AFTER all generation was done. The dry run that
preceded it only exercised `parse_args`, so the whole write path -- the part
that actually failed -- was never executed off-GPU.

A fake `vllm` module makes the real `main()` runnable in pytest: same argument
parsing, same seed loop, same prefix assertion, same artifacts. Any name that
does not resolve, any key that does not exist, and any ordering mistake in the
write path fails here in a second instead of after an hour on an A100.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
import types

import pytest

_PATH = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "run_sequor_s0_0_branch_arm.py"
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
    """Deterministic, and long enough on one item to exercise the cap path."""

    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs

    def get_tokenizer(self):
        return _Tokenizer()

    def chat(self, conversations, params, **_kwargs):
        out = []
        for conversation in conversations:
            # tuple_101_1's own constraint text: the item id never appears in a
            # prompt, so the stub keys on what the model would actually see.
            long = "Output the markdown in a code block" in json.dumps(conversation)
            n = params.max_tokens if long else 11
            out.append(_Completion(_Out(f"stub reply ({len(conversation)} deep)", n,
                                        "length" if long else "stop")))
        return out


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
    spec = importlib.util.spec_from_file_location("run_sequor_s0_0_branch_arm", _PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(runner, monkeypatch, out_dir: pathlib.Path, *extra: str) -> None:
    monkeypatch.setattr(sys, "argv", [
        "run_sequor_s0_0_branch_arm.py",
        "--agent-model", "stub/model",
        "--out-dir", str(out_dir),
        "--tuples-path", str(TUPLES), "--gold-path", str(GOLD),
        "--n-items", "3", "--n-turns", "4", "--max-new-tokens", "64",
        "--mode", "debug", *extra,
    ])
    runner.main()


def test_a_single_seed_arm_writes_all_three_artifacts(runner, monkeypatch, tmp_path):
    out = tmp_path / "arm"
    _run(runner, monkeypatch, out)
    rows = [json.loads(l) for l in (out / "trajectories.jsonl").open()]
    assert len(rows) == 3 * 4 + 3 * 3          # base rows + counterfactual rows
    report = json.loads((out / "arm_report.json").read_text())
    assert report["n_pairs"] == 3 * 3
    assert report["batches"] > 0               # the field whose name broke job 15761810
    assert report["mode"] == "debug"
    config = json.loads((out / "run_config.json").read_text())
    assert config["seeds"] == [0] and config["decoding_mode"] == "greedy"
    assert config["provenance"]["git_sha"]


def test_three_seeds_triple_the_rows_and_keep_pairs_within_a_seed(runner, monkeypatch, tmp_path):
    out = tmp_path / "arm3"
    _run(runner, monkeypatch, out, "--seeds", "0", "1", "2")
    rows = [json.loads(l) for l in (out / "trajectories.jsonl").open()]
    assert len(rows) == 3 * (3 * 4 + 3 * 3)
    assert {r["seed"] for r in rows} == {0, 1, 2}
    report = json.loads((out / "arm_report.json").read_text())
    assert report["n_pairs"] == 3 * 3 * 3
    assert report["seeds"] == [0, 1, 2]


def test_the_system_placement_switch_reaches_the_rows(runner, monkeypatch, tmp_path):
    out = tmp_path / "arm_sys"
    _run(runner, monkeypatch, out, "--constraints-in-system")
    rows = [json.loads(l) for l in (out / "trajectories.jsonl").open()]
    assert all(r["constraints_in_system"] for r in rows)
    assert json.loads((out / "arm_report.json").read_text())["constraints_in_system"] is True


def test_per_item_cap_accounting_flags_only_the_offending_item(runner, monkeypatch, tmp_path):
    """The stub truncates one item's responses and no others -- the shape of
    job 15756689, where a global average of 18.6% hid two items at 92% and 100%."""

    out = tmp_path / "arm_cap"
    _run(runner, monkeypatch, out)
    report = json.loads((out / "arm_report.json").read_text())
    assert report["items_over_cap_criterion"] == ["tuple_101_1"]
    assert report["cap_criterion_pass"] is False
    assert report["token_cap_by_item"]["tuple_101_1"]["token_cap_share"] == 1.0
    assert all(v["token_cap_share"] == 0.0 for k, v in report["token_cap_by_item"].items()
               if k != "tuple_101_1")


def test_rows_survive_a_failure_in_the_summary(runner, monkeypatch, tmp_path):
    """The actual loss mode of job 15761810: generation finished, the summary
    raised, and nothing had been written. Trajectories must already be on disk
    by then."""

    out = tmp_path / "arm_crash"
    monkeypatch.setattr(runner, "assert_pairs_share_prefix",
                        lambda rows: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError, match="boom"):
        _run(runner, monkeypatch, out)
    rows = [json.loads(l) for l in (out / "trajectories.jsonl").open()]
    assert len(rows) == 3 * 4 + 3 * 3
    assert not (out / "arm_report.json").exists()


def test_an_existing_out_dir_is_refused_before_the_model_loads(runner, monkeypatch, tmp_path):
    out = tmp_path / "arm_exists"
    out.mkdir()
    with pytest.raises(SystemExit, match="refusing to write into existing"):
        _run(runner, monkeypatch, out)
