"""End-to-end test of scripts/probe_sequor_response_length.py against a stub vLLM.

Same discipline as tests/test_sequor_arm_runner_end_to_end.py, and for the
same reason: job 15761810 lost 52 GPU-minutes to a NameError in the write
path, which a `parse_args` dry run never reaches. The real `main()` runs here
against a fake `vllm` module, so every name in the write path resolves off-GPU.

What this probe must get right, and therefore what is asserted: it selects
from the WHOLE bank rather than the 16 gold-covered dialogues, it records what
leaving that set costs, it runs zero-control (no counterfactual rows), and its
per-item cap accounting names the item that binds rather than reporting a
global average that hides it.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
import types

import pytest

_PATH = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "probe_sequor_response_length.py"
TUPLES = pathlib.Path(__file__).resolve().parents[1] / "resources" / "sequor" / "sequor_tuples3.jsonl"
GOLD = pathlib.Path(__file__).resolve().parents[1] / "resources" / "sequor" / \
    "sequor_gold_judge_calibration.jsonl"

LONG_ITEM_MARKER = "Output the markdown in a code block"


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
        out = []
        for conversation in conversations:
            long = LONG_ITEM_MARKER in json.dumps(conversation)
            n = params.max_tokens if long else 11
            out.append(_Completion(_Out(f"stub reply ({len(conversation)} deep)", n,
                                        "length" if long else "stop")))
        return out


class _SamplingParams:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.max_tokens = kwargs["max_tokens"]


@pytest.fixture
def probe(monkeypatch):
    stub = types.ModuleType("vllm")
    stub.LLM = _StubLLM
    stub.SamplingParams = _SamplingParams
    monkeypatch.setitem(sys.modules, "vllm", stub)
    spec = importlib.util.spec_from_file_location("probe_sequor_response_length", _PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(probe, monkeypatch, out_dir: pathlib.Path, *extra: str) -> None:
    monkeypatch.setattr(sys, "argv", [
        "probe_sequor_response_length.py",
        "--agent-model", "stub/model",
        "--out-dir", str(out_dir),
        "--tuples-path", str(TUPLES), "--gold-path", str(GOLD),
        "--n-items", "20", "--n-turns", "3", "--max-new-tokens", "64",
        "--decoding", "greedy", "--mode", "debug", *extra,
    ])
    probe.main()


def test_the_probe_writes_rows_and_a_report(probe, monkeypatch, tmp_path):
    out = tmp_path / "probe"
    _run(probe, monkeypatch, out)
    rows = [json.loads(line) for line in (out / "trajectories.jsonl").open()]
    assert len(rows) == 20 * 3
    report = json.loads((out / "probe_report.json").read_text())
    assert report["n_rows"] == 20 * 3 and report["n_items"] == 20
    assert report["mode"] == "debug"
    assert json.loads((out / "run_config.json").read_text())["provenance"]["git_sha"]


def test_it_is_zero_control_so_no_row_carries_an_action(probe, monkeypatch, tmp_path):
    """A reminded row here would be a counterfactual pair, which is the S0-0
    arm's job and costs twice the generations. The probe only needs the length
    of the trajectory the model actually walks."""

    out = tmp_path / "probe_zero"
    _run(probe, monkeypatch, out)
    rows = [json.loads(line) for line in (out / "trajectories.jsonl").open()]
    assert {r["branch"] for r in rows} == {"base"}
    assert all(r["u_remind"] == 0 for r in rows)
    assert all(r["inserted_tokens"] == 0 for r in rows)


def test_it_leaves_the_gold_covered_set_and_says_what_that_costs(probe, monkeypatch, tmp_path):
    """Only 16 dialogues are fully gold-covered, so an N=40 arm must draw on
    constraints the judge was never calibrated on (screening section 9.2). The
    probe must not silently fall back to the covered 16, and the share outside
    the calibration set has to be in the artifact, not in someone's memory."""

    out = tmp_path / "probe_cov"
    _run(probe, monkeypatch, out)
    report = json.loads((out / "probe_report.json").read_text())
    coverage = report["gold_coverage"]
    assert coverage["n_items"] == 20                       # more than the 16 covered ones exist
    assert coverage["n_items_fully_covered"] < 20
    assert 0.0 < coverage["share_outside_calibration_set"] < 1.0
    assert coverage["n_constraints_judged"] == 20 * 3
    assert "reported verbatim" in coverage["note"]


def test_the_binding_item_is_named_not_averaged_away(probe, monkeypatch, tmp_path):
    """The failure this whole probe exists to prevent: one item at 100% inside
    a global average that looks acceptable."""

    out = tmp_path / "probe_cap"
    _run(probe, monkeypatch, out)
    report = json.loads((out / "probe_report.json").read_text())
    over = report["items_over_cap_criterion"]
    assert over, "the stub truncates one item on purpose; it must be named"
    assert report["token_cap_share"] < 0.10, "and the global share must be the misleading one"
    for item in over:
        assert report["token_cap_by_item"][item]["token_cap_share"] > 0.05
        assert report["token_cap_by_item"][item]["constraints"]


def test_it_refuses_to_write_into_an_existing_directory(probe, monkeypatch, tmp_path):
    out = tmp_path / "probe_twice"
    out.mkdir()
    with pytest.raises(SystemExit, match="refusing to write into existing"):
        _run(probe, monkeypatch, out)


def test_the_report_says_it_is_not_evidence(probe, monkeypatch, tmp_path):
    """Infrastructure, by the classification this job was submitted under. The
    artifact has to carry that itself so a later reader cannot mistake a
    probe's numbers for a gate's."""

    out = tmp_path / "probe_ne"
    _run(probe, monkeypatch, out)
    report = json.loads((out / "probe_report.json").read_text())
    assert "No gate reads it" in report["not_evidence"]
