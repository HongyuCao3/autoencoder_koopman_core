"""Tests for scripts/calibrate_sequor_judge_vllm.py's CPU half.

`load_shared_module` exists because of a failure a dry run caught on
2026-09-09, before any GPU was requested: the script imported
`persona_drift.sequor_calibration_metrics`, which runs
`persona_drift/__init__.py`, which imports `.analysis` -> pandas. The
`constraint` line's vLLM environment (plan section 8) has vLLM and nothing
else, deliberately, so the runner died at import.

The fix must keep the property the two-runner design is FOR: both backends
read the same prompt, the same verdict parser and the same metric code, so a
disagreement between their reports can only come from the model or the
sampler. Loading a second copy of those functions would defeat the
`--compare-to` check while leaving it looking like it works.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import pytest

from persona_drift import sequor_calibration_metrics, sequor_constraint_judge

_PATH = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "calibrate_sequor_judge_vllm.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("calibrate_sequor_judge_vllm", _PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mod = _load_runner()

GOLD = pathlib.Path(__file__).resolve().parents[1] / "resources" / "sequor" / "sequor_gold_judge_calibration.jsonl"


def test_prompt_template_is_the_same_bytes_as_the_package_copy():
    assert mod.CONSTRAINT_JUDGE_PROMPT_TEMPLATE == sequor_constraint_judge.CONSTRAINT_JUDGE_PROMPT_TEMPLATE


def test_shared_modules_are_loaded_from_the_package_source_files():
    for name, package_module in (
        ("sequor_calibration_metrics", sequor_calibration_metrics),
        ("sequor_constraint_judge", sequor_constraint_judge),
    ):
        assert mod.load_shared_module(name).__file__ == package_module.__file__


def test_verdict_parser_agrees_with_the_package_copy():
    for text in (
        "reasoning\nFinal Verdict: [[Yes]]",
        "Final Verdict: [[No]]",
        "Final Verdict: Yes",
        "the judge rambled and never voted",
        "Final Verdict: [[Yes]]\nFinal Verdict: [[No]]",
    ):
        assert mod.extract_verdict(text) == sequor_constraint_judge.extract_verdict(text)


def test_it_imports_without_the_analysis_stack(monkeypatch):
    """The 2026-09-09 regression, as a test: with pandas unavailable the
    runner must still load. `sys.modules[name] = None` makes `import name`
    raise, which is what the vLLM env does for real."""

    for blocked in ("pandas", "torch", "transformers"):
        monkeypatch.setitem(sys.modules, blocked, None)
    for cached in [k for k in sys.modules if k == "persona_drift" or k.startswith("persona_drift.")]:
        monkeypatch.delitem(sys.modules, cached, raising=False)

    runner = _load_runner()
    assert runner.CONSTRAINT_JUDGE_PROMPT_TEMPLATE.startswith("An assistant has been asked")
    assert runner.provenance(switches={"backend": "vllm"})["git_sha"]


def test_stratified_slice_reproduces_the_smoke_run_selection():
    """The backend-consistency check compares verdicts row by row, which only
    means anything if `--limit 24` on vLLM picks the same 24 gold rows the HF
    smoke run (job 15719117) scored."""

    smoke = pathlib.Path(__file__).resolve().parents[1] / "outputs" / "sequor_s0_smoke" / \
        "judge_calibration_qwen3_14b_smoke.json"
    if not smoke.exists():
        pytest.skip("smoke report not present")
    rows = [json.loads(l) for l in GOLD.open() if l.strip()]
    selected = [r["gold_id"] for r in mod.stratified_slice(rows, 24)]
    assert selected == [r["gold_id"] for r in json.loads(smoke.read_text())["rows"]]


def test_out_path_guard_refuses_before_the_model_loads(tmp_path, monkeypatch):
    """`--out-path` pointing at an existing artifact must abort. This runs in
    an env with no vLLM, so reaching the import would raise ModuleNotFoundError
    instead of SystemExit -- which is exactly the ordering being asserted."""

    existing = tmp_path / "report.json"
    existing.write_text("{}")
    monkeypatch.setattr(
        sys, "argv",
        ["calibrate_sequor_judge_vllm.py", "--judge-model", "Qwen/Qwen3-14B", "--out-path", str(existing)],
    )
    with pytest.raises(SystemExit):
        mod.main()
