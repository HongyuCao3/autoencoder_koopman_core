"""The line registry of `eval_surrogate_rows_behavioral.py`.

Two things are nailed down here, both of which would be invisible if they
broke. (1) Adding the second-backbone entries must not change what the script
does when invoked the way the published three columns were produced -- so
`--lines` with no argument still resolves to exactly those three, in order.
(2) Every gemma entry must read a DIFFERENT artefact from its Qwen twin while
keeping the readout columns identical: a gemma spec that silently pointed at
the Qwen arm would report the Qwen column twice and the two would agree
perfectly, which is the failure mode that looks like a result.
"""
import importlib.util
import pathlib
import sys

import pytest

SCRIPTS = pathlib.Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS / ".." / "src"))
_spec = importlib.util.spec_from_file_location(
    "eval_surrogate_rows_behavioral", SCRIPTS / "eval_surrogate_rows_behavioral.py")
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)

PAIRS = [("constraint", "constraint_gemma4"), ("gsm8k_sharded", "gsm8k_sharded_gemma4")]


def test_default_lines_are_the_published_three_in_order():
    assert mod.DEFAULT_LINES == ("constraint", "gsm8k_sharded", "defense")
    parser_default = ["constraint", "gsm8k_sharded", "defense"]
    assert list(mod.DEFAULT_LINES) == parser_default
    assert set(mod.DEFAULT_LINES) <= set(mod.LINES)


def test_defense_has_no_second_backbone_twin():
    """User ruling 2026-09-17: its judge is the agent, so the backbone cannot move alone."""
    assert "defense_gemma4" not in mod.LINES


@pytest.mark.parametrize("qwen,gemma", PAIRS)
def test_gemma_twin_reads_a_different_artefact(qwen, gemma):
    key = "readout" if "readout" in mod.LINES[qwen] else "trajectories"
    assert key in mod.LINES[gemma]
    assert mod.LINES[qwen][key] != mod.LINES[gemma][key]
    assert "gemma4" in mod.LINES[gemma][key]


@pytest.mark.parametrize("qwen,gemma", PAIRS)
def test_gemma_twin_keeps_the_readout_columns_identical(qwen, gemma):
    for field in ("item_col", "y_col", "u_col", "branches"):
        assert mod.LINES[qwen][field] == mod.LINES[gemma][field], field


@pytest.mark.parametrize("qwen,gemma", PAIRS)
def test_gemma_twin_judge_is_not_the_gemma_agent(qwen, gemma):
    """global.md: a self-judged score is never reportable. The judge stays put."""
    assert "gemma" not in mod.LINES[gemma]["judge"].lower()
