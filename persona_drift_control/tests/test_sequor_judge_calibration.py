"""Tests for scripts/calibrate_sequor_judge.py's CPU half.

stratified_slice exists because of a bug the dry run caught: the gold file is
ordered by (source_model, label) cell, so `rows[:24]` is 24 gpt-5.2 `follow`
rows and nothing else -- a smoke run that never sees a `violate` row measures
nothing about the half of the task that matters, and length_floors crashed on
the empty half rather than saying so.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib

import pytest

_PATH = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "calibrate_sequor_judge.py"
_SPEC = importlib.util.spec_from_file_location("calibrate_sequor_judge", _PATH)
mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mod)

GOLD = pathlib.Path(__file__).resolve().parents[1] / "resources" / "sequor" / "sequor_gold_judge_calibration.jsonl"


def _rows() -> list[dict]:
    return [json.loads(l) for l in GOLD.open() if l.strip()]


def test_gold_file_is_ordered_by_cell_so_a_head_slice_is_one_cell():
    """The premise of stratified_slice. If upstream ever ships a shuffled file
    this test fails loudly rather than the slice quietly becoming pointless."""
    head = _rows()[:24]
    assert len({(r["source_model"], r["label"]) for r in head}) == 1


def test_stratified_slice_covers_every_cell():
    sl = mod.stratified_slice(_rows(), 24)
    assert len(sl) == 24
    assert len({(r["source_model"], r["label"]) for r in sl}) == 4


def test_stratified_slice_is_deterministic():
    rows = _rows()
    assert [r["gold_id"] for r in mod.stratified_slice(rows, 16)] == \
           [r["gold_id"] for r in mod.stratified_slice(rows, 16)]


def test_stratified_slice_never_exceeds_the_limit():
    for limit in (4, 7, 24, 100):
        assert len(mod.stratified_slice(_rows(), limit)) <= limit


def test_length_floors_reproduce_the_hand_computed_values():
    """The two G-S0-0 floors, recomputed from the vendored file. If the
    vendored gold set is ever re-fetched and these move, the gate's meaning
    moved with them."""
    f = mod.length_floors(_rows())
    assert f["paired_cells"] == 500
    assert f["paired_accuracy"] == pytest.approx(0.648, abs=2e-3)
    assert f["unpaired_accuracy"] == pytest.approx(0.597, abs=2e-3)
    assert f["median_words_violate"] > f["median_words_follow"]


def test_length_floors_survive_a_single_label_slice():
    single = [r for r in _rows() if r["label"] == "follow"][:10]
    f = mod.length_floors(single)
    assert f["paired_cells"] == 0 and f["paired_accuracy"] is None
    assert f["median_words_violate"] is None


@pytest.mark.parametrize("pred,gold,expected", [
    ([True] * 50 + [False] * 50, [True] * 50 + [False] * 50, 1.0),
    ([True] * 50 + [False] * 50, [True, False] * 50, 0.0),
    ([True] * 100, [True] * 50 + [False] * 50, 0.0),
])
def test_cohens_kappa(pred, gold, expected):
    assert mod.cohens_kappa(pred, gold) == pytest.approx(expected, abs=1e-9)
