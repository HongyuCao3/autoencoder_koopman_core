"""Tests for src/persona_drift/tsar_cefr_bank.py.

The bank's whole job is to keep `source_id` trustworthy: the 5-fold split cuts
along it, and a paragraph whose two target-level rows landed in different folds
would put the reporting text inside the fitting fold. So the parser refuses any
row whose `text_id` does not agree with its `target_cefr`, and these tests
plant that disagreement.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from persona_drift import tsar_cefr_bank as bank


def write_rows(tmp_path: pathlib.Path, rows: list[dict]) -> pathlib.Path:
    path = tmp_path / "split.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def row(text_id: str, target: str, original: str = "A long source paragraph.",
        reference: str = "A short one.") -> dict:
    return {"dataset_id": "tsar2025trial", "text_id": text_id, "original": original,
            "target_cefr": target, "reference": reference}


def test_target_level_case_is_normalised(tmp_path):
    """trial writes `a2`, test writes `A2`; nothing downstream should case-fold again."""
    items = bank.load_tsar_bank(write_rows(tmp_path, [row("01-a2", "a2"), row("02-A2", "A2")]))
    assert [i.target_cefr for i in items] == ["A2", "A2"]


def test_source_id_pairs_the_two_target_levels(tmp_path):
    items = bank.load_tsar_bank(write_rows(tmp_path, [row("07-a2", "a2"), row("07-b1", "b1")]))
    assert [i.source_id for i in items] == ["07", "07"]
    assert bank.source_ids(items) == ["07"]


def test_text_id_disagreeing_with_target_is_rejected(tmp_path):
    """Planted defect: the id says b1, the field says a2. Silently trusting either
    one would mis-assign the row's fold."""
    path = write_rows(tmp_path, [row("07-b1", "a2")])
    with pytest.raises(ValueError, match="source id"):
        bank.load_tsar_bank(path)


def test_unknown_target_level_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="unexpected target_cefr"):
        bank.load_tsar_bank(write_rows(tmp_path, [row("07-a1", "a1")]))


def test_duplicate_text_id_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="duplicate"):
        bank.load_tsar_bank(write_rows(tmp_path, [row("07-a2", "a2"), row("07-a2", "a2")]))


def test_source_ids_keeps_first_appearance_order(tmp_path):
    rows = [row("03-a2", "a2"), row("01-a2", "a2"), row("03-b1", "b1")]
    assert bank.source_ids(bank.load_tsar_bank(write_rows(tmp_path, rows))) == ["03", "01"]


def test_pinned_revisions_match_the_documented_sizes():
    """resources/tsar_cefr/README.md and this table are the same claim; if they
    drift, a run reads data the provenance record does not describe."""
    assert bank.SPLITS["trial"]["n_rows"] == 40 and bank.SPLITS["trial"]["n_sources"] == 20
    assert bank.SPLITS["test"]["n_rows"] == 200 and bank.SPLITS["test"]["n_sources"] == 100
