"""Backend-independent metrics for the `constraint` line's S0 judge calibration.

Lives in the package rather than in scripts/calibrate_sequor_judge.py because
two runners need it -- the HF-transformers one and the vLLM one -- and a
calibration whose two backends compute their metrics from two copies of the
code cannot be used to check the two backends against each other.

No torch import: the vLLM environment (plan section 8) does not have this
project's dependencies installed.
"""

from __future__ import annotations

import statistics


def stratified_slice(rows: list[dict], limit: int) -> list[dict]:
    """N rows spread evenly over the four (source_model, label) cells, taking
    each cell's first rows so the selection is deterministic. A head slice
    would see one cell only, and a smoke run that never sees a `violate` row
    measures nothing about how the judge behaves on the half that matters."""

    cells: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        cells.setdefault((r["source_model"], r["label"]), []).append(r)
    per_cell = max(1, limit // len(cells))
    out = [r for cell in cells.values() for r in cell[:per_cell]]
    return out[:limit] if len(out) > limit else out


def cohens_kappa(pred: list[bool], gold: list[bool]) -> float:
    n = len(pred)
    po = sum(1 for p, g in zip(pred, gold) if p == g) / n
    p_yes = sum(pred) / n * sum(gold) / n
    p_no = (1 - sum(pred) / n) * (1 - sum(gold) / n)
    pe = p_yes + p_no
    return (po - pe) / (1 - pe) if pe != 1 else 0.0


def length_floors(rows: list[dict]) -> dict:
    """What a judge that reads only response length would score on these rows.
    Paired: within a (task, constraint) cell, call the shorter response the
    follower. Unpaired: shorter than the median is a follower."""

    words = {r["gold_id"]: len(r["response"].split()) for r in rows}
    by_cell: dict[tuple[str, str], dict[str, dict]] = {}
    for r in rows:
        by_cell.setdefault((r["task"], r["constraint"]), {})[r["label"]] = r
    paired = [(c["follow"], c["violate"]) for c in by_cell.values() if len(c) == 2]
    paired_correct = sum(1 for f, v in paired if words[f["gold_id"]] < words[v["gold_id"]])
    median = statistics.median(words.values())
    unpaired_correct = sum(1 for r in rows if (words[r["gold_id"]] < median) == (r["label"] == "follow"))

    def median_for(label: str) -> float | None:
        vals = [words[r["gold_id"]] for r in rows if r["label"] == label]
        return statistics.median(vals) if vals else None

    return {
        "paired_cells": len(paired),
        "paired_accuracy": paired_correct / len(paired) if paired else None,
        "unpaired_accuracy": unpaired_correct / len(rows),
        "median_words": median,
        "median_words_follow": median_for("follow"),
        "median_words_violate": median_for("violate"),
    }
