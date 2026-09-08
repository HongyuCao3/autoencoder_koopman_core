"""docs/experiments/two_task_success_plan.md section 2 E1 item 4 (B1): at
least three checks on scripts/analyze_ergo_append_comparison.py -- the
paired-difference gate matches a hand computation on synthetic two-arm data,
`--identity` gives 1.00/0.00 on synthetic all-same/all-different data, and a
mismatched item_id set between the two arms raises instead of silently
comparing the intersection.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

from analyze_ergo_append_comparison import (  # noqa: E402
    compute_gate,
    compute_identity,
    main,
)


def _row(item_id: str, trajectory_id: str, turn: int, num_shards: int, y: float, raw_output: str, seed: int = 0) -> dict:
    return {
        "item_id": item_id,
        "trajectory_id": trajectory_id,
        "turn": turn,
        "num_shards": num_shards,
        "y_task_success": y,
        "judge_raw_output": raw_output,
        "agent_message": raw_output,
        "seed": seed,
    }


def _single_turn_rows(items_to_y: dict[str, float], raw_output_fn=lambda item, y: f"answer-{y}") -> list[dict]:
    """One-turn, single-seed trajectories: turn == num_shards == 1 for every
    item, so `_per_item_final_turn_success` just returns `items_to_y` itself
    and `_final_turn_field_by_trajectory` is trivial to hand-verify."""

    rows = []
    for item_id, y in items_to_y.items():
        rows.append(_row(item_id, f"{item_id}__seed0", turn=1, num_shards=1, y=y, raw_output=raw_output_fn(item_id, y)))
    return rows


def test_gate_paired_difference_matches_hand_computation():
    # Every item: arm a scores 1.0, arm b scores 0.0 -- diffs are [1, 1, 1]
    # for every item, so the paired mean diff is exactly 1.0 and every
    # bootstrap resample (drawn only from {1.0}) is also exactly 1.0, giving
    # a degenerate, hand-computable CI of [1.0, 1.0].
    rows_a = _single_turn_rows({"i1": 1.0, "i2": 1.0, "i3": 1.0})
    rows_b = _single_turn_rows({"i1": 0.0, "i2": 0.0, "i3": 0.0})
    gate = compute_gate(rows_a, "arm_a", rows_b, "arm_b")
    assert gate["n_pairs"] == 3
    assert gate["mean_diff"] == pytest.approx(1.0)
    assert gate["ci_low"] == pytest.approx(1.0)
    assert gate["ci_high"] == pytest.approx(1.0)
    assert gate["excludes_zero_and_positive"] is True
    assert gate["trajectory_id_intersection"] == 3


def test_gate_matches_manual_numpy_recomputation_on_mixed_values():
    items_a = {"i1": 1.0, "i2": 0.0, "i3": 1.0, "i4": 0.5}
    items_b = {"i1": 0.0, "i2": 0.0, "i3": 0.0, "i4": 0.5}
    rows_a = _single_turn_rows(items_a)
    rows_b = _single_turn_rows(items_b)
    gate = compute_gate(rows_a, "arm_a", rows_b, "arm_b")

    common = sorted(set(items_a) & set(items_b))
    diffs = np.array([items_a[i] - items_b[i] for i in common])
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(diffs), size=(10000, len(diffs)))
    means = diffs[idx].mean(axis=1)
    assert gate["mean_diff"] == pytest.approx(float(diffs.mean()))
    assert gate["ci_low"] == pytest.approx(float(np.percentile(means, 2.5)))
    assert gate["ci_high"] == pytest.approx(float(np.percentile(means, 97.5)))


def test_identity_is_one_when_all_final_turn_raw_outputs_match():
    rows_a = _single_turn_rows({"i1": 1.0, "i2": 0.0}, raw_output_fn=lambda item, y: "SAME")
    rows_b = _single_turn_rows({"i1": 1.0, "i2": 0.0}, raw_output_fn=lambda item, y: "SAME")
    result = compute_identity(rows_a, "arm_a", rows_b, "arm_b")
    assert result["rate"] == pytest.approx(1.0)
    assert result["n_identical"] == 2
    assert result["n_trajectories"] == 2


def test_identity_is_zero_when_all_final_turn_raw_outputs_differ():
    rows_a = _single_turn_rows({"i1": 1.0, "i2": 0.0}, raw_output_fn=lambda item, y: "A")
    rows_b = _single_turn_rows({"i1": 1.0, "i2": 0.0}, raw_output_fn=lambda item, y: "B")
    result = compute_identity(rows_a, "arm_a", rows_b, "arm_b")
    assert result["rate"] == pytest.approx(0.0)
    assert result["n_identical"] == 0
    assert result["n_trajectories"] == 2


def test_mismatched_item_id_sets_raises_for_gate():
    rows_a = _single_turn_rows({"i1": 1.0, "i2": 0.0})
    rows_b = _single_turn_rows({"i1": 1.0, "i3": 0.0})
    with pytest.raises(ValueError, match="item_id sets differ"):
        compute_gate(rows_a, "arm_a", rows_b, "arm_b")


def test_mismatched_item_id_sets_raises_for_identity():
    rows_a = _single_turn_rows({"i1": 1.0, "i2": 0.0})
    rows_b = _single_turn_rows({"i1": 1.0, "i3": 0.0})
    with pytest.raises(ValueError, match="item_id sets differ"):
        compute_identity(rows_a, "arm_a", rows_b, "arm_b")


def test_cli_end_to_end_writes_expected_gate_and_identity(tmp_path, capsys):
    # End-to-end smoke test through main(): two arm directories on disk, one
    # --gate and one --identity declaration, output written once and refused
    # on a second call (no-overwrite convention).
    for name, y_by_item, raw in (
        ("armA", {"i1": 1.0, "i2": 1.0}, "SAME"),
        ("armB", {"i1": 0.0, "i2": 0.0}, "SAME"),
    ):
        arm_dir = tmp_path / name
        arm_dir.mkdir()
        rows = _single_turn_rows(y_by_item, raw_output_fn=lambda item, y, raw=raw: raw)
        with open(arm_dir / "trajectories.jsonl", "w") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")

    out_path = tmp_path / "report.json"
    argv = [
        "analyze_ergo_append_comparison.py",
        "--arm",
        f"armA={tmp_path / 'armA'}",
        "--arm",
        f"armB={tmp_path / 'armB'}",
        "--gate",
        "armA-armB",
        "--identity",
        "armA,armB",
        "--output",
        str(out_path),
    ]
    old_argv = sys.argv
    sys.argv = argv
    try:
        main()
    finally:
        sys.argv = old_argv
    capsys.readouterr()

    report = json.loads(out_path.read_text())
    assert report["gates"]["armA-armB"]["mean_diff"] == pytest.approx(1.0)
    assert report["identities"]["armA,armB"]["rate"] == pytest.approx(1.0)

    sys.argv = argv
    try:
        with pytest.raises(SystemExit):
            main()
    finally:
        sys.argv = old_argv
