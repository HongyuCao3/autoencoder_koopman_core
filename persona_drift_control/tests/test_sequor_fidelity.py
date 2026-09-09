"""Tests for scripts/analyze_sequor_fidelity.py.

The criterion has two halves and both must bite: a variant that lowers the
curve but not to upstream's level does NOT close the gap (it would send us
re-running the S0-0 arm into the same ceiling), and neither does one that
lands at the right level by moving a couple of items while the item-paired CI
still spans zero.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib

import pytest

_PATH = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "analyze_sequor_fidelity.py"
_SPEC = importlib.util.spec_from_file_location("analyze_sequor_fidelity", _PATH)
mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mod)

N_ITEMS, N_TURNS = 12, 20


@pytest.fixture(autouse=True)
def fast_bootstrap(monkeypatch):
    monkeypatch.setattr(mod, "N_BOOTSTRAP", 400)


def _rows(variant: str, level: float, turn1: float | None = None) -> list[dict]:
    """Every item holds its constraints with probability `level`, deterministically
    spread so item means differ but the variant mean is `level`."""

    rows = []
    for i in range(N_ITEMS):
        item_level = min(1.0, max(0.0, level + 0.1 * ((i % 3) - 1)))
        for t in range(1, N_TURNS + 1):
            p = turn1 if (t == 1 and turn1 is not None) else item_level
            held = ((i * 37 + t * 61) % 100) / 100.0 < p  # spread, not correlated with t
            rows.append({
                "variant": variant, "item_id": f"i{i}", "turn": t, "branch": "base",
                "y_binary": held, "y_graded": 1.0 if held else 2 / 3,
            })
    return rows


def _readout(*row_groups: list[dict]) -> dict:
    return {
        "mode": "canonical", "judge_kind": "independent", "judge_model": "Qwen/Qwen3-14B",
        "rows": [r for group in row_groups for r in group],
    }


def test_a_variant_that_lowers_the_curve_to_upstreams_level_closes_the_gap(tmp_path):
    readout = _readout(_rows("turn1_greedy", 0.85, turn1=0.85), _rows("system_default", 0.45, turn1=0.5))
    path = tmp_path / "r.json"
    path.write_text(json.dumps(readout))
    import sys
    sys.argv = ["x", "--readout", str(path), "--out-path", str(tmp_path / "out.json")]
    mod.main()
    out = json.loads((tmp_path / "out.json").read_text())
    assert out["verdict"] == "GAP CLOSED"
    assert out["variants_closing_the_gap"] == ["system_default"]
    d = out["variants"]["system_default"]["paired_vs_baseline"]
    assert d["mean_difference"] < -mod.GAP_MARGIN and d["ci"][1] < 0


def test_a_variant_that_lowers_the_curve_but_stays_above_upstream_does_not_close_it(tmp_path):
    """0.70 at turn 1 is a real drop from 0.85 and still nowhere near ~0.50:
    re-running the branch arm there would meet the same ceiling."""

    readout = _readout(_rows("turn1_greedy", 0.85, turn1=0.85), _rows("system_greedy", 0.70, turn1=0.70))
    path = tmp_path / "r.json"
    path.write_text(json.dumps(readout))
    import sys
    sys.argv = ["x", "--readout", str(path), "--out-path", str(tmp_path / "out.json")]
    mod.main()
    out = json.loads((tmp_path / "out.json").read_text())
    assert out["verdict"] == "GAP NOT CLOSED"
    assert out["variants"]["system_greedy"]["closes_gap"] is False


def test_no_variant_moving_means_the_gap_is_not_ours_to_close(tmp_path):
    readout = _readout(_rows("turn1_greedy", 0.85, turn1=0.85), _rows("turn1_default", 0.84, turn1=0.85))
    path = tmp_path / "r.json"
    path.write_text(json.dumps(readout))
    import sys
    sys.argv = ["x", "--readout", str(path), "--out-path", str(tmp_path / "out.json")]
    mod.main()
    assert json.loads((tmp_path / "out.json").read_text())["verdict"] == "GAP NOT CLOSED"


def test_pairing_is_by_item():
    variant = {"i0": 0.4, "i1": 0.5, "i2": 0.6}
    baseline = {"i0": 0.8, "i1": 0.8, "i2": 0.8}
    d = mod.paired_difference(variant, baseline, seed=0)
    assert d["n_items"] == 3
    assert d["mean_difference"] == pytest.approx(-0.3)
    assert d["ci"][1] < 0


def test_a_self_judged_readout_is_refused(tmp_path):
    readout = _readout(_rows("turn1_greedy", 0.8))
    readout["judge_kind"] = "self"
    path = tmp_path / "r.json"
    path.write_text(json.dumps(readout))
    import sys
    sys.argv = ["x", "--readout", str(path), "--out-path", str(tmp_path / "out.json")]
    with pytest.raises(SystemExit, match="reporting judge"):
        mod.main()


def test_a_readout_without_variants_is_refused(tmp_path):
    rows = _rows("turn1_greedy", 0.8)
    for r in rows:
        r["variant"] = None
    path = tmp_path / "r.json"
    path.write_text(json.dumps(_readout(rows)))
    import sys
    sys.argv = ["x", "--readout", str(path), "--out-path", str(tmp_path / "out.json")]
    with pytest.raises(SystemExit, match="not a fidelity arm"):
        mod.main()


def test_a_debug_arm_is_refused(tmp_path):
    readout = _readout(_rows("turn1_greedy", 0.8))
    readout["mode"] = "debug"
    path = tmp_path / "r.json"
    path.write_text(json.dumps(readout))
    import sys
    sys.argv = ["x", "--readout", str(path), "--out-path", str(tmp_path / "out.json")]
    with pytest.raises(SystemExit, match="not evidence"):
        mod.main()


def test_missing_baseline_is_refused(tmp_path):
    path = tmp_path / "r.json"
    path.write_text(json.dumps(_readout(_rows("system_default", 0.5))))
    import sys
    sys.argv = ["x", "--readout", str(path), "--out-path", str(tmp_path / "out.json")]
    with pytest.raises(SystemExit, match="baseline variant"):
        mod.main()
