"""Tests for scripts/analyze_sequor_s1_pilot.py, on synthetic readouts.

The load-bearing claim of this script is the variance SPLIT: adding seeds
shrinks only the within-item term. So it is tested against two planted worlds
with the same total spread -- one where all of it is between items (seeds buy
nothing) and one where all of it is within an item (seeds buy everything). A
single pooled sd passes neither test, which is the point: ERGO paid for one
MDE standing in for another, and this is the same substitution one level down.
"""

from __future__ import annotations

import importlib.util
import json
import math
import pathlib

import pytest

_PATH = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "analyze_sequor_s1_pilot.py"
_SPEC = importlib.util.spec_from_file_location("analyze_sequor_s1_pilot", _PATH)
mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mod)

ARMS = ("zero_control", "constant_remind", "bernoulli", "antithetic")


@pytest.fixture(autouse=True)
def fast_bootstrap(monkeypatch):
    monkeypatch.setattr(mod, "N_BOOTSTRAP", 400)


def _rows(per_item_seed_gain, n_turns=20, late_from=15, base=0.5):
    """`per_item_seed_gain[item][seed]` becomes the exact late-window contrast."""

    rows = []
    for item, by_seed in per_item_seed_gain.items():
        for seed, gain in by_seed.items():
            for turn in range(1, n_turns + 1):
                for arm in ARMS:
                    y = base
                    if arm == "constant_remind" and turn >= late_from:
                        y = base + gain
                    rows.append({
                        "item_id": item, "seed": seed, "turn": turn, "branch": arm,
                        "y_graded": y, "u_remind": int(arm == "constant_remind"),
                    })
    return rows


def test_contrast_pairs_within_item_and_seed_and_respects_the_window():
    rows = _rows({"a": {0: 0.2}, "b": {0: 0.4}})
    contrast = mod.per_item_contrast(rows, "constant_remind", "zero_control", range(15, 21))
    assert contrast == {"a": {0: pytest.approx(0.2)}, "b": {0: pytest.approx(0.4)}}
    early = mod.per_item_contrast(rows, "constant_remind", "zero_control", range(1, 15))
    assert early == {"a": {0: pytest.approx(0.0)}, "b": {0: pytest.approx(0.0)}}


def test_a_seed_present_on_one_side_only_is_not_paired():
    rows = [r for r in _rows({"a": {0: 0.2, 1: 0.2}})
            if not (r["seed"] == 1 and r["branch"] == "zero_control")]
    contrast = mod.per_item_contrast(rows, "constant_remind", "zero_control", range(15, 21))
    assert list(contrast["a"]) == [0]


def test_between_item_world_seeds_buy_nothing():
    """Every seed of an item gives the same gain, items differ. All variance is
    between items, so MDE must not improve when seeds are added."""

    gains = {"i1": 0.0, "i2": 0.1, "i3": 0.2, "i4": 0.3}
    rows = _rows({item: {s: g for s in (0, 1, 2)} for item, g in gains.items()})
    contrast = mod.per_item_contrast(rows, "constant_remind", "zero_control", range(15, 21))
    components = mod.variance_components(contrast)
    assert components["sd_within_item_across_seeds"] == pytest.approx(0.0)
    assert components["sd_between_items"] == pytest.approx(components["sd_of_item_means"])
    table = mod.sizing_table(components, 0.10)["mde_by_n_and_seeds"]
    assert table["s1"]["n40"] == pytest.approx(table["s3"]["n40"])


def test_within_item_world_seeds_buy_the_whole_mde():
    """Items are identical on average, seeds differ. Then MDE at s=3 must be
    1/sqrt(3) of MDE at s=1 -- the arithmetic the sizing decision rests on."""

    per_item = {f"i{k}": {0: -0.2, 1: 0.0, 2: 0.2} for k in range(6)}
    contrast = mod.per_item_contrast(_rows(per_item), "constant_remind", "zero_control", range(15, 21))
    components = mod.variance_components(contrast)
    assert components["sd_between_items"] == pytest.approx(0.0)
    assert components["between_item_variance_is_negative"]
    table = mod.sizing_table(components, 0.10)["mde_by_n_and_seeds"]
    assert table["s3"]["n40"] == pytest.approx(table["s1"]["n40"] / math.sqrt(3), rel=1e-6)


def test_required_n_round_trips_through_the_mde_formula():
    per_item = {f"i{k}": {s: 0.1 * k + 0.05 * s for s in (0, 1, 2)} for k in range(8)}
    contrast = mod.per_item_contrast(_rows(per_item), "constant_remind", "zero_control", range(15, 21))
    sizing = mod.sizing_table(mod.variance_components(contrast), 0.10)
    n = sizing["required_n_items"]["s3"]
    var = (mod.variance_components(contrast)["sd_between_items"] ** 2
           + mod.variance_components(contrast)["sd_within_item_across_seeds"] ** 2 / 3)
    assert mod.POWER_Z * math.sqrt(var / n) <= 0.10
    assert mod.POWER_Z * math.sqrt(var / (n - 1)) > 0.10


def test_range_by_turn_reports_the_ceiling_and_flags_a_frozen_turn():
    rows = _rows({"a": {0: 0.5}, "b": {0: 0.5}}, base=0.5)
    for row in rows:
        if row["turn"] == 20 and row["branch"] == "constant_remind":
            row["y_graded"] = 1.0
    ranges = mod.range_by_turn(rows, list(ARMS))
    t20 = ranges["constant_remind"]["by_turn"]["20"]
    assert t20["ceiling_share"] == pytest.approx(1.0)
    assert t20["n_distinct"] == 1
    assert "20" in ranges["constant_remind"]["turns_failing_g_s1_spread"]


def _readout(tmp_path, rows, cap_by_item):
    arm_dir = tmp_path / "arm"
    arm_dir.mkdir()
    (arm_dir / "arm_report.json").write_text(json.dumps({
        "arms": list(ARMS), "n_turns": 20, "seeds": [0, 1, 2], "cap_criterion": 0.05,
        "token_cap_by_item": cap_by_item,
        "cell_coverage": {"cells_are_matched": True, "n_cells_with_exactly_one_reminder": 12},
        "schedule_checks": {"0": {"bernoulli_u_mean": 0.5, "antithetic_is_exact_complement": True,
                                  "turn1_action_free": True}},
        "gold_coverage": {"share_outside_calibration_set": 0.0},
    }))
    path = tmp_path / "readout.json"
    path.write_text(json.dumps({
        "judge_kind": "independent", "mode": "canonical", "arm_dir": str(arm_dir),
        "agent_model": "m", "judge_model": "j", "rows": rows,
    }))
    return path


def test_main_refuses_when_the_cap_bound_and_no_exclusion_is_named(tmp_path, monkeypatch, capsys):
    per_item = {f"i{k}": {s: 0.1 for s in (0, 1, 2)} for k in range(4)}
    rows = _rows(per_item)
    path = _readout(tmp_path, rows, {
        "i0": {"n_rows": 240, "n_hit_token_cap": 40, "token_cap_share": 0.166},
        "i1": {"n_rows": 240, "n_hit_token_cap": 0, "token_cap_share": 0.0},
        "i2": {"n_rows": 240, "n_hit_token_cap": 0, "token_cap_share": 0.0},
        "i3": {"n_rows": 240, "n_hit_token_cap": 0, "token_cap_share": 0.0},
    })
    argv = ["analyze", "--independent-readout", str(path), "--out-path", str(tmp_path / "out.json")]
    monkeypatch.setattr("sys.argv", argv)
    with pytest.raises(SystemExit):
        mod.main()
    assert not (tmp_path / "out.json").exists()

    monkeypatch.setattr("sys.argv", argv + ["--exclude-item", "i0"])
    mod.main()
    report = json.loads((tmp_path / "out.json").read_text())
    assert report["s1a_sizing"]["late_window"]["variance_components"]["n_items"] == 3
    assert report["sensitivity_with_capped_items"]["late_window"]["variance_components"]["n_items"] == 4
    assert "not_a_verdict" in report


def test_main_refuses_a_self_judged_readout(tmp_path, monkeypatch):
    rows = _rows({f"i{k}": {0: 0.1} for k in range(3)})
    path = _readout(tmp_path, rows, {f"i{k}": {"n_rows": 20, "n_hit_token_cap": 0,
                                               "token_cap_share": 0.0} for k in range(3)})
    payload = json.loads(path.read_text())
    payload["judge_kind"] = "self"
    path.write_text(json.dumps(payload))
    monkeypatch.setattr("sys.argv", ["analyze", "--independent-readout", str(path),
                                     "--out-path", str(tmp_path / "out.json")])
    with pytest.raises(SystemExit):
        mod.main()
