"""Tests for scripts/analyze_sequor_s1_gates.py.

Every gate here gets a planted positive control (an effect the gate must see)
and a planted negative control (a defect the gate must catch). A gate that
silently passes is worse than no gate: the failure it was built to catch
becomes invisible in the report, which is the standing rule behind
`run_config_guard` and its meta-test.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib

import pytest

SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "analyze_sequor_s1_gates.py"

ARMS = ["zero_control", "constant_remind", "bernoulli", "antithetic"]
N_ITEMS = 6
N_TURNS = 20
SEEDS = [0, 1, 2]
LATE_FROM = 15


def _load():
    spec = importlib.util.spec_from_file_location("_s1_gates", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pilot(module):
    return module._sibling_module("analyze_sequor_s1_pilot.py")


def _y(arm: str, item: int, turn: int, seed: int, effect: float) -> float:
    """A readout with item-to-item spread, a turn ramp downward, and a planted
    late-window gain under `constant_remind`."""

    base = 0.5 + 0.05 * (item % 4) + 0.01 * seed - 0.01 * turn
    if arm == "constant_remind":
        base += effect
    if arm in ("bernoulli", "antithetic"):
        base += effect / 2
    return round(min(1.0, max(0.0, base)), 6)


def _rows(effect: float = 0.20, saturate: tuple[str, int] | None = None) -> list[dict]:
    rows = []
    for arm in ARMS:
        for item in range(N_ITEMS):
            for seed in SEEDS:
                for turn in range(1, N_TURNS + 1):
                    if arm == "zero_control":
                        u = 0
                    elif arm == "constant_remind":
                        u = int(turn >= 2)
                    elif arm == "bernoulli":
                        u = int(turn >= 2 and (turn + item + seed) % 2 == 0)
                    else:
                        u = int(turn >= 2 and (turn + item + seed) % 2 == 1)
                    y = _y(arm, item, turn, seed, effect)
                    if saturate and (arm, turn) == saturate:
                        y = 1.0
                    rows.append({
                        "trajectory_id": f"tuple_{item}__{arm}__s{seed}",
                        "item_id": f"tuple_{item}", "turn": turn, "branch": arm,
                        "u_remind": u, "seed": seed, "y_graded": y,
                        "followed": [True, False, True], "echo_jaccard_prev": None,
                        "inserted_tokens": 0, "hit_token_cap": False,
                    })
    return rows


def _arm_report(rows: list[dict]) -> dict:
    u_mean = {}
    for seed in SEEDS:
        acted = [r for r in rows if r["branch"] == "bernoulli" and r["seed"] == seed and r["turn"] >= 2]
        u_mean[str(seed)] = sum(r["u_remind"] for r in acted) / len(acted)
    return {
        "mode": "canonical", "arms": ARMS, "seeds": SEEDS, "n_turns": N_TURNS,
        "n_items": N_ITEMS, "n_rows": len(rows),
        "schedule_checks": {
            str(seed): {"n_items": N_ITEMS, "n_turns": N_TURNS,
                        "bernoulli_u_mean": u_mean[str(seed)],
                        "antithetic_is_exact_complement": True, "turn1_action_free": True}
            for seed in SEEDS
        },
        "cell_coverage": {"n_cells": N_ITEMS * (N_TURNS - 1) * len(SEEDS),
                          "n_cells_with_both_arms": N_ITEMS * (N_TURNS - 1) * len(SEEDS),
                          "n_cells_with_exactly_one_reminder": N_ITEMS * (N_TURNS - 1) * len(SEEDS),
                          "cells_are_matched": True},
        "gold_coverage": {"n_items": N_ITEMS, "n_constraints_judged": 3 * N_ITEMS,
                          "n_constraints_in_calibration_set": 6,
                          "share_outside_calibration_set": 1 - 6 / (3 * N_ITEMS),
                          "n_items_fully_covered": 2, "gold_coverage_by_item": {}},
        "cap_criterion": 0.05, "n_hit_token_cap": 0, "token_cap_share": 0.0,
        "token_cap_by_item": {f"tuple_{i}": {"n_rows": 10, "n_hit_token_cap": 0, "token_cap_share": 0.0}
                              for i in range(N_ITEMS)},
        "items_over_cap_criterion": [], "cap_criterion_pass": True, "max_new_tokens": 2048,
    }


def _write_arm(tmp_path: pathlib.Path, rows: list[dict], *, judge_kind: str = "independent") -> pathlib.Path:
    arm_dir = tmp_path / "arm"
    arm_dir.mkdir()
    (arm_dir / "arm_report.json").write_text(json.dumps(_arm_report(rows)))
    readout = {
        "mode": "canonical", "arm_dir": str(arm_dir), "agent_model": "agent",
        "judge_model": "judge", "judge_kind": judge_kind, "k_constraints": 3,
        "n_rows": len(rows), "n_rows_unusable": 0, "rows": rows,
    }
    path = tmp_path / f"readout_{judge_kind}.json"
    path.write_text(json.dumps(readout))
    return path


def _run(module, rows: list[dict]) -> dict:
    arm_report = _arm_report(rows)
    return module.gate_g_s1(arm_report, rows, ARMS, N_TURNS)


def test_admission_passes_on_a_well_formed_arm():
    module = _load()
    gate = _run(module, _rows())
    assert gate["verdict"] == "PASS", gate["checks"]
    assert gate["n_rows"] == gate["n_rows_expected"] == len(ARMS) * N_ITEMS * N_TURNS * len(SEEDS)


def test_row_count_clause_counts_seeds():
    """The plan's "arms x 40 x 20" predates the 3-seed ruling. If the seed
    factor were dropped, a complete arm would fail admission -- so the count
    is asserted against the real product, and a genuinely short arm still
    fails."""

    module = _load()
    rows = _rows()
    assert _run(module, rows)["checks"]["row_count_matches_design"]
    truncated = [r for r in rows if not (r["branch"] == "antithetic" and r["turn"] == N_TURNS)]
    assert not _run(module, truncated)["checks"]["row_count_matches_design"]


def test_spread_clause_is_per_arm_and_pooling_would_hide_it():
    """A saturated turn in ONE arm must fail admission even though the pooled
    set over all four arms still has spread at that turn. This is the clause
    signed in screening section 10 item 9 -- the pilot's `constant_remind`
    had 2 distinct values at t2 while the pool had more."""

    module = _load()
    rows = _rows(saturate=("constant_remind", 7))
    gate = _run(module, rows)
    assert gate["verdict"] == "FAIL"
    assert gate["turns_failing_spread_by_arm"]["constant_remind"] == ["7"]
    assert gate["turns_failing_spread_by_arm"]["zero_control"] == []

    pooled = [r["y_graded"] for r in rows if r["turn"] == 7]
    assert len(set(pooled)) >= 3, "the pooled set must still look healthy, else the test proves nothing"


def test_unbalanced_u_fails_admission():
    module = _load()
    rows = _rows()
    for row in rows:
        if row["branch"] == "bernoulli" and row["turn"] >= 2 and row["seed"] == 1:
            row["u_remind"] = 1
    gate = _run(module, rows)
    assert gate["verdict"] == "FAIL"
    assert gate["u_mean_in_band"] == {"0": True, "1": False, "2": True}


def test_a_turn_with_only_one_action_fails_admission():
    module = _load()
    rows = _rows()
    for row in rows:
        if row["branch"] in ("bernoulli", "antithetic") and row["turn"] == 9:
            row["u_remind"] = 0
    gate = _run(module, rows)
    assert gate["verdict"] == "FAIL"
    assert gate["turns_missing_an_action"] == [9]


def test_planted_endpoint_effect_is_recovered_and_ruled_resolved():
    module = _load()
    pilot = _pilot(module)
    rows = _rows(effect=0.20)
    block = module.s1a_contrast(pilot, rows, range(LATE_FROM, N_TURNS + 1), seed=0)
    assert block["contrast"] == pytest.approx(0.20, abs=0.01)
    assert block["ci"][0] > 0
    assert block["effect_over_mde"] > 1
    assert module.s1a_verdict(block)["verdict"] == "RESOLVED"


def test_no_effect_hands_back_and_never_closes_the_line():
    """The pre-registered consequence of a null endpoint contrast is to
    re-size S1, not to rule on executor authority (K3 already answered that on
    one-step pairs). No branch of this verdict may close the line."""

    module = _load()
    pilot = _pilot(module)
    block = module.s1a_contrast(pilot, _rows(effect=0.0), range(LATE_FROM, N_TURNS + 1), seed=0)
    assert block["contrast"] == pytest.approx(0.0, abs=1e-9)
    verdict = module.s1a_verdict(block)
    assert verdict["verdict"] == "REPORT_AND_HAND_BACK"
    assert verdict["closes_the_line"] is False


def test_significant_but_under_its_own_mde_is_handed_back_not_promoted():
    """The S0-0 K3 shape: a CI excluding zero at ~1.96 sigma while the effect
    sits below the 80%-power MDE. RESOLVED requires both halves."""

    module = _load()
    block = {"contrast": 0.05, "ci": [0.01, 0.09], "mde_at_80pct_this_arm": 0.08,
             "effect_over_mde": 0.625, "ci_excludes_zero": True, "resolved": False}
    verdict = module.s1a_verdict(block)
    assert verdict["verdict"] == "REPORT_AND_HAND_BACK"
    assert "below" in verdict["reason"]


def test_the_estimand_window_is_not_free_to_move():
    """A late-only effect must be visible in the signed window and absent from
    an early window: if the two windows gave the same answer, the pre-
    registration would be decorative."""

    module = _load()
    pilot = _pilot(module)
    rows = []
    for row in _rows(effect=0.0):
        if row["branch"] == "constant_remind" and row["turn"] >= LATE_FROM:
            row = {**row, "y_graded": row["y_graded"] + 0.2}
        rows.append(row)
    late = module.s1a_contrast(pilot, rows, range(LATE_FROM, N_TURNS + 1), seed=0)
    early = module.s1a_contrast(pilot, rows, range(2, LATE_FROM), seed=0)
    assert late["contrast"] == pytest.approx(0.2, abs=0.01)
    assert early["contrast"] == pytest.approx(0.0, abs=1e-9)


def test_the_self_readout_is_refused(tmp_path):
    """Gates are computed on the reporting readout. The self-judged one is the
    in-loop SELECTION signal and may never produce a reported number."""

    module = _load()
    gates = module._sibling_module("analyze_sequor_s0_0_gates.py")
    path = _write_arm(tmp_path, _rows(), judge_kind="self")
    with pytest.raises(SystemExit, match="judge_kind"):
        gates.load_readout(path, "independent")


def test_debug_artifacts_are_refused(tmp_path):
    module = _load()
    gates = module._sibling_module("analyze_sequor_s0_0_gates.py")
    path = _write_arm(tmp_path, _rows())
    payload = json.loads(path.read_text())
    payload["mode"] = "debug"
    path.write_text(json.dumps(payload))
    with pytest.raises(SystemExit, match="debug"):
        gates.load_readout(path, "independent")


def test_end_to_end_writes_a_report_and_refuses_to_overwrite(tmp_path, monkeypatch):
    module = _load()
    path = _write_arm(tmp_path, _rows())
    out = tmp_path / "s1_gates_report.json"
    monkeypatch.setattr(
        "sys.argv",
        ["analyze_sequor_s1_gates.py", "--independent-readout", str(path), "--out-path", str(out),
         "--late-from", str(LATE_FROM)],
    )
    module.main()
    report = json.loads(out.read_text())
    assert report["g_s1"]["verdict"] == "PASS"
    assert report["s1a_verdict"]["verdict"] == "RESOLVED"
    assert report["s1a"]["late_window"]["judge_kind"] == "independent"
    assert "outside the judge's calibration set" in report["caveat"]

    with pytest.raises(SystemExit, match="refusing to overwrite"):
        module.main()
