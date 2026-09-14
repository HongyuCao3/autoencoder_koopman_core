"""Tests for the zero-GPU G-S1 re-score under a named item exclusion.

What is being protected is the EXCLUSION FLAG, not the arithmetic. An
`--exclude-item` switch is the kind of tool that grows: today it drops the one
item the gate named, tomorrow it drops the item that spoils a result. Each
guard therefore gets its own planted defect, built from its own rows (the S3
lesson against one shared end-to-end fixture):

  * the flag refuses an item the cap check did not flag, and refuses an item
    that is not in the rows at all;
  * every admission block is recomputed after the drop, so a defect that
    survives in the retained rows still fails the gate;
  * the grid check catches a dropped-and-duplicated cell that a row COUNT
    would pass;
  * the dropped rows are reported and left on disk.

If these pass silently while the flag is broken, the failure they exist to
catch is invisible in the report -- the run_config_guard rule.
"""

from __future__ import annotations

import importlib.util
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analyze_tsar_cefr_gpu1.py"
sys.path.insert(0, str(ROOT / "src"))


def load_script():
    spec = importlib.util.spec_from_file_location("tsar_gpu1", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ana = load_script()
ACTIONS = ("step_down", "half_step_down", "paraphrase", "copy")


def make_rows(n_items=4, seeds=(0, 1, 2), steps=(1, 2, 3, 4, 5, 6)):
    """A clean arm: balanced actions, range at every step, nothing truncated.

    Step 1's action is keyed on (item, seed) so all four actions reach it --
    the step-1 confirmatory contrast needs `step_down` and `copy` to coexist
    there. The remaining steps cycle a counter, which keeps the pooled shares
    inside the [0.20, 0.30] band the admission gate checks.
    """
    rows, counter = [], 0
    for i in range(n_items):
        text_id = f"{i:02d}-a2"
        for seed in seeds:
            for step in steps:
                if step == 1:
                    action = ACTIONS[(i + seed) % 4]
                else:
                    action = ACTIONS[counter % 4]
                    counter += 1
                rows.append({
                    "text_id": text_id, "source_id": f"{i:02d}", "target_cefr": "A2",
                    "seed": seed, "step": step, "action": action,
                    "text_in": "in", "text_out": "in" if action == "copy" else "out",
                    "n_tokens_out": 100, "hit_token_cap": False,
                    "level_expected": 3.0 + 0.1 * i + 0.05 * seed,
                    "level_official": "B1",
                    "meaning_to_source": 0.9, "fkgl": 9.0,
                    "source_level_expected": 3.4, "source_fkgl": 10.0,
                })
    return rows


def with_action_effect(rows, step_down_effect, jitter=0.1):
    """Levels driven by the action, plus per-item jitter so the range block
    still passes -- an admission failure would mask what the D-2 test is for."""
    for row in rows:
        base = 3.5 + jitter * (int(row["source_id"]) % 7)
        row["source_level_expected"] = base
        row["level_expected"] = base + (step_down_effect if row["action"] == "step_down" else 0.0)
    return rows


PROBE = [{"level_official": "B1", "target_cefr": "A2"} for _ in range(10)]


def truncate(rows, text_id, n):
    """Plant n truncations on one item -- the defect the cap check exists for."""
    hit = [r for r in rows if r["text_id"] == text_id][:n]
    for row in hit:
        row["hit_token_cap"] = True
    return rows


# --- the clean arm admits ---------------------------------------------------

def test_clean_arm_admits_with_no_exclusion():
    report = ana.build_report(make_rows(), PROBE, [])
    assert report["admitted"] is True
    assert report["exclusion"]["items"] == []


# --- guard 1: the flag reaches only what the gate named ---------------------

def test_refuses_an_item_the_cap_check_did_not_flag():
    rows = truncate(make_rows(), "00-a2", 1)
    with pytest.raises(SystemExit) as err:
        ana.build_report(rows, PROBE, ["01-a2"])       # a clean item
    assert "did not flag it" in str(err.value)


def test_refuses_an_item_that_is_not_in_the_rows():
    with pytest.raises(SystemExit) as err:
        ana.build_report(make_rows(), PROBE, ["99-zz"])
    assert "no such text_id" in str(err.value)


def test_refuses_every_exclusion_when_the_cap_check_flagged_nothing():
    with pytest.raises(SystemExit) as err:
        ana.build_report(make_rows(), PROBE, ["00-a2"])
    assert "flagged: none" in str(err.value)


# --- the exclusion does what it is for, and no more -------------------------

def test_excluding_the_flagged_item_flips_the_verdict():
    rows = truncate(make_rows(), "00-a2", 1)
    report = ana.build_report(rows, PROBE, ["00-a2"])
    assert report["admitted_before_exclusion"] is False
    assert report["admitted"] is True
    assert report["admission_before_exclusion"]["cap_pressure"]["items_over_cap"] == ["00-a2"]
    assert report["admission"]["cap_pressure"]["items_over_cap"] == []


def test_a_second_item_still_over_cap_keeps_the_gate_failing():
    """Dropping one named item must not blanket-pass the arm."""
    rows = truncate(truncate(make_rows(), "00-a2", 1), "01-a2", 1)
    report = ana.build_report(rows, PROBE, ["00-a2"])
    assert report["admitted"] is False
    assert report["admission"]["cap_pressure"]["items_over_cap"] == ["01-a2"]


def test_every_block_is_recomputed_after_the_drop_not_just_the_failing_one():
    """An action-balance defect in the RETAINED rows must still fail the gate
    even though the exclusion fixed the block that originally failed."""
    rows = truncate(make_rows(), "00-a2", 1)
    for row in rows:
        if row["text_id"] != "00-a2":
            row["action"] = "copy"                      # share 1.0, band is [0.20, 0.30]
    report = ana.build_report(rows, PROBE, ["00-a2"])
    assert report["admission"]["cap_pressure"]["passed"] is True
    assert report["admission"]["action_balance"]["passed"] is False
    assert report["admitted"] is False


def test_row_count_and_grid_follow_the_drop():
    rows = truncate(make_rows(), "00-a2", 1)
    report = ana.build_report(rows, PROBE, ["00-a2"])
    assert report["admission"]["row_count"]["value"] == 54      # 3 items x 3 seeds x 6 steps
    assert report["admission"]["row_count"]["n_items"] == 3
    assert report["admission"]["row_count"]["passed"] is True


# --- guard 2: the grid check is stricter than a row count -------------------

def test_grid_catches_a_dropped_and_duplicated_cell_that_a_count_would_pass():
    rows = make_rows()
    rows[5] = dict(rows[4])                             # same cell twice, count unchanged
    block = ana.grid_completeness(rows)
    assert block["value"] == block["expected"]          # a count alone passes
    assert block["n_missing_cells"] == 1
    assert block["n_duplicated_cells"] == 1
    assert block["passed"] is False


# --- guard 3: what left is reported, and the rows are untouched -------------

def test_the_excluded_item_is_profiled_in_the_same_file():
    rows = truncate(make_rows(), "00-a2", 1)
    report = ana.build_report(rows, PROBE, ["00-a2"])
    profile = report["exclusion"]["profiles"][0]
    assert profile["text_id"] == "00-a2"
    assert profile["n_rows_removed"] == 18
    assert profile["cap_hits"] == 1
    assert profile["cap_hit_share"] == pytest.approx(1 / 18)
    assert profile["truncated_rows"][0]["action"] in ACTIONS
    assert "level_expected_by_action" in profile["retained_pool_for_contrast"]
    assert report["exclusion"]["rows_are_untouched_on_disk"] is True


def test_the_report_carries_both_verdicts_and_the_ruling():
    rows = truncate(make_rows(), "00-a2", 1)
    report = ana.build_report(rows, PROBE, ["00-a2"])
    assert "admission_before_exclusion" in report
    assert "2026-09-14" in report["exclusion"]["ruling"]
    assert "S1 precedent" in report["exclusion"]["pre_registered_by"]
    assert str(report["exclusion"]["n_rows_retained"]) in report["caption_obligation"]


def test_the_threshold_is_not_settable_from_the_command_line():
    """A gate whose threshold is an argument is not a gate (the G-T1 rule)."""
    out = subprocess.run([sys.executable, str(SCRIPT), "--help"],
                         capture_output=True, text=True, timeout=300)
    assert "--exclude-item" in out.stdout
    for settable in ("--threshold", "--max-item-cap", "--cap-hit-share", "--min-distinct"):
        assert settable not in out.stdout


# --- the drop is checked for direction, and the check is labelled -----------

def test_direction_diagnostic_names_a_backward_item_as_backward():
    """Plant an item where step_down reads HARDER than copy: the diagnostic
    must rank it most backward and must show the drop moving the pooled gap
    in the actuator's favour."""
    rows = truncate(make_rows(), "00-a2", 1)
    for row in rows:
        if row["text_id"] == "00-a2":
            row["level_expected"] = 4.0 if row["action"] == "step_down" else 2.0
        else:
            row["level_expected"] = 2.0 if row["action"] == "step_down" else 4.0
    diag = ana.direction_diagnostic(rows, [r for r in rows if r["text_id"] != "00-a2"], "00-a2")
    assert diag["excluded_item"] < 0                       # backward
    assert diag["excluded_item_rank_from_most_backward"] == 1
    assert diag["retained_rows"] > diag["all_rows"]        # the drop flatters the actuator
    assert diag["n_items_backward"] == 1


def test_direction_diagnostic_is_labelled_as_not_being_d2():
    rows = truncate(make_rows(), "00-a2", 1)
    report = ana.build_report(rows, PROBE, ["00-a2"])
    diag = report["exclusion"]["profiles"][0]["direction_diagnostic"]
    assert "D-2 pairs by source" in diag["not_the_d2_answer"]
    assert "sign_convention" in diag


# --- D-2: the one-step change, and the pre-registered branch ----------------

def test_input_level_is_the_previous_step_output_and_the_source_at_step_one():
    rows = make_rows(n_items=1, seeds=(0,), steps=(1, 2, 3))
    for r in rows:
        r["level_expected"] = {1: 3.0, 2: 2.5, 3: 2.2}[r["step"]]
        r["source_level_expected"] = 3.4
    dated = {r["step"]: r for r in ana.with_input_level(rows)}
    assert dated[1]["level_in"] == 3.4                       # the source
    assert dated[2]["level_in"] == 3.0                       # step 1's output
    assert dated[3]["delta_level"] == pytest.approx(-0.3)


def test_d2_resolves_when_step_down_drives_the_level_down():
    rows = with_action_effect(make_rows(n_items=30), step_down_effect=-1.0)
    d2 = ana.d2_contrast(ana.with_input_level(rows))
    verdict = ana.d2_verdict(d2, ana.d2_contrast(ana.with_input_level(rows), steps=(1,)))
    assert d2["point"] < 0
    assert verdict["verdict"] == "RESOLVED"
    assert verdict["fires"] is False


def test_d2_fires_when_the_effect_is_significant_but_backwards():
    """Authority with the sign reversed is not authority."""
    rows = with_action_effect(make_rows(n_items=30), step_down_effect=+1.0)
    d2 = ana.d2_contrast(ana.with_input_level(rows))
    verdict = ana.d2_verdict(d2, ana.d2_contrast(ana.with_input_level(rows), steps=(1,)))
    assert verdict["verdict"] == "RESOLVED_WRONG_DIRECTION"
    assert verdict["fires"] is True


def test_d2_is_undecidable_when_the_effect_sits_under_the_arms_own_mde():
    """A contrast below the MDE is a point estimate at under 80% power -- the
    lesson K3 taught this line -- and is pre-registered as UNDECIDABLE."""
    import random as _r
    rng = _r.Random(0)
    rows = with_action_effect(make_rows(n_items=30), step_down_effect=0.0)
    for r in rows:
        r["level_expected"] += rng.gauss(0, 0.5)               # noise, no action effect
    d2 = ana.d2_contrast(ana.with_input_level(rows))
    verdict = ana.d2_verdict(d2, ana.d2_contrast(ana.with_input_level(rows), steps=(1,)))
    assert d2["abs_effect_over_mde"] < 1.0
    assert verdict["verdict"] == "UNDECIDABLE"
    assert verdict["fires"] is True


def test_d2_pairs_and_bootstraps_over_the_unit_the_plan_fixed():
    rows = make_rows(n_items=30)
    d2 = ana.d2_contrast(ana.with_input_level(rows))
    assert d2["paired_by"] == "source_id" == d2["bootstrap_over"]
    assert d2["n_sources_paired"] <= 30


def test_d2_refuses_to_run_when_admission_does_not_pass():
    """A death condition computed on rejected rows has no standing."""
    rows = truncate(make_rows(n_items=30), "00-a2", 1)
    with pytest.raises(SystemExit) as err:
        ana.d2_report(rows, [])                              # the flagged item NOT excluded
    assert "refusing to judge D-2" in str(err.value)
    assert "cap_pressure" in str(err.value)


def test_d2_runs_once_the_named_exclusion_makes_admission_pass():
    rows = truncate(with_action_effect(make_rows(n_items=30), -1.0), "00-a2", 1)
    report = ana.d2_report(rows, ["00-a2"])
    assert report["verdict"]["death_condition"] == "D-2"
    assert report["exclusion"]["n_rows_retained"] == len(rows) - 18


def test_step_one_confirmatory_reports_itself_uncomputable_without_taking_d2_down():
    """The random excitation may never draw `copy` at step 1 for enough sources.
    That is a fact about the draw, recorded -- not a reason the primary dies."""
    rows = with_action_effect(make_rows(n_items=30), step_down_effect=-1.0)
    for r in rows:
        if r["step"] == 1 and r["action"] == "copy":
            r["action"] = "paraphrase"                       # no `copy` at step 1 at all
    report = ana.d2_report(rows, [])
    assert report["confirmatory_step1_only"]["computable"] is False
    assert report["verdict"]["confirmatory_computable"] is False
    assert report["verdict"]["confirmatory_agrees_in_sign"] is None
    assert report["verdict"]["verdict"] == "RESOLVED"         # the primary still stands
