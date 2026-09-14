"""Tests for the `tsar_cefr` GPU-1 open-loop excitation arm.

Two things are being protected. The ADMISSION checks (plan section 4.3) must
each catch their own planted defect, built from their own rows -- the S3 lesson
against one shared end-to-end fixture. And the EXCITATION itself must be
independent of the state and replayable, because an action sequence that drifts
with the bank's order or with a later item makes every identified operator
unreproducible.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_tsar_cefr_excitation_arm.py"
sys.path.insert(0, str(ROOT / "src"))

from persona_drift import tsar_cefr_actions as A  # noqa: E402


def load_script():
    spec = importlib.util.spec_from_file_location("excite", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


arm = load_script()


# --- the action space ------------------------------------------------------

def test_the_four_signed_actions_and_nothing_else():
    assert A.ACTION_NAMES == ("step_down", "half_step_down", "paraphrase", "copy")
    assert A.ZERO_INPUT_ACTION == "copy"


def test_actions_replay_from_seed_and_text_id_alone():
    assert A.draw_actions(0, "07-A2", 6) == A.draw_actions(0, "07-A2", 6)


def test_different_seeds_excite_differently():
    """Decoding is greedy, so the seed IS the excitation. If it did not move the
    action sequence the three seeds would be three identical runs."""
    assert A.draw_actions(0, "07-A2", 6) != A.draw_actions(1, "07-A2", 6)


def test_one_items_actions_do_not_depend_on_any_other_item():
    """Planted defect: a single RNG stream. Adding or reordering an item would
    then silently rewrite every later item's excitation."""
    before = A.draw_actions(0, "07-A2", 6)
    A.draw_actions(0, "01-A2", 6)
    A.draw_actions(0, "99-B1", 6)
    assert A.draw_actions(0, "07-A2", 6) == before


def test_the_prompt_carries_only_the_action_and_the_target():
    a2 = A.user_message("Some text.", "step_down", "A2")
    b1 = A.user_message("Some text.", "step_down", "B1")
    assert a2.replace("A2", "B1") == b1
    for name in A.ACTION_NAMES:
        assert A.action_by_name(name).phrase in A.user_message("Some text.", name, "B1")


def test_the_step_prompt_never_names_the_level_to_move_to():
    """`step_down` and `half_step_down` must stay separable near the target; a
    prompt that says "go to B1" collapses them once the text is already close."""
    for name in ("step_down", "half_step_down", "paraphrase", "copy"):
        assert "level B1" not in A.action_by_name(name).phrase


def test_an_unknown_target_level_is_refused_at_the_boundary():
    with pytest.raises(ValueError):
        A.user_message("Some text.", "copy", "C1")


def test_the_saturation_probe_is_allowed_to_name_the_level():
    """It is a different prompt on purpose -- naming the level is the probe."""
    assert "B1" in A.saturation_message("Some text.", "B1")


# --- the schedule ----------------------------------------------------------

class _Item:
    def __init__(self, text_id, source_id, target, original):
        self.text_id, self.source_id = text_id, source_id
        self.target_cefr, self.original = target, original


def _items(n_sources=3):
    return [_Item(f"{i:02d}-{t}", f"{i:02d}", t, f"paragraph {i}")
            for i in range(n_sources) for t in ("A2", "B1")]


def test_the_schedule_is_items_times_seeds():
    schedule = arm.build_schedule(_items(3), [0, 1, 2], 6)
    assert len(schedule) == 3 * 2 * 3
    assert all(len(entry["actions"]) == 6 for entry in schedule)


def test_duplicate_action_sequences_are_counted_not_hidden():
    """Under greedy decoding two seeds that draw the same six actions produce
    byte-identical rows. Expected at 4^6; it has to appear in the report."""
    schedule = arm.build_schedule(_items(1), [0, 1], 6)
    schedule[1]["actions"] = list(schedule[0]["actions"])
    assert arm.duplicate_action_sequences(schedule) == 1
    assert arm.duplicate_action_sequences(arm.build_schedule(_items(2), [0], 6)) == 0


def test_the_token_cap_scales_with_the_paragraph_and_has_a_floor():
    assert arm.token_cap(400) == 600
    assert arm.token_cap(10) == arm.MIN_MAX_NEW_TOKENS


# --- admission (plan section 4.3) ------------------------------------------

def _rows(actions_seq, levels=None, caps=None, text_ids=None):
    n = len(actions_seq)
    levels = levels if levels is not None else [3.0 + 0.1 * i for i in range(n)]
    caps = caps if caps is not None else [False] * n
    text_ids = text_ids if text_ids is not None else [f"{i:02d}-B1" for i in range(n)]
    return [{"text_id": text_ids[i], "step": 1 + i % 6, "action": actions_seq[i],
             "level_expected": levels[i], "hit_token_cap": caps[i],
             "text_in": "in", "text_out": "out"} for i in range(n)]


def test_action_balance_passes_on_a_uniform_draw():
    assert arm.action_balance(_rows(list(A.ACTION_NAMES) * 25))["passed"]


def test_action_balance_catches_a_starved_action():
    """Planted defect: `copy` almost never drawn. Without the zero-input action
    there is nothing to identify `A` against."""
    seq = ["step_down"] * 40 + ["half_step_down"] * 30 + ["paraphrase"] * 28 + ["copy"] * 2
    got = arm.action_balance(_rows(seq))
    assert not got["passed"] and got["shares"]["copy"] == pytest.approx(0.02)


def test_per_step_range_catches_the_defense_failure_at_one_step():
    """Planted defect: step 4 collapses to a single value. D-1 (2) is exactly
    the G-T1 question asked again on real trajectories."""
    rows = _rows(["copy"] * 24, levels=[2.0 + 0.11 * i for i in range(24)])
    for row in rows:
        if row["step"] == 4:
            row["level_expected"] = 2.5
    got = arm.per_step_range(rows)
    assert not got["passed"] and not got["per_step"][4]["passed"]
    assert got["per_step"][1]["passed"] and got["death_condition"] == "D-1 (2)"


def test_per_step_range_passes_when_every_step_still_moves():
    rows = _rows(["copy"] * 18, levels=[2.0 + 0.13 * i for i in range(18)])
    assert arm.per_step_range(rows)["passed"]


def test_cap_pressure_is_judged_per_item_not_pooled():
    """Planted defect: one item hits the cap on every step while the pooled
    share stays under 5%. Pooling is what hides the long paragraphs -- the S1
    lesson (`tuple_294_1`, three times)."""
    caps = [False] * 200
    ids = [f"{i // 10:02d}-B1" for i in range(200)]
    for i in range(0, 4):
        caps[i] = True                      # item 00: 4 of its 10 steps, and only it
    got = arm.cap_pressure(_rows(["copy"] * 200, caps=caps, text_ids=ids))
    assert got["pooled_share"] < arm.MAX_ITEM_CAP_HIT_SHARE
    assert not got["passed"] and got["items_over_cap"] == ["00-B1"]


def test_cap_pressure_passes_when_nothing_is_truncated():
    assert arm.cap_pressure(_rows(["copy"] * 40))["passed"]


# --- D-0 -------------------------------------------------------------------

def _probe(hits, total):
    return [{"target_cefr": "B1", "level_official": "B1" if i < hits else "B2"}
            for i in range(total)]


def test_saturation_fires_above_eighty_percent():
    """D-0: if one move already lands on the target there is no regulation
    problem for a controller to be better at."""
    got = arm.saturation_probe_verdict(_probe(34, 40))
    assert got["fires"] and got["one_shot_hit_share"] == pytest.approx(0.85)


def test_saturation_does_not_fire_at_the_threshold():
    assert not arm.saturation_probe_verdict(_probe(32, 40))["fires"]


def test_saturation_is_judged_on_the_official_readout():
    """The control readout is the expected level and has no argmax; D-0 is a
    hit/miss question, so it goes on the reporting layer."""
    assert "official" in arm.saturation_probe_verdict(_probe(1, 40))["note"]


# --- the report ------------------------------------------------------------

def test_the_report_does_not_claim_the_two_conditions_it_cannot_decide():
    rows = _rows(list(A.ACTION_NAMES) * 25)
    report = arm.arm_report(rows, _probe(1, 40), arm.build_schedule(_items(3), [0], 6))
    assert report["still_open"] == ["D-2 executor authority", "D-2.5 causal upper bound"]


def test_unchanged_verbatim_is_reported_for_the_zero_input_action():
    rows = _rows(["copy"] * 4)
    rows[0]["text_out"] = rows[0]["text_in"]
    got = arm.unchanged_verbatim(rows)
    assert got["n_copy_steps"] == 4 and got["share"] == pytest.approx(0.25)


def test_the_signed_constants_are_the_plans():
    assert arm.SIGNED_TEMPERATURE == 6.5            # G-T1, not re-fitted on this arm
    assert arm.ACTION_SHARE_BAND == (0.20, 0.30)
    assert arm.MIN_STEP_DISTINCT_LEVELS == 3
    assert arm.MAX_ITEM_CAP_HIT_SHARE == 0.05
    assert arm.SATURATION_HIT_THRESHOLD == 0.80
    assert arm.CAP_LENGTH_MULTIPLIER == 1.5
