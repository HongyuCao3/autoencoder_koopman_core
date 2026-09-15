"""Tests for the GPU-2 pair: run_tsar_cefr_closed_loop_arms.py and
analyze_tsar_cefr_s3_gates.py (plan sections 5.4 / 6.3).

Plan 6.3 names the failure these tests exist to prevent: S3's two bugs both came
from ONE end-to-end fixture whose dicts were written into two files, so a join
that was wrong in production was right in the test. Every test below therefore
builds its own rows, and the arm-policy tests never touch the analyzer's rows.

The second thing under test is the set of REFUSALS. Four values plan 5.4 leaves
to a trial-set ruling, plus the seed axis, are pre-run raises rather than
defaults, and a guard that is never exercised is indistinguishable from one that
silently passes -- the lesson G-T1 wrote down when criterion (1) counted float
noise as dynamic range.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


arms = load("tsar_arms", "run_tsar_cefr_closed_loop_arms.py")
gates = load("tsar_gates", "analyze_tsar_cefr_s3_gates.py")


# =============================================================================
# ARM POLICIES -- each is the plan's own wording, so each test quotes it
# =============================================================================

def test_greedy_reactive_edges_are_the_plans_three_rules():
    """'>= 1 level -> step down; 0.5-1 -> half step; <= 0.5 -> hold'."""
    assert arms.greedy_reactive_action(3.0, "A2") == "step_down"      # d = 1.0
    assert arms.greedy_reactive_action(2.99, "A2") == "half_step_down"  # d = 0.99
    assert arms.greedy_reactive_action(2.5, "A2") == "half_step_down"  # d = 0.5
    assert arms.greedy_reactive_action(2.49, "A2") == "copy"           # d = 0.49
    assert arms.greedy_reactive_action(1.0, "A2") == "copy"            # overshoot


def test_fixed_ladder_counts_levels_not_steps():
    assert arms.fixed_ladder_actions(4.9, "A2", 6)[:3] == ["step_down"] * 3
    assert arms.fixed_ladder_actions(4.9, "A2", 6)[3:] == ["copy"] * 3
    assert arms.fixed_ladder_actions(2.1, "A2", 6) == ["copy"] * 6
    assert len(arms.fixed_ladder_actions(5.4, "A2", 2)) == 2


def test_one_shot_uses_the_level_naming_prompt_only_on_step_one():
    state = {"step": 0, "target_cefr": "A2"}
    assert arms.next_action("one_shot", state, None, None) == "one_shot_prompt"
    assert arms.next_action("one_shot", {**state, "step": 1}, None, None) == "copy"
    rendered = arms.render("one_shot_prompt", "Some text.", "A2")
    assert "CEFR level A2" in rendered and "not simpler than" in rendered


def test_operator_step_matches_a_hand_computed_value(tmp_path):
    fit = {"final_operator": {
        "K": [[0.5, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
              [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        "B": [[-1.0, -0.5, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        "c": [0.1, 0.0, 0.0, 0.0], "u_layout": ["step_down", "half_step_down", "paraphrase"]}}
    path = tmp_path / "op.json"
    path.write_text(json.dumps(fit))
    op = arms.Operator(path)
    assert op.step([4.0, 1.0, 4.0, 1.0], "copy")[0] == pytest.approx(2.1)
    assert op.step([4.0, 1.0, 4.0, 1.0], "step_down")[0] == pytest.approx(1.1)
    assert op.step([4.0, 1.0, 4.0, 1.0], "half_step_down")[0] == pytest.approx(1.6)


def test_mpc_picks_the_first_action_of_the_best_sequence(tmp_path):
    """Planted: only `step_down` moves ell, and lambda is small enough not to veto it."""
    fit = {"final_operator": {
        "K": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
              [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        "B": [[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        "c": [0.0, 0.0, 0.0, 0.0], "u_layout": ["step_down", "half_step_down", "paraphrase"]}}
    path = tmp_path / "op.json"
    path.write_text(json.dumps(fit))
    op = arms.Operator(path)
    assert arms.mpc_action(op, [5.0, 1.0, 5.0, 1.0], "A2", beta=0.0, lam=0.01) == "step_down"
    # A prohibitive action cost must flip it to the zero-input action.
    assert arms.mpc_action(op, [2.0, 1.0, 2.0, 1.0], "A2", beta=0.0, lam=100.0) == "copy"


def test_mpc_horizon_is_the_signed_four():
    assert arms.MPC_HORIZON == 4
    assert arms.MAX_STEPS == 6


# =============================================================================
# REFUSALS -- the unsigned values must stop the job, not pick a default
# =============================================================================

def make_args(**over):
    base = dict(mode="canonical", decoding="sample", decoding_temperature=0.7,
                seeds=[0, 1, 2], stop_on_arrival="on_copy", beta=1.0, lambda_cost=0.05,
                arms=["greedy_reactive"], dp_path_mapping=None, dp_path_reward=None)
    base.update(over)
    return type("A", (), base)()


def test_canonical_refuses_without_a_decoding_ruling():
    with pytest.raises(SystemExit, match="decoding"):
        arms.check_rulings(make_args(decoding=None))


def test_canonical_refuses_greedy_with_more_than_one_seed():
    with pytest.raises(SystemExit, match="byte-identical"):
        arms.check_rulings(make_args(decoding="greedy"))


def test_greedy_with_one_seed_is_allowed_but_beta_still_required():
    arms.check_rulings(make_args(decoding="greedy", seeds=[0]))
    with pytest.raises(SystemExit, match="beta"):
        arms.check_rulings(make_args(decoding="greedy", seeds=[0], beta=None))


def test_canonical_refuses_without_a_stop_rule():
    with pytest.raises(SystemExit, match="stop-on-arrival"):
        arms.check_rulings(make_args(stop_on_arrival=None))


def test_dp_path_adjacent_only_is_refused_because_it_collapses_into_fixed_ladder():
    with pytest.raises(SystemExit, match="fixed_ladder"):
        arms.check_rulings(make_args(arms=["dp_path"], dp_path_mapping="adjacent_only"))


def test_dp_path_level_prompt_still_needs_a_reward_source():
    with pytest.raises(SystemExit, match="dp-path-reward"):
        arms.check_rulings(make_args(arms=["dp_path"], dp_path_mapping="level_prompt"))


def test_the_published_reward_matrix_is_refused_because_it_is_not_in_the_repo():
    with pytest.raises(SystemExit, match="not in this repo"):
        arms.check_rulings(make_args(arms=["dp_path"], dp_path_mapping="level_prompt",
                                     dp_path_reward="published"))


def test_dp_path_with_the_signed_pair_of_rulings_is_allowed():
    arms.check_rulings(make_args(arms=["dp_path"], dp_path_mapping="level_prompt",
                                 dp_path_reward="trial_measured"))


def test_debug_mode_asks_for_nothing():
    arms.check_rulings(make_args(mode="debug", decoding=None, beta=None))


# =============================================================================
# THE ANALYZER -- rows built here, never shared with the arm tests above
# =============================================================================

def traj_row(arm, text_id, seed, step, level_official, meaning=0.95, tokens=100,
             source_official="B2", target=None):
    return {"arm": arm, "text_id": text_id, "source_id": text_id.split("-")[0],
            "target_cefr": target or text_id.split("-")[1].upper(), "seed": seed,
            "step": step, "action": "step_down", "n_tokens_out": tokens,
            "level_official": level_official, "meaning_to_source": meaning,
            "source_level_official": source_official}


def full_rows(level_by_arm, seeds=(0, 1, 2), n_items=4, meaning=0.95):
    rows = []
    for arm in gates.ARMS:
        for seed in seeds:
            for i in range(n_items):
                for step in (1, 2):
                    rows.append(traj_row(arm, f"{i:02d}-a2", seed, step,
                                         level_by_arm[arm], meaning=meaning))
    return rows


def test_terminal_row_is_the_largest_step_not_a_fixed_index():
    rows = [traj_row("one_shot", "01-a2", 0, 1, "B1"),
            traj_row("one_shot", "01-a2", 0, 4, "A2"),
            traj_row("one_shot", "01-a2", 0, 2, "B2")]
    terminals = gates.terminal_rows(rows)
    assert terminals[("one_shot", "01-a2", 0)]["level_official"] == "A2"


def test_duplicate_join_key_is_refused():
    rows = [traj_row("one_shot", "01-a2", 0, 1, "A2"),
            traj_row("one_shot", "01-a2", 0, 1, "B1")]
    path = pathlib.Path("/tmp/dup.jsonl")
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(SystemExit, match="not unique"):
        gates.load_rows(path)


def test_two_seeds_are_refused():
    rows = full_rows({a: "A2" for a in gates.ARMS}, seeds=(0, 1))
    with pytest.raises(SystemExit, match="below the floor"):
        gates.build_table(rows, 0.5, "source_level")


def test_a_missing_arm_stops_the_table():
    rows = [r for r in full_rows({a: "A2" for a in gates.ARMS}) if r["arm"] != "koopman_mpc"]
    with pytest.raises(SystemExit, match="koopman_mpc"):
        gates.build_table(rows, 0.5, "source_level")


def test_perfect_arm_scores_zero_and_a_one_level_miss_scores_one():
    rows = full_rows({**{a: "B1" for a in gates.ARMS}, "koopman_mpc": "A2"})
    table = gates.build_table(rows, 0.5, "source_level")["arms"]
    assert table["koopman_mpc"]["rmse_mean"] == pytest.approx(0.0)
    assert table["one_shot"]["rmse_mean"] == pytest.approx(1.0)
    assert "n = 3 seeds" in table["koopman_mpc"]["reportable_form"]


def test_meaning_failures_score_the_source_level_or_leave_the_denominator():
    rows = full_rows({a: "A2" for a in gates.ARMS}, meaning=0.10)
    kept = gates.build_table(rows, 0.5, "source_level")["arms"]["one_shot"]
    assert kept["rmse_mean"] == pytest.approx(2.0)   # source B2 = 4 vs target A2 = 2
    assert kept["meaning_failures"] == 12
    with pytest.raises(SystemExit, match="no scored trajectory"):
        gates.build_table(rows, 0.5, "exclude")


def test_paired_contrast_is_negative_when_ours_is_better():
    """Ours hits the target on three of four sources, the reference on none."""
    rows = []
    for arm in gates.ARMS:
        for seed in (0, 1, 2):
            for i in range(4):
                level = "A2" if (arm == gates.OURS and i < 3) else "B1"
                rows.append(traj_row(arm, f"{i:02d}-a2", seed, 1, level))
    got = gates.paired_contrast(gates.terminal_rows(rows), 0.5, "source_level")
    assert got["mean"] < 0
    assert got["ci95"][1] < 0.0, "three of four sources improved should exclude zero"
    assert "NEGATIVE means Ours is better" in got["sign_convention"]


def test_two_arms_that_collapse_onto_each_other_are_refused_not_scored():
    """Zero variance is the shape dp_path-as-fixed_ladder and greedy-seeds both take."""
    rows = full_rows({a: "B1" for a in gates.ARMS})
    with pytest.raises(SystemExit, match="zero variance"):
        gates.paired_contrast(gates.terminal_rows(rows), 0.5, "source_level")


def test_undecidable_needs_both_halves_of_the_s3_clause():
    """Ours and the reference differ on one source out of eight: tiny effect, wide CI."""
    rows = []
    for arm in gates.ARMS:
        for seed in (0, 1, 2):
            for i in range(8):
                level = "A2" if (arm == gates.OURS and i == 0) else "B1"
                rows.append(traj_row(arm, f"{i:02d}-a2", seed, 1, level))
    got = gates.paired_contrast(gates.terminal_rows(rows), 0.5, "source_level")
    assert got["mean"] < 0 and abs(got["mean"]) < got["mde"]
    assert got["ci95"][0] <= 0.0 <= got["ci95"][1]
    assert got["undecidable"] is True


def test_cost_axes_report_steps_and_tokens_separately():
    rows = full_rows({a: "A2" for a in gates.ARMS})
    cost = gates.cost_axes(rows, "one_shot")
    assert cost["mean_steps_used"] == pytest.approx(2.0)
    assert cost["mean_tokens"] == pytest.approx(200.0)


def test_canonical_analysis_refuses_without_the_unsigned_threshold(tmp_path):
    with pytest.raises(SystemExit, match="meaning-threshold"):
        gates.main(["--rows", str(tmp_path / "x.jsonl"), "--out-dir", str(tmp_path / "o"),
                    "--mode", "canonical"])
    with pytest.raises(SystemExit, match="meaning-failure-policy"):
        gates.main(["--rows", str(tmp_path / "x.jsonl"), "--out-dir", str(tmp_path / "o"),
                    "--mode", "canonical", "--meaning-threshold", "0.5"])


# =============================================================================
# EQUAL DENOMINATORS -- the failure `--stop-on-arrival on_copy` would introduce
# =============================================================================

def test_arms_scored_on_different_samples_are_refused():
    """A trajectory that stops before generating must not quietly leave the sample."""
    rows = full_rows({a: "A2" for a in gates.ARMS})
    rows = [r for r in rows if not (r["arm"] == "koopman_mpc" and r["text_id"] == "00-a2")]
    with pytest.raises(SystemExit, match="different samples"):
        gates.build_table(rows, 0.5, "source_level")


def test_a_step_zero_row_is_a_valid_terminal():
    """The source paragraph is the terminal state of a trajectory that never ran."""
    rows = []
    for arm in gates.ARMS:
        for seed in (0, 1, 2):
            for i in range(4):
                step = 0 if arm == "one_shot" else 1
                rows.append(traj_row(arm, f"{i:02d}-a2", seed, step, "B2", tokens=0))
    table = gates.build_table(rows, 0.5, "source_level")["arms"]
    assert table["one_shot"]["rmse_mean"] == pytest.approx(2.0)   # B2 = 4 vs A2 = 2
    assert table["one_shot"]["cost"]["mean_steps_used"] == pytest.approx(0.0)
    assert table["one_shot"]["cost"]["mean_tokens"] == pytest.approx(0.0)


# =============================================================================
# dp_path -- the reward matrix and the plan it produces
# =============================================================================

def test_reward_scheme_is_the_published_one():
    assert arms.reward_of("A2", "A2") == pytest.approx(1.0)
    assert arms.reward_of("A2", "B1") == pytest.approx(0.5)
    assert arms.reward_of("A2", "A1") == pytest.approx(0.5)
    assert arms.reward_of("A2", "B2") == pytest.approx(-1.0)


def test_reward_matrix_normalises_and_keeps_the_attempt_counts():
    raw = {("B2", "B1"): [1.0, 1.0], ("B2", "A2"): [-1.0], ("B1", "A2"): [0.5]}
    got = arms.normalise_rewards(raw)
    assert got["R"]["B2->B1"] == pytest.approx(1.0)
    assert got["R"]["B2->A2"] == pytest.approx(0.0)
    assert 0.0 < got["R"]["B1->A2"] < 1.0
    assert got["n_attempts"]["B2->B1"] == 2


def test_dp_prefers_the_higher_reward_path_over_the_short_one():
    """C1 -> A2 via B2,B1 scores 3 x 1.0; the direct jump scores 0.0."""
    R = {"C1->B2": 1.0, "B2->B1": 1.0, "B1->A2": 1.0, "C1->A2": 0.0,
         "C1->B1": 0.0, "B2->A2": 0.0}
    plan = arms.dp_plan(R, "C1", "A2", max_steps=6)
    assert plan["path"] == ["B2", "B1", "A2"]
    assert plan["unobserved_used"] == 0


def test_dp_takes_the_jump_when_the_jump_is_what_works():
    R = {"C1->B2": 0.0, "B2->B1": 0.0, "B1->A2": 0.0, "C1->A2": 1.0,
         "C1->B1": 0.0, "B2->A2": 0.0}
    assert arms.dp_plan(R, "C1", "A2", max_steps=6)["path"] == ["A2"]


def test_dp_prefers_a_fully_observed_path_even_at_lower_reward():
    """A transition this model was never observed making is not plannable."""
    R = {"C1->B2": 0.2, "B2->B1": 0.2, "B1->A2": 0.2}      # C1->A2 never attempted
    plan = arms.dp_plan(R, "C1", "A2", max_steps=6)
    assert plan["path"] == ["B2", "B1", "A2"]
    assert plan["unobserved_used"] == 0


def test_dp_is_empty_when_the_source_is_already_at_or_below_target():
    assert arms.dp_plan({}, "A2", "A2", max_steps=6)["path"] == []
    assert arms.dp_plan({}, "A1", "B1", max_steps=6)["path"] == []


def test_dp_path_emits_level_actions_then_stops():
    state = {"step": 0, "dp_plan": {"path": ["B1", "A2"]}, "target_cefr": "A2"}
    assert arms.next_action("dp_path", state, None, None) == "level:B1"
    assert arms.next_action("dp_path", {**state, "step": 1}, None, None) == "level:A2"
    assert arms.next_action("dp_path", {**state, "step": 2}, None, None) == "copy"


def test_level_action_renders_the_widened_template():
    rendered = arms.render("level:B2", "Some text.", "A2")
    assert "CEFR level B2" in rendered and "CEFR level A2" not in rendered


def test_dp_degeneracy_is_counted_not_repaired():
    """Uniform non-negative R makes the summed objective prefer the longest path."""
    R = {f"{j}->{i}": 0.5 for j in arms.LEVELS_LOW_TO_HIGH for i in arms.LEVELS_LOW_TO_HIGH
         if arms.CEFR_CODE[i] < arms.CEFR_CODE[j]}
    plan = arms.dp_plan(R, "C1", "A2", max_steps=6)
    assert plan["path"] == plan["adjacent_path"] == ["B2", "B1", "A2"]
    got = arms.dp_degeneracy({"01-a2": plan})
    assert got["share"] == pytest.approx(1.0)


def test_dp_degeneracy_sees_a_real_jump():
    R = {"C1->A2": 1.0, "C1->B2": 0.0, "B2->B1": 0.0, "B1->A2": 0.0,
         "C1->B1": 0.0, "B2->A2": 0.0}
    plan = arms.dp_plan(R, "C1", "A2", max_steps=6)
    assert plan["path"] == ["A2"] and plan["adjacent_path"] == ["B2", "B1", "A2"]
    assert arms.dp_degeneracy({"01-a2": plan})["share"] == pytest.approx(0.0)


def test_adjacent_path_walks_one_level_at_a_time():
    assert arms.adjacent_path("C1", "A2") == ["B2", "B1", "A2"]
    assert arms.adjacent_path("B1", "A2") == ["A2"]
    assert arms.adjacent_path("A2", "A2") == []
