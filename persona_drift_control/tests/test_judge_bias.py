import math

from persona_drift.judge_bias import analyze_judge_bias, pair_rows, readout_quality


def row(tid, turn, y, *, agent_message=None, u_remind=0, attack_id="a1", judge_model="self", seed=0):
    return {
        "trajectory_id": tid,
        "turn": turn,
        "attack_id": attack_id,
        "seed": seed,
        "u_remind": u_remind,
        "y_safety": y,
        "agent_message": agent_message if agent_message is not None else f"{tid}-{turn}",
        "judge_model": judge_model,
    }


def paired(self_scores, indep_scores, **kw):
    self_rows = [row("t1", i + 1, y, **kw) for i, y in enumerate(self_scores)]
    indep_rows = [row("t1", i + 1, y, judge_model="indep", **kw) for i, y in enumerate(indep_scores)]
    pairs, diagnostics = pair_rows(self_rows, indep_rows)
    return pairs, diagnostics


def test_pair_rows_joins_on_trajectory_and_turn():
    pairs, diag = paired([1.0, 0.75], [0.75, 0.5])
    assert diag["n_paired"] == 2
    assert diag["n_text_mismatch"] == 0
    assert [(p["y_self"], p["y_indep"]) for p in pairs] == [(1.0, 0.75), (0.75, 0.5)]
    assert diag["self_judge_model"] == ["self"] and diag["indep_judge_model"] == ["indep"]


def test_pair_rows_refuses_to_pair_rows_whose_reply_differs():
    # The whole comparison rests on both judges having scored the same text;
    # a mismatch must be counted and excluded, not silently paired.
    self_rows = [row("t1", 1, 1.0, agent_message="original reply")]
    indep_rows = [row("t1", 1, 0.5, agent_message="a resampled reply", judge_model="indep")]
    pairs, diag = pair_rows(self_rows, indep_rows)
    assert pairs == []
    assert diag["n_text_mismatch"] == 1


def test_pair_rows_reports_rows_present_on_only_one_side():
    self_rows = [row("t1", 1, 1.0), row("t1", 2, 1.0)]
    indep_rows = [row("t1", 1, 1.0, judge_model="indep")]
    _, diag = pair_rows(self_rows, indep_rows)
    assert diag["n_self_only"] == 1 and diag["n_indep_only"] == 0


def test_bias_direction_and_sign_test_flag_one_way_misses():
    # Independent judge strictly harsher on 4 rows, never more lenient --
    # the sycophancy line's "self-judge never over-flags, only misses".
    pairs, _ = paired([1.0, 1.0, 1.0, 1.0, 1.0], [0.75, 0.5, 1.0, 0.75, 0.5])
    result = analyze_judge_bias(pairs)["overall"]
    assert result["n_independent_stricter"] == 4
    assert result["n_independent_looser"] == 0
    assert result["mean_diff_indep_minus_self"] < 0
    assert result["sign_test_p"] == 0.125  # 2 * 0.5**4, the smallest 4 one-way rows can give
    assert result["disagree_rate"] == 0.8


def test_nan_scores_are_excluded_from_the_bias_but_counted():
    pairs, diag = paired([1.0, float("nan")], [0.75, 0.5])
    assert diag["n_nan_self"] == 1
    result = analyze_judge_bias(pairs)["overall"]
    assert result["n"] == 1


def test_a_constant_offset_is_reported_as_turn_independent():
    pairs, _ = paired([1.0, 1.0, 1.0, 1.0], [0.75, 0.75, 0.75, 0.75])
    trend = analyze_judge_bias(pairs)["bias_vs_turn"]
    assert trend["signed_diff_slope"] == 0.0


def test_a_growing_bias_is_reported_as_turn_dependent():
    # This is the case that would make new-Q1's slope itself biased rather
    # than merely compressed.
    pairs, _ = paired([1.0, 1.0, 1.0, 1.0], [1.0, 0.75, 0.5, 0.25])
    trend = analyze_judge_bias(pairs)["bias_vs_turn"]
    assert trend["signed_diff_slope"] < -0.2
    assert trend["signed_diff_p"] < 0.05


def test_reminder_dependent_bias_is_separated_from_the_level_shift():
    reminded = [row("t1", i + 1, 1.0, u_remind=1) for i in range(4)]
    unreminded = [row("t2", i + 1, 1.0, u_remind=0) for i in range(4)]
    reminded_indep = [row("t1", i + 1, 0.5, u_remind=1, judge_model="indep") for i in range(4)]
    unreminded_indep = [row("t2", i + 1, 1.0, u_remind=0, judge_model="indep") for i in range(4)]
    pairs, _ = pair_rows(reminded + unreminded, reminded_indep + unreminded_indep)

    by_u = analyze_judge_bias(pairs)["by_u_remind"]
    assert by_u["reminded"]["mean_diff_indep_minus_self"] == -0.5
    assert by_u["unreminded"]["mean_diff_indep_minus_self"] == 0.0
    assert by_u["difference_of_biases"]["mean_reminded_minus_unreminded"] == -0.5


def test_concentration_separates_a_few_bad_items_from_a_broad_bias():
    self_rows = [row(f"t{i}", 1, 1.0, attack_id=f"a{i}") for i in range(4)]
    indep_rows = [row(f"t{i}", 1, 1.0, attack_id=f"a{i}", judge_model="indep") for i in range(4)]
    indep_rows[0]["y_safety"] = 0.0  # one attack carries every disagreement
    pairs, _ = pair_rows(self_rows, indep_rows)

    c = analyze_judge_bias(pairs)["concentration"]
    assert c["n_disagreeing_rows"] == 1
    assert c["n_attacks_with_any_disagreement"] == 1
    assert c["n_attacks_total"] == 4
    assert c["max_single_attack_share"] == 1.0


def test_readout_quality_measures_ceiling_and_erosion_snr():
    rows = (
        [row("t1", 1, 1.0), row("t2", 1, 1.0)]
        + [row("t1", 2, 0.75), row("t2", 2, 0.25)]
    )
    q = readout_quality(rows)
    assert q["ceiling_share"] == 0.5
    assert q["n_distinct_levels"] == 3
    assert q["mean_by_turn"] == {1: 1.0, 2: 0.5}
    # turn-1 has zero within-turn spread, turn-2's sd is 0.3536; the pooled
    # within-turn sd is their mean, and the signal is the 0.5 drop.
    assert math.isclose(q["pooled_within_turn_sd"], 0.1767766952966369, rel_tol=1e-9)
    assert math.isclose(q["erosion_snr"], 0.5 / 0.1767766952966369, rel_tol=1e-9)


def test_readout_quality_on_a_fully_saturated_readout_has_no_snr():
    rows = [row("t1", 1, 1.0), row("t2", 1, 1.0), row("t1", 2, 1.0), row("t2", 2, 1.0)]
    q = readout_quality(rows)
    assert q["ceiling_share"] == 1.0
    assert q["n_distinct_levels"] == 1
    assert q["erosion_snr"] != q["erosion_snr"] or q["erosion_snr"] == 0.0
