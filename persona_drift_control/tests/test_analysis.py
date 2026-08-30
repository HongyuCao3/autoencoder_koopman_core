from persona_drift.analysis import analyze_screening


def _row(trajectory_id, turn, y_probe, u_remind, condition, system_prompt_id="character_traits_000"):
    return {
        "trajectory_id": trajectory_id,
        "turn": turn,
        "y_probe": y_probe,
        "y_probe_sd": 0.01,
        "u_remind": u_remind,
        "excitation_design": condition,
        "prompt_category": "character_traits",
        "system_prompt_id": system_prompt_id,
        "refusal_flag": False,
        "parse_failure": False,
    }


def test_q1_passes_on_a_clear_synthetic_drift():
    rows = [_row("t1", turn, 1.0 - 0.1 * (turn - 1), 0, "zero_control") for turn in range(1, 17)]
    report = analyze_screening(rows)
    assert report["q1_drift_exists"]["pass"] is True


def test_q1_fails_when_there_is_no_drift():
    rows = [_row("t1", turn, 0.9, 0, "zero_control") for turn in range(1, 17)]
    report = analyze_screening(rows)
    assert report["q1_drift_exists"]["pass"] is False


def test_q2_and_q3_use_only_excite_condition_and_detect_a_synthetic_effect():
    rows = []
    for turn in range(1, 17):
        u = 1 if turn % 2 == 0 else 0
        # y_probe boosted for two turns after a remind, so both the
        # next-turn and the +2-turn effect should be detectable.
        boost = 0.3 if (turn >= 2 and rows and rows[-1]["u_remind"] == 1) else 0.0
        rows.append(_row("t1", turn, 0.5 + boost, u, "iid"))
    report = analyze_screening(rows)
    assert report["q2_input_effective"]["n_pairs"] == 15
    assert report["q3_inertia"]["n_pairs"] == 14


def test_overall_pass_requires_all_three():
    rows = [_row("t1", turn, 0.9, 0, "zero_control") for turn in range(1, 17)]
    report = analyze_screening(rows)
    assert report["overall_pass"] is False


def test_q1_drift_trend_aggregates_by_prompt_not_by_trajectory():
    rows = []
    # two seeds of the SAME prompt, both drifting down -- should collapse to
    # one prompt-level slope, not count as two independent data points.
    for seed_tag in ("seed0", "seed1"):
        rows += [
            _row(f"t1_{seed_tag}", turn, 1.0 - 0.05 * turn, 0, "zero_control", system_prompt_id="character_traits_001")
            for turn in range(1, 17)
        ]
    # one prompt drifting up
    rows += [
        _row("t2", turn, 0.05 * turn, 0, "zero_control", system_prompt_id="character_traits_002")
        for turn in range(1, 17)
    ]
    report = analyze_screening(rows)
    trend = report["q1_drift_trend"]
    assert trend["n_prompts"] == 2
    assert trend["n_negative_slope_prompts"] == 1
    assert trend["n_positive_slope_prompts"] == 1
    assert set(trend["per_prompt_mean_slope"].keys()) == {"character_traits_001", "character_traits_002"}


def test_saturated_prompt_is_flagged_but_varying_prompt_is_not():
    saturated = [
        _row("t1", turn, 1.0, 0, "zero_control", system_prompt_id="character_traits_013")
        for turn in range(1, 17)
    ]
    varying = [
        _row("t2", turn, 0.5 + 0.01 * turn, 0, "zero_control", system_prompt_id="character_traits_006")
        for turn in range(1, 17)
    ]
    report = analyze_screening(saturated + varying)
    assert report["diagnostics"]["saturated_prompt_ids"] == ["character_traits_013"]
