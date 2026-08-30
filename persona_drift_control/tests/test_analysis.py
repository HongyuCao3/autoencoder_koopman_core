from persona_drift.analysis import analyze_screening


def _row(trajectory_id, turn, y_probe, u_remind, condition):
    return {
        "trajectory_id": trajectory_id,
        "turn": turn,
        "y_probe": y_probe,
        "y_probe_sd": 0.01,
        "u_remind": u_remind,
        "excitation_design": condition,
        "prompt_category": "character_traits",
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
