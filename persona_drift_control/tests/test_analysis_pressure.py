from persona_drift.analysis_pressure import analyze_pressure_screening


def _row(trajectory_id, turn, y_probe, user_mode, system_prompt_id="character_traits_000"):
    return {
        "trajectory_id": trajectory_id,
        "turn": turn,
        "y_probe": y_probe,
        "user_mode": user_mode,
        "prompt_category": "character_traits",
        "system_prompt_id": system_prompt_id,
        "refusal_flag": False,
        "parse_failure": False,
    }


def _flat_baseline(trajectory_id, system_prompt_id, value=0.9, n_turns=16):
    return [_row(trajectory_id, turn, value, "live", system_prompt_id) for turn in range(1, n_turns + 1)]


def _eroding_pressure(trajectory_id, system_prompt_id, slope, n_turns=16):
    return [
        _row(trajectory_id, turn, 1.0 + slope * (turn - 1), "pressure", system_prompt_id)
        for turn in range(1, n_turns + 1)
    ]


# 4 independent "prompts" with similar-but-not-identical negative slopes: a
# two-point sample (t-test df=1) needs an implausibly large |t| to ever reach
# p<0.05, so this uses enough independent prompts for the one-sample t-test
# to have real power -- the same reason the real pilot uses >=4 prompts
# per condition rather than 2 (see drift_confirmation_pilot.md's own
# 3-prompt-vs-10-prompt power lesson).
_PRESSURE_SLOPES = {"character_traits_000": -0.04, "character_traits_001": -0.05, "character_traits_002": -0.06, "character_traits_003": -0.07}


def _all_eroding_pressure_rows():
    rows = []
    for i, (prompt_id, slope) in enumerate(_PRESSURE_SLOPES.items()):
        rows += _eroding_pressure(f"p{i}", prompt_id, slope)
    return rows


def _all_flat_baseline_rows():
    rows = []
    for i, prompt_id in enumerate(_PRESSURE_SLOPES):
        rows += _flat_baseline(f"b{i}", prompt_id)
    return rows


def test_pressure_condition_detects_a_clear_synthetic_erosion():
    report = analyze_pressure_screening(_all_eroding_pressure_rows())
    assert report["q1_escalating_pressure"]["pass"] is True
    assert report["q1_escalating_pressure"]["n_negative_slope_prompts"] == 4


def test_baseline_condition_stays_flat_and_fails():
    report = analyze_pressure_screening(_all_flat_baseline_rows())
    assert report["q1_baseline_no_pressure"]["pass"] is False


def test_user_mode_splits_baseline_and_pressure_rows_independently():
    rows = _all_flat_baseline_rows() + _all_eroding_pressure_rows()
    report = analyze_pressure_screening(rows)
    assert report["q1_baseline_no_pressure"]["n_prompts"] == 4
    assert report["q1_escalating_pressure"]["n_prompts"] == 4
    assert report["q1_baseline_no_pressure"]["pass"] is False
    assert report["q1_escalating_pressure"]["pass"] is True
