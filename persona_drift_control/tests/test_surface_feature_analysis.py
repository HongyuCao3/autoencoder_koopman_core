import pandas as pd

from persona_drift.surface_feature_analysis import analyze_input_effect, analyze_zero_control_drift


def _drift_row(trajectory_id, turn, feature_val, system_prompt_id="p1", condition="zero_control"):
    return {
        "trajectory_id": trajectory_id,
        "turn": turn,
        "system_prompt_id": system_prompt_id,
        "excitation_design": condition,
        "some_feature": feature_val,
    }


def test_analyze_zero_control_drift_detects_a_synthetic_drift():
    rows = [_drift_row("t1", turn, 1.0 - 0.1 * (turn - 1)) for turn in range(1, 17)]
    rows += [_drift_row("t2", turn, 1.0 - 0.1 * (turn - 1), system_prompt_id="p2") for turn in range(1, 17)]
    df = pd.DataFrame(rows)
    report = analyze_zero_control_drift(df, "some_feature")
    assert report["n_prompts"] == 2
    assert report["n_negative_slope_prompts"] == 2
    assert report["t_test_mean_slope_vs_zero"]["p"] < 0.05


def test_analyze_zero_control_drift_no_effect_when_flat():
    rows = [_drift_row("t1", turn, 0.5) for turn in range(1, 17)]
    rows += [_drift_row("t2", turn, 0.5, system_prompt_id="p2") for turn in range(1, 17)]
    df = pd.DataFrame(rows)
    report = analyze_zero_control_drift(df, "some_feature")
    assert report["mean_drop_turn1_to_last"] == 0.0


def _input_row(trajectory_id, turn, feature_val, u_remind, condition="iid"):
    return {
        "trajectory_id": trajectory_id,
        "turn": turn,
        "u_remind": u_remind,
        "excitation_design": condition,
        "some_feature": feature_val,
    }


def test_analyze_input_effect_detects_a_synthetic_next_turn_and_lag2_effect():
    rows = []
    for turn in range(1, 17):
        u = 1 if turn % 2 == 0 else 0
        boost = 0.3 if (turn >= 2 and rows and rows[-1]["u_remind"] == 1) else 0.0
        rows.append(_input_row("t1", turn, 0.5 + boost, u))
    df = pd.DataFrame(rows)
    report = analyze_input_effect(df, "some_feature")
    assert report["q2_next_turn"]["n_pairs"] == 15
    assert report["q3_lag2_inertia"]["n_pairs"] == 14
    assert report["q2_next_turn"]["diff"] > 0


def test_analyze_input_effect_no_effect_when_feature_is_constant():
    rows = [_input_row("t1", turn, 0.5, turn % 2) for turn in range(1, 17)]
    df = pd.DataFrame(rows)
    report = analyze_input_effect(df, "some_feature")
    assert report["q2_next_turn"]["diff"] == 0.0


def test_analyze_input_effect_only_uses_iid_rows():
    iid_rows = [_input_row("t1", turn, 0.5, turn % 2, "iid") for turn in range(1, 17)]
    zero_rows = [_input_row("t2", turn, 999.0, 0, "zero_control") for turn in range(1, 17)]
    df = pd.DataFrame(iid_rows + zero_rows)
    report = analyze_input_effect(df, "some_feature")
    assert report["q2_next_turn"]["mean_given_u0"] < 900
