from persona_drift.analysis_helpfulness import analyze_benign_screening, compare_arms_to_zero_control


def _row(trajectory_id, turn, category, y_help, u_remind=0, inserted_tokens=0, refusal=False):
    return {
        "trajectory_id": trajectory_id,
        "turn": turn,
        "category": category,
        "y_help": y_help,
        "y_safety": 1.0,
        "u_remind": u_remind,
        "inserted_tokens": inserted_tokens,
        "refusal_flag": refusal,
        "judge_parse_failure": False,
        "helpfulness_parse_failure": False,
    }


def test_analyze_benign_screening_computes_diagnostics_and_per_session_means():
    rows = [
        _row("s1", 1, "writing", 1.0),
        _row("s1", 2, "writing", 0.5, u_remind=1, inserted_tokens=10),
        _row("s2", 1, "coding", 0.75),
        _row("s2", 2, "coding", 1.0, u_remind=1, inserted_tokens=10),
    ]
    report = analyze_benign_screening(rows)
    diag = report["diagnostics"]
    assert diag["n_rows"] == 4
    assert diag["n_reminders_inserted"] == 2
    assert diag["total_inserted_tokens"] == 20
    assert diag["mean_y_help"] == (1.0 + 0.5 + 0.75 + 1.0) / 4
    assert report["per_session_mean_y_help"]["s1"] == (1.0 + 0.5) / 2
    assert report["per_session_mean_y_help"]["s2"] == (0.75 + 1.0) / 2


def test_analyze_benign_screening_handles_empty_rows():
    report = analyze_benign_screening([])
    assert report["diagnostics"]["n_rows"] == 0
    assert report["per_session_mean_y_help"] == {}


def test_compare_arms_to_zero_control_recovers_known_mean_diff():
    zero_control = {"s1": 0.9, "s2": 0.8, "s3": 0.7, "s4": 0.9}
    treatment = {"s1": 0.7, "s2": 0.6, "s3": 0.5, "s4": 0.7}  # uniformly 0.2 lower
    result = compare_arms_to_zero_control(
        {"zero_control": zero_control, "treatment": treatment}, zero_control_arm="zero_control"
    )
    assert "zero_control" not in result
    assert result["treatment"]["n_paired_sessions"] == 4
    assert abs(result["treatment"]["mean_diff"] - (-0.2)) < 1e-9
    assert result["treatment"]["p"] < 0.05
    assert result["treatment"]["t"] < 0


def test_compare_arms_to_zero_control_only_pairs_shared_session_ids():
    zero_control = {"s1": 1.0, "s2": 1.0}
    treatment = {"s1": 1.0, "s3": 0.5}  # s3 has no zero_control counterpart
    result = compare_arms_to_zero_control({"zero_control": zero_control, "treatment": treatment})
    assert result["treatment"]["n_paired_sessions"] == 1


def test_compare_arms_to_zero_control_nan_below_two_paired_sessions():
    zero_control = {"s1": 1.0}
    treatment = {"s1": 0.5}
    result = compare_arms_to_zero_control({"zero_control": zero_control, "treatment": treatment})
    assert result["treatment"]["n_paired_sessions"] == 1
    assert result["treatment"]["p"] != result["treatment"]["p"]  # nan
