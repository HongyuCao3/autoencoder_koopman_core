from persona_drift.analysis_adversarial import analyze_adversarial_screening


def _row(trajectory_id, attack_id, category, turn, y_safety, refusal=False, parse_failure=False):
    return {
        "trajectory_id": trajectory_id,
        "attack_id": attack_id,
        "category": category,
        "turn": turn,
        "y_safety": y_safety,
        "refusal_flag": refusal,
        "judge_parse_failure": parse_failure,
    }


def test_detects_clean_escalation_as_negative_slope():
    # 4 attacks, each with a monotonic decline (safety eroding), slope
    # magnitude varying slightly per attack so the t-test isn't degenerate
    # (zero-variance) across attacks.
    rows = []
    for a in range(4):
        decline = 0.15 + 0.02 * a
        for seed in (0, 1):
            tid = f"attack{a}__seed{seed}"
            for turn in range(1, 6):
                rows.append(_row(tid, f"attack{a}", "cat", turn, y_safety=1.0 - decline * (turn - 1)))
    report = analyze_adversarial_screening(rows)
    q1 = report["new_q1_escalation"]
    assert q1["n_attacks"] == 4
    assert q1["n_negative_slope_attacks"] == 4
    assert q1["t_test_mean_slope_vs_zero"]["t"] < 0
    assert q1["pass"] is True


def test_flat_trajectories_do_not_pass_new_q1():
    rows = []
    for a in range(4):
        for turn in range(1, 6):
            rows.append(_row(f"attack{a}__seed0", f"attack{a}", "cat", turn, y_safety=0.5))
    report = analyze_adversarial_screening(rows)
    assert report["new_q1_escalation"]["pass"] is False


def test_autocorrelation_detects_perfectly_correlated_lag1():
    rows = []
    for turn, y in zip(range(1, 6), [1.0, 0.8, 0.6, 0.4, 0.2]):
        rows.append(_row("t0", "a0", "cat", turn, y_safety=y))
    for turn, y in zip(range(1, 6), [0.9, 0.7, 0.5, 0.3, 0.1]):
        rows.append(_row("t1", "a1", "cat", turn, y_safety=y))
    report = analyze_adversarial_screening(rows)
    q3 = report["new_q3_autocorrelation"]
    assert q3["n_pairs"] == 8
    assert q3["slope"] > 0
    assert q3["pass"] is True


def test_nan_y_safety_rows_are_dropped_from_autocorrelation():
    rows = [
        _row("t0", "a0", "cat", 1, y_safety=float("nan"), parse_failure=True),
        _row("t0", "a0", "cat", 2, y_safety=0.5),
        _row("t0", "a0", "cat", 3, y_safety=0.6),
    ]
    report = analyze_adversarial_screening(rows)
    assert report["new_q3_autocorrelation"]["n_pairs"] == 1


def test_diagnostics_summarize_refusal_and_parse_failure_rates():
    rows = [
        _row("t0", "a0", "chem", 1, 0.5, refusal=True),
        _row("t0", "a0", "chem", 2, 0.5, refusal=False, parse_failure=True),
    ]
    report = analyze_adversarial_screening(rows)
    diag = report["diagnostics"]
    assert diag["refusal_rate"] == 0.5
    assert diag["judge_parse_failure_rate"] == 0.5
    assert diag["y_safety_by_category"]["chem"]["mean"] == 0.5
    assert diag["y_safety_by_turn"][1]["n"] == 1
