from persona_drift.analysis_dose_response import analyze_dose_response


def _row(query_id, alpha, y_safety, refusal=False, parse_failure=False):
    return {
        "query_id": query_id,
        "alpha": alpha,
        "y_safety": y_safety,
        "refusal_flag": refusal,
        "judge_parse_failure": parse_failure,
    }


ALPHAS = (-1.0, -0.5, 0.0, 0.5, 1.0)


def test_detects_clean_positive_dose_response_as_passing():
    # 4 queries, each y_safety increasing with alpha (steering toward the
    # safe pole raises the safety score), slope magnitude varying slightly
    # per query so the t-test isn't degenerate across queries.
    rows = []
    for q in range(4):
        gain = 0.2 + 0.02 * q
        for alpha in ALPHAS:
            rows.append(_row(f"query{q}", alpha, y_safety=0.5 + gain * alpha))
    report = analyze_dose_response(rows)
    q2 = report["new_q2_dose_response"]
    assert q2["n_queries"] == 4
    assert q2["n_positive_slope_queries"] == 4
    assert q2["t_test_mean_slope_vs_zero"]["t"] > 0
    assert q2["pass"] is True


def test_flat_response_does_not_pass_new_q2():
    rows = []
    for q in range(4):
        for alpha in ALPHAS:
            rows.append(_row(f"query{q}", alpha, y_safety=0.5))
    report = analyze_dose_response(rows)
    assert report["new_q2_dose_response"]["pass"] is False


def test_negative_dose_response_does_not_pass_new_q2():
    # direction backwards from what its own sign predicts: y_safety falls as
    # alpha rises -- should fail even though the trend is not flat.
    rows = []
    for q in range(4):
        for alpha in ALPHAS:
            rows.append(_row(f"query{q}", alpha, y_safety=0.5 - 0.2 * alpha))
    report = analyze_dose_response(rows)
    q2 = report["new_q2_dose_response"]
    assert q2["t_test_mean_slope_vs_zero"]["t"] < 0
    assert q2["pass"] is False


def test_nan_y_safety_rows_drop_that_query_from_the_slope_fit():
    rows = [
        _row("q0", -1.0, y_safety=float("nan"), parse_failure=True),
        _row("q0", -0.5, y_safety=0.4),
        _row("q0", 0.0, y_safety=0.5),
        _row("q0", 0.5, y_safety=0.6),
        _row("q0", 1.0, y_safety=0.7),
    ]
    report = analyze_dose_response(rows)
    assert report["new_q2_dose_response"]["n_queries"] == 0


def test_diagnostics_summarize_refusal_parse_failure_and_per_alpha_means():
    rows = [
        _row("q0", -1.0, 0.2, refusal=True),
        _row("q0", 1.0, 0.8, refusal=False, parse_failure=True),
    ]
    report = analyze_dose_response(rows)
    diag = report["diagnostics"]
    assert diag["refusal_rate"] == 0.5
    assert diag["judge_parse_failure_rate"] == 0.5
    assert diag["y_safety_by_alpha"][-1.0]["mean"] == 0.2
    assert diag["y_safety_by_alpha"][1.0]["n"] == 1
