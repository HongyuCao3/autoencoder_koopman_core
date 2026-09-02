from persona_drift.analysis_sycophancy import analyze_sycophancy_screening


def _row(trajectory_id, item_id, category, turn, y_consistency, is_flip=None, refusal=False, parse_failure=False):
    if is_flip is None:
        is_flip = y_consistency == 0.0
    return {
        "trajectory_id": trajectory_id,
        "item_id": item_id,
        "category": category,
        "turn": turn,
        "y_consistency": y_consistency,
        "is_flip": is_flip,
        "refusal_flag": refusal,
        "judge_parse_failure": parse_failure,
    }


def test_detects_clean_capitulation_as_negative_slope():
    # 4 items, each with a monotonic decline in y_consistency, slope
    # magnitude varying slightly per item so the t-test isn't degenerate.
    rows = []
    for a in range(4):
        decline = 0.2 + 0.02 * a
        for seed in (0, 1):
            tid = f"item{a}__seed{seed}"
            for turn in range(1, 6):
                rows.append(_row(tid, f"item{a}", "cat", turn, y_consistency=1.0 - decline * (turn - 1)))
    report = analyze_sycophancy_screening(rows)
    q1 = report["new_q1_escalation"]
    assert q1["n_items"] == 4
    assert q1["n_negative_slope_items"] == 4
    assert q1["t_test_mean_slope_vs_zero"]["t"] < 0
    assert q1["pass"] is True


def test_flat_trajectories_do_not_pass_new_q1():
    rows = []
    for a in range(4):
        for turn in range(1, 6):
            rows.append(_row(f"item{a}__seed0", f"item{a}", "cat", turn, y_consistency=1.0))
    report = analyze_sycophancy_screening(rows)
    assert report["new_q1_escalation"]["pass"] is False


def test_autocorrelation_detects_perfectly_correlated_lag1():
    rows = []
    for turn, y in zip(range(1, 6), [1.0, 0.8, 0.5, 0.3, 0.0]):
        rows.append(_row("t0", "item0", "cat", turn, y_consistency=y))
    for turn, y in zip(range(1, 6), [1.0, 1.0, 0.8, 0.5, 0.3]):
        rows.append(_row("t1", "item1", "cat", turn, y_consistency=y))
    report = analyze_sycophancy_screening(rows)
    q3 = report["new_q3_autocorrelation"]
    assert q3["n_pairs"] == 8
    assert q3["slope"] > 0
    assert q3["pass"] is True


def test_discrete_flip_events_records_first_flip_turn_and_rate():
    rows = []
    # item0: flips at turn 3, stays flipped
    for turn, y in zip(range(1, 6), [1.0, 1.0, 0.0, 0.0, 0.0]):
        rows.append(_row("t0", "item0", "cat", turn, y_consistency=y))
    # item1: never flips
    for turn in range(1, 6):
        rows.append(_row("t1", "item1", "cat", turn, y_consistency=1.0))
    report = analyze_sycophancy_screening(rows)
    flips = report["discrete_flip_events"]
    assert flips["turn_of_flip"]["t0"] == 3
    assert flips["turn_of_flip"]["t1"] is None
    assert flips["number_of_flips"]["t0"] == 3
    assert flips["number_of_flips"]["t1"] == 0
    assert flips["n_trajectories"] == 2
    assert flips["n_ever_flipped"] == 1
    assert flips["flip_rate"] == 0.5


def test_discrete_flip_events_pass_requires_at_least_one_flip():
    rows = []
    for turn in range(1, 6):
        rows.append(_row("t0", "item0", "cat", turn, y_consistency=1.0))
    report = analyze_sycophancy_screening(rows)
    assert report["discrete_flip_events"]["pass"] is False


def test_diagnostics_summarize_refusal_and_parse_failure_rates():
    rows = [
        _row("t0", "item0", "false_presupposition", 1, 0.5, refusal=True),
        _row("t0", "item0", "false_presupposition", 2, 0.5, refusal=False, parse_failure=True),
    ]
    report = analyze_sycophancy_screening(rows)
    diag = report["diagnostics"]
    assert diag["refusal_rate"] == 0.5
    assert diag["judge_parse_failure_rate"] == 0.5
    assert diag["y_consistency_by_category"]["false_presupposition"]["mean"] == 0.5
    assert diag["y_consistency_by_turn"][1]["n"] == 1
