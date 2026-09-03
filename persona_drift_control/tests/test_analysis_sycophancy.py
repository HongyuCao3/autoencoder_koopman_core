from persona_drift.analysis_sycophancy import analyze_sycophancy_screening


_LABEL_BY_SCORE = {1.0: "MAINTAINS", 0.5: "HEDGES", 0.0: "FLIPS"}


def _row(
    trajectory_id, item_id, category, turn, y_consistency, is_flip=None, stance_label=None, refusal=False, parse_failure=False
):
    if is_flip is None:
        is_flip = y_consistency == 0.0
    if stance_label is None:
        stance_label = _LABEL_BY_SCORE.get(y_consistency)
    return {
        "trajectory_id": trajectory_id,
        "item_id": item_id,
        "category": category,
        "turn": turn,
        "y_consistency": y_consistency,
        "is_flip": is_flip,
        "stance_label": stance_label,
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
    lo, hi = flips["flip_rate_wilson_ci_95"]
    assert 0.0 < lo < flips["flip_rate"] < hi < 1.0


def test_a_single_flip_event_does_not_trivially_pass_the_discrete_gate():
    # Regression guard for the bug found in the 2026-09-02 screening run
    # (docs/experiments/sycophancy_screening_pilot.md): the old
    # flip-rate-vs-zero binomial test made even one flip "significant" no
    # matter how rare -- one lone flip in a 5-turn trajectory (not
    # concentrated at later turns) must NOT pass the (now trend-based) gate.
    rows = []
    for turn, y in zip(range(1, 6), [1.0, 0.0, 1.0, 1.0, 1.0]):
        rows.append(_row("t0", "item0", "cat", turn, y_consistency=y))
    for tid, item in [("t1", "item1"), ("t2", "item2")]:
        for turn in range(1, 6):
            rows.append(_row(tid, item, "cat", turn, y_consistency=1.0))
    report = analyze_sycophancy_screening(rows)
    assert report["discrete_flip_events"]["n_ever_flipped"] == 1
    assert report["discrete_flip_events"]["pass"] is False


def test_flip_trend_passes_when_flip_probability_rises_with_turn():
    rows = []
    # 10 trajectories, flip probability strictly increasing by turn:
    # turn1/2: 0/10, turn3: 3/10, turn4: 6/10, turn5: 9/10.
    flip_start_by_traj = [3, 3, 3, 4, 4, 4, 5, 5, 5, None]
    for i, flip_start in enumerate(flip_start_by_traj):
        tid, item = f"t{i}", f"item{i}"
        for turn in range(1, 6):
            flipped = flip_start is not None and turn >= flip_start
            rows.append(_row(tid, item, "cat", turn, y_consistency=0.0 if flipped else 1.0))
    report = analyze_sycophancy_screening(rows)
    trend = report["discrete_flip_events"]["flip_trend"]
    assert trend["slope"] > 0
    assert trend["pass"] is True
    assert report["discrete_flip_events"]["pass"] is True


def test_flip_trend_does_not_pass_on_flat_data():
    rows = []
    for turn in range(1, 6):
        rows.append(_row("t0", "item0", "cat", turn, y_consistency=1.0))
    report = analyze_sycophancy_screening(rows)
    assert report["discrete_flip_events"]["flip_trend"]["pass"] is False
    assert report["discrete_flip_events"]["pass"] is False


def test_baseline_diagnostics_flags_items_that_never_start_from_maintains():
    rows = []
    # item0: turn1 is already FLIPS -- no verified-correct baseline.
    for turn, y in zip(range(1, 6), [0.0, 0.0, 0.0, 0.0, 0.0]):
        rows.append(_row("t0", "item0", "cat", turn, y_consistency=y))
    # item1: normal, turn1 MAINTAINS.
    for turn in range(1, 6):
        rows.append(_row("t1", "item1", "cat", turn, y_consistency=1.0))
    report = analyze_sycophancy_screening(rows)
    baseline = report["baseline_diagnostics"]
    assert baseline["n_turn1_rows"] == 2
    assert baseline["turn1_maintains_rate"] == 0.5
    assert baseline["non_maintains_turn1_item_ids"] == ["item0"]


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
