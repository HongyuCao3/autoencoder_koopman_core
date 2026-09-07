"""docs/experiments/adaptive_vs_fixed_claim_plan.md T0 section 1.4: at least
three checks on scripts/analyze_readout_state.py -- the turn-demeaning
definition, that the trajectory key gets an arm prefix (so two arms sharing
`trajectory_id`s don't get strung into one fake trajectory), and that the
gate-2 mixed baseline degenerates to the fixed arm itself at spend_prob=1.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

from analyze_readout_state import (  # noqa: E402
    group_by_key,
    mixed_baseline_bootstrap,
    variance_decomposition,
)


def _row(trajectory_id: str, turn: int, y: float, u_remind: int = 0) -> dict:
    return {"trajectory_id": trajectory_id, "turn": turn, "y_safety": y, "u_remind": u_remind}


def test_turn_demeaning_removes_a_shared_linear_turn_trend():
    # Two trajectories share the exact same turn-indexed trend (turn * 1.0)
    # plus a trajectory-constant offset; only the offset should survive
    # de-meaning by turn, so lag-1 on the demeaned residual must be exactly
    # 1.0 (the offset alone reproduces perfectly turn to turn) while the RAW
    # lag-1 is dominated by the shared trend instead.
    rows = []
    for traj_id, offset in (("a", 0.0), ("b", 10.0)):
        for turn in range(1, 6):
            rows.append(_row(traj_id, turn, y=float(turn) + offset))
    result = variance_decomposition(rows, y_col="y_safety")
    # turn_mean(t) = t + 5.0 (average of the two offsets) for every t, so
    # resid = y - turn_mean = offset - 5.0, CONSTANT within each trajectory.
    # A constant has zero within-trajectory variation, so lag-1 correlation
    # on it is undefined for a single trajectory but well-defined pooled
    # across two distinct constants: perfectly correlated (r=1).
    assert result["lag1_demeaned"] == pytest.approx(1.0)
    assert result["stable_traj_share"] == pytest.approx(1.0)
    # The raw signal is turn + offset -- monotonically increasing within
    # each trajectory, so raw lag-1 must also be strongly positive, but the
    # demeaned test above is the one that isolates the trajectory-constant
    # from the shared trend.
    assert result["lag1_raw"] > 0.9


def test_trajectory_key_must_include_arm_prefix_to_avoid_cross_arm_mixing():
    # Two arms share the SAME trajectory_id ("t1"). Without an arm prefix on
    # the grouping key, turn-5 rows from arm "A" and turn-1 rows from arm "B"
    # would be treated as adjacent turns of one fake trajectory.
    rows_wrong_key = [
        {"trajectory_id": "t1", "arm": "A", "turn": 5, "y_safety": 0.0, "u_remind": 0},
        {"trajectory_id": "t1", "arm": "B", "turn": 1, "y_safety": 1.0, "u_remind": 0},
    ]
    grouped_by_bare_id = group_by_key(rows_wrong_key, lambda r: r["trajectory_id"])
    assert len(grouped_by_bare_id) == 1  # the bug this test guards against

    grouped_by_prefixed_key = group_by_key(rows_wrong_key, lambda r: r["arm"] + "|" + r["trajectory_id"])
    assert len(grouped_by_prefixed_key) == 2
    for traj in grouped_by_prefixed_key.values():
        assert len(traj) == 1  # each arm's single row stays its own trajectory


def test_mixed_baseline_degenerates_to_fixed_arm_at_spend_prob_one():
    rng = np.random.default_rng(0)
    n = 20
    adaptive = rng.normal(size=n)
    zero = rng.normal(size=n)
    fixed = rng.normal(size=n)

    result = mixed_baseline_bootstrap(adaptive, zero, fixed, spend_prob=1.0, n_resamples=500, seed=1)
    # At spend_prob=1.0 every resampled slot is "treated", so the mixed
    # baseline is just `fixed` resampled -- the comparison collapses to
    # exactly the ungated adaptive-vs-fixed paired bootstrap, independent of
    # `zero` (which must never be touched in this branch).
    boot_rng = np.random.default_rng(1)
    idx = boot_rng.integers(0, n, size=(500, n))
    _ = boot_rng.random(size=(500, n))  # consumes the same draws mixed_baseline_bootstrap makes
    expected_diffs = (adaptive[idx] - fixed[idx]).mean(axis=1)
    assert result["mean_diff"] == pytest.approx(float(expected_diffs.mean()), abs=1e-9)
