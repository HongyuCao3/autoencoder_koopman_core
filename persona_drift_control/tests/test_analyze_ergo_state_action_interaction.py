"""docs/experiments/two_task_success_plan.md section 2, E3 Step 3: unit
tests for scripts/analyze_ergo_state_action_interaction.py --

1. the regression recovers a known `b2` (interaction coefficient) within
   its reported bootstrap CI on synthetic data generated from that exact
   coefficient;
2. `u_{t+1}` alignment is correct -- on synthetic data where the true
   generating process depends on `u_{t+1}` (`b`'s own `u_reset`) and is
   perfectly ANTI-correlated with `u_t` (`a`'s own `u_reset`), the
   recovered `b0` has the sign of the true (`u_{t+1}`-aligned) effect, not
   the flipped sign a `u_t`-misaligned implementation would produce;
3. transitions with missing/NaN `closeness` on either side of the pair are
   dropped (counted, not silently included) and do not crash the fit;
4. non-consecutive turn pairs (a gap in `turn`) are dropped and counted
   separately from the missing-closeness count.
"""

from __future__ import annotations

import math
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

from analyze_ergo_state_action_interaction import (  # noqa: E402
    _bootstrap_by_item,
    _design_matrix,
    _ols_beta,
    build_transitions,
    fit_interaction_model,
)


def _row(item_id: str, trajectory_id: str, turn: int, num_shards: int, closeness: float, u_reset: int) -> dict:
    return {
        "item_id": item_id,
        "trajectory_id": trajectory_id,
        "turn": turn,
        "num_shards": num_shards,
        "closeness": closeness,
        "u_reset": u_reset,
        "shard_frac": turn / num_shards,
    }


def test_b2_recovery_within_ci():
    """Generate c_{t+1} = a*c_t + b0*u_next + b2*u_next*c_t + g*shard_frac_next + const
    (small deterministic per-item jitter, no random noise needed since this
    is testing whether the OLS+bootstrap machinery recovers a *known*
    coefficient, not robustness to noise) and check b2 lands inside its
    reported 95% CI."""
    true_a, true_b0, true_b2, true_g, true_const = 0.5, 0.2, -0.6, 0.1, 0.15
    num_shards = 6
    rng = np.random.default_rng(42)

    rows: list[dict] = []
    n_items = 30
    for item_idx in range(n_items):
        item_id = f"item{item_idx}"
        traj_id = f"traj{item_idx}"
        c = 0.3 + 0.4 * (item_idx % 3) / 2.0  # spread starting closeness across items
        u_seq = rng.integers(0, 2, size=num_shards).tolist()
        rows.append(_row(item_id, traj_id, 1, num_shards, c, u_seq[0]))
        for t in range(1, num_shards):
            u_next = u_seq[t]
            shard_frac_next = (t + 1) / num_shards
            c_next = true_a * c + true_b0 * u_next + true_b2 * u_next * c + true_g * shard_frac_next + true_const
            rows.append(_row(item_id, traj_id, t + 1, num_shards, c_next, u_next))
            c = c_next

    transitions, drop_info = build_transitions(rows)
    assert drop_info["n_dropped_turn_gap"] == 0
    assert drop_info["n_dropped_missing_closeness"] == 0
    assert len(transitions) == n_items * (num_shards - 1)

    fit = fit_interaction_model(transitions, n_bootstrap=500, bootstrap_seed=0)
    b2 = fit["coefficients"]["b2"]
    assert b2["ci_low"] <= true_b2 <= b2["ci_high"], f"true b2={true_b2} not in CI {b2}"
    assert abs(b2["point_estimate"] - true_b2) < 0.05


def test_u_next_alignment_not_u_t():
    """Each synthetic item is a single (a, b) turn pair with `u_t` (a's own
    `u_reset`) and `u_next` (b's own `u_reset`) set to PERFECT OPPOSITES
    (`u_next = 1 - u_t`). Ground truth: `c_t1 = const + b0*u_next` (a =
    b2 = g = 0, exactly, no noise) -- `c_t` is drawn independently at
    random and has no causal effect, and `shard_frac_next` varies (via
    randomized `num_shards`/`turn`) so it does not collide with the
    intercept column.

    A correctly u_next-aligned fit recovers `b0 > 0` matching the true
    value. An implementation that mistakenly used `u_t` (e.g. via the F0
    off-by-one that reads a third, further-ahead row) would instead be
    regressing against `u_next`'s exact complement and would recover a
    NEGATIVE coefficient of the same magnitude on whichever u-column it
    used -- this test's sign+magnitude check catches that misalignment.
    """
    true_b0, true_const = 0.3, 0.4
    rng = np.random.default_rng(7)
    rows: list[dict] = []
    n_pairs = 200
    for i in range(n_pairs):
        item_id = f"item{i}"
        traj_id = f"traj{i}"
        num_shards = int(rng.integers(2, 9))
        turn_a = int(rng.integers(1, num_shards))  # 1 .. num_shards-1
        turn_b = turn_a + 1
        c_t = float(rng.uniform(0.1, 0.9))
        u_t = int(rng.integers(0, 2))
        u_next = 1 - u_t
        c_t1 = true_const + true_b0 * u_next
        rows.append(_row(item_id, traj_id, turn_a, num_shards, c_t, u_t))
        rows.append(_row(item_id, traj_id, turn_b, num_shards, c_t1, u_next))

    transitions, drop_info = build_transitions(rows)
    assert drop_info["n_dropped_turn_gap"] == 0
    assert drop_info["n_dropped_missing_closeness"] == 0
    assert len(transitions) == n_pairs

    fit = fit_interaction_model(transitions, n_bootstrap=200, bootstrap_seed=1)
    b0 = fit["coefficients"]["b0"]
    assert b0["point_estimate"] > 0, f"expected positive b0 (u_next-aligned), got {b0}"
    assert abs(b0["point_estimate"] - true_b0) < 1e-6
    const = fit["coefficients"]["const"]
    assert abs(const["point_estimate"] - true_const) < 1e-6
    a_coef = fit["coefficients"]["a"]
    b2_coef = fit["coefficients"]["b2"]
    g_coef = fit["coefficients"]["g"]
    assert abs(a_coef["point_estimate"]) < 1e-6
    assert abs(b2_coef["point_estimate"]) < 1e-6
    assert abs(g_coef["point_estimate"]) < 1e-6


def test_missing_and_nan_closeness_dropped():
    rows = [
        _row("itemA", "trajA", 1, 4, 0.2, 0),
        _row("itemA", "trajA", 2, 4, 0.4, 1),
        _row("itemA", "trajA", 3, 4, float("nan"), 0),  # NaN closeness -> transition (turn2,turn3) dropped
        _row("itemA", "trajA", 4, 4, 0.6, 1),  # transition (turn3, turn4) also dropped (turn3 NaN on the "a" side)
        _row("itemB", "trajB", 1, 3, 0.1, 0),
        _row("itemB", "trajB", 2, 3, None, 1),  # None closeness
        _row("itemB", "trajB", 3, 3, 0.5, 0),
    ]
    transitions, drop_info = build_transitions(rows)
    # itemA: (t1,t2) survives, (t2,t3) and (t3,t4) dropped for NaN.
    # itemB: (t1,t2) dropped for None, (t2,t3) dropped for None on the "a" side.
    assert drop_info["n_dropped_missing_closeness"] == 4
    assert drop_info["n_dropped_turn_gap"] == 0
    assert len(transitions) == 1
    assert transitions[0]["item_id"] == "itemA"
    assert transitions[0]["c_t"] == 0.2
    assert transitions[0]["c_t1"] == 0.4

    # Fit must not crash on the surviving single transition (degenerate but well-defined).
    X, y = _design_matrix(transitions)
    beta = _ols_beta(X, y)
    assert beta.shape == (5,)
    assert not np.any(np.isnan(beta))


def test_turn_gap_dropped_and_counted_separately():
    rows = [
        _row("itemC", "trajC", 1, 5, 0.1, 0),
        _row("itemC", "trajC", 2, 5, 0.3, 1),
        # turn 3 missing entirely -> (turn2, turn4) pair has a gap, dropped.
        _row("itemC", "trajC", 4, 5, 0.5, 0),
        _row("itemC", "trajC", 5, 5, 0.7, 1),
    ]
    transitions, drop_info = build_transitions(rows)
    assert drop_info["n_dropped_turn_gap"] == 1
    assert drop_info["n_dropped_missing_closeness"] == 0
    # surviving: (turn1,turn2) and (turn4,turn5)
    assert len(transitions) == 2


def test_bootstrap_by_item_resamples_at_item_granularity():
    """Sanity check on the bootstrap helper itself: with a single item, every
    resample must reproduce exactly that item's transitions (no cross-item
    mixing), so the point estimate equals every bootstrap draw."""
    rows = [
        _row("only_item", "traj0", 1, 4, 0.2, 0),
        _row("only_item", "traj0", 2, 4, 0.35, 1),
        _row("only_item", "traj0", 3, 4, 0.4, 0),
        _row("only_item", "traj0", 4, 4, 0.5, 1),
    ]
    transitions, _ = build_transitions(rows)
    X, y = _design_matrix(transitions)
    point_beta = _ols_beta(X, y)
    boot_betas = _bootstrap_by_item(transitions, n_resamples=10, seed=0)
    for i in range(10):
        assert np.allclose(boot_betas[i], point_beta)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
