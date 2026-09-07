#!/usr/bin/env python3
"""E0 / F0 (docs/experiments/signal_resolution_plan.md, superseding
docs/experiments/backup/readout_controllability_gate_plan.md section 4): the
readout-controllability gate (RC-0/RC-1/RC-2/RC-3) for the ERGO/Laban line,
fixing signal_resolution_plan.md section 0's offline audit numbers into a
re-runnable script. Answers whether `y_task_success` carries feedback-usable,
controllable state before any `KoopmanMPCController` (Phase C) work is
allowed to start.

**2026-09-07 F0 fix**: the original version of this script had two
measurement bugs that had (incorrectly) terminated the ERGO line --
signal_resolution_plan.md section 0.1/0.2 has the full derivation:

1. Segment 3's `u_{t+1}` was actually `u_{t+2}` (an off-by-one in which row
   of a 3-row zip supplied it) -- fixed in `build_input_gain_transitions`.
2. Segment 1's ARX rollout let the model free-run the `shard_frac` state
   dimension, which is deterministic and fully known in advance -- an
   unfair comparison against null models that get the true value at every
   row. Fixed by `rollout_output_error_aux_truth` (truth-overrides the aux
   dimension after every step) plus a 20-item-split protocol (a single
   split's point estimate is unreliable, section 0.2).

Four segments (RC-0 is trivial for this readout, checked in `main`), all
offline / CPU-only, on outputs/ergo_math_phaseB_random_excite/trajectories.jsonl
(666 rows, 60 items) and that directory's koopman_fit_report.json:

1. RC-1: 20-split held-out MSE of the fitted ARX surrogate (aux dimension
   truth-overridden during rollout) against three trivial nulls (constant,
   per-turn mean, stateless OLS on [1, shard_frac, u_reset]), pass if it
   beats the best null in >=18/20 splits.
2. RC-2: de-trended lag-1 autocorrelation of y_task_success, trend removed
   by shard_frac decile bin (not turn -- trajectories have different
   lengths, so pooling by raw turn would mix short and long items).
3. RC-3: input-gain OLS y_{t+1} ~ 1 + y_t + u_t + u_{t+1} + shard_frac +
   item fixed effects (one-hot, one item dropped), with and without
   u_{t+1} (`u_{t+1}` is `b`'s own u_reset, the action on the SAME row as
   the predicted target -- see the F0 fix note above).
4. Registers that `richer_abs_sign` is vacuous for this domain's binary y
   (|y| == y, sign(y) == y for y in {0, 1}).

Every number here is checked against docs/experiments/signal_resolution_plan.md
section 0 and the RC-B pre-registered verdict (section 1.4) -- that section
IS this script's spec.
"""

from __future__ import annotations

import json
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

from persona_drift.modeling.dataset import (  # noqa: E402
    ReducedStateConfig,
    build_identification_dataset,
    build_reduced_state_pairs,
    group_by_trajectory,
    load_trajectories,
)
from persona_drift.modeling.evaluate import rollout_output_error  # noqa: E402
from persona_drift.modeling.koopman import KoopmanSurrogate, no_extra_features  # noqa: E402

ROWS_PATH = pathlib.Path("outputs/ergo_math_phaseB_random_excite/trajectories.jsonl")
FIT_REPORT_PATH = pathlib.Path("outputs/ergo_math_phaseB_random_excite/koopman_fit_report.json")
OUT_PATH = pathlib.Path("outputs/ergo_math_phaseB_random_excite/readout_state_report.json")

Y_COL = "y_task_success"
U_COL = "u_reset"
N_SPLITS = 20
N_HELD_OUT_ITEMS = 15
RIDGE = 1e-6
STATE_CONFIG = ReducedStateConfig(nu=1, mu=1, aux_cols=("shard_frac",), contemporaneous_v=True)


def _add_shard_frac(rows: list[dict]) -> list[dict]:
    for row in rows:
        row["shard_frac"] = row["turn"] / row["num_shards"]
    return rows


def group_by_key(rows: list[dict], key_fn) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(key_fn(row), []).append(row)
    for traj_rows in groups.values():
        traj_rows.sort(key=lambda r: r["turn"])
    return groups


def ols(X: np.ndarray, y: np.ndarray) -> dict:
    """Plain OLS, classical (homoskedastic) standard errors -- mirrors
    scripts/analyze_readout_state.py's `ols` exactly (no statsmodels here)."""

    n, k = X.shape
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    rss = float(np.sum(resid**2))
    tss = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - rss / tss
    dof = n - k
    sigma2 = rss / dof
    xtx_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(sigma2 * xtx_inv))
    t = beta / se
    p = 2.0 * stats.t.sf(np.abs(t), dof)
    return {"beta": beta.tolist(), "se": se.tolist(), "t": t.tolist(), "p": p.tolist(), "r2": r2, "n": n, "dof": dof}


# ---------------------------------------------------------------------------
# Segment 1: RC-1, 20-split held-out MSE vs. three trivial nulls, ARX rollout
# reported both naive and with the aux (shard_frac) dimension truth-overridden.
# ---------------------------------------------------------------------------


def rollout_output_error_aux_truth(predictor, rows: list[dict], config: ReducedStateConfig) -> float:
    """Same multi-step rollout as modeling.evaluate.rollout_output_error
    (not modified -- shared with the defense line), except after every
    `predictor.step` the aux block (state's last dimension, `shard_frac`)
    is overwritten with its true known value instead of left to the
    model's own free-running prediction. signal_resolution_plan.md section
    0.2: shard_frac's fitted self-coefficient is 1.005 > 1, so naive
    rollout diverges on a quantity that is deterministic and fully known in
    advance (`turn/num_shards`) -- letting the model "predict" it and
    penalizing rollout error for getting it wrong is not testing whether
    the model's state is useful, it's testing whether the model can guess
    an already-known constant. The null models below get the true
    shard_frac at every row (not compounded), so this override is what
    makes the RC-1 comparison apples-to-apples."""

    squared_errors: list[float] = []
    for traj_rows in group_by_trajectory(rows, id_col="trajectory_id").values():
        pairs = build_reduced_state_pairs(traj_rows, config, y_col=Y_COL, u_col=U_COL)
        if not pairs:
            continue
        z = pairs[0]["z"]
        squared_errors.append((predictor.readout(z) - pairs[0]["y"]) ** 2)
        for pair in pairs:
            z = predictor.step(z, pair["v"])
            z[-1] = pair["z_next"][-1]
            true_y_next = pair["z_next"][config.nu - 1]
            squared_errors.append((predictor.readout(z) - true_y_next) ** 2)
    return float(np.mean(squared_errors)) if squared_errors else float("nan")


def _null_mses(train_rows: list[dict], held_out_rows: list[dict]) -> dict:
    train_y = np.array([r[Y_COL] for r in train_rows])
    held_y = np.array([r[Y_COL] for r in held_out_rows])

    global_mean = float(train_y.mean())
    mse_constant = float(np.mean((held_y - global_mean) ** 2))

    turn_mean: dict[int, float] = {}
    for t in sorted({r["turn"] for r in train_rows}):
        turn_mean[t] = float(np.mean([r[Y_COL] for r in train_rows if r["turn"] == t]))
    pred_turn = np.array([turn_mean.get(r["turn"], global_mean) for r in held_out_rows])
    mse_turn_mean = float(np.mean((held_y - pred_turn) ** 2))

    X_train = np.array([[1.0, r["shard_frac"], float(r[U_COL])] for r in train_rows])
    fit = ols(X_train, train_y)
    beta = np.array(fit["beta"])
    X_held = np.array([[1.0, r["shard_frac"], float(r[U_COL])] for r in held_out_rows])
    mse_stateless_ols = float(np.mean((held_y - X_held @ beta) ** 2))

    return {"constant": mse_constant, "turn_mean": mse_turn_mean, "stateless_ols": mse_stateless_ols}


def run_segment1(rows: list[dict]) -> dict:
    """docs/experiments/signal_resolution_plan.md section 1.2's 20-split
    protocol, copied verbatim (do not rewrite the split logic)."""

    items = sorted({r["item_id"] for r in rows})
    per_split = []
    for seed in range(N_SPLITS):
        rng = random.Random(seed)
        ids = items[:]
        rng.shuffle(ids)
        held_out_items = set(ids[:N_HELD_OUT_ITEMS])
        train_rows = [r for r in rows if r["item_id"] not in held_out_items]
        held_out_rows = [r for r in rows if r["item_id"] in held_out_items]

        nulls = _null_mses(train_rows, held_out_rows)
        best_null = min(nulls.values())

        dataset = build_identification_dataset(train_rows, STATE_CONFIG, y_col=Y_COL, u_col=U_COL)
        model = KoopmanSurrogate(extra_features_fn=no_extra_features, ridge=RIDGE).fit(dataset)
        mse_naive = rollout_output_error(model, held_out_rows, STATE_CONFIG, y_col=Y_COL, u_col=U_COL)
        mse_aux_truth = rollout_output_error_aux_truth(model, held_out_rows, STATE_CONFIG)

        per_split.append(
            {
                "seed": seed,
                **nulls,
                "best_null": best_null,
                "arx_naive": mse_naive,
                "arx_aux_truth": mse_aux_truth,
                "naive_beats_best_null": bool(mse_naive < best_null),
                "aux_truth_beats_best_null": bool(mse_aux_truth < best_null),
            }
        )

    def _mean(key: str) -> float:
        return float(np.mean([s[key] for s in per_split]))

    naive_pass_count = sum(1 for s in per_split if s["naive_beats_best_null"])
    aux_truth_pass_count = sum(1 for s in per_split if s["aux_truth_beats_best_null"])

    return {
        "n_splits": N_SPLITS,
        "per_split": per_split,
        "mean_constant": _mean("constant"),
        "mean_turn_mean": _mean("turn_mean"),
        "mean_stateless_ols": _mean("stateless_ols"),
        "mean_arx_naive": _mean("arx_naive"),
        "mean_arx_aux_truth": _mean("arx_aux_truth"),
        "naive_pass_count": naive_pass_count,
        "aux_truth_pass_count": aux_truth_pass_count,
        "rc1_pass": aux_truth_pass_count >= 18,
    }


def print_segment1(r: dict) -> None:
    print(f"\n=== segment 1 (RC-1): {r['n_splits']}-split held-out MSE vs. trivial nulls ===")
    print(f"mean const={r['mean_constant']:.4f}  mean turn_mean={r['mean_turn_mean']:.4f}  mean stateless_ols={r['mean_stateless_ols']:.4f}")
    print(f"mean ARX naive={r['mean_arx_naive']:.4f} (beats best null in {r['naive_pass_count']}/{r['n_splits']} splits)")
    print(f"mean ARX aux=truth={r['mean_arx_aux_truth']:.4f} (beats best null in {r['aux_truth_pass_count']}/{r['n_splits']} splits)")
    print(f"RC-1 pass (aux=truth beats best null in >=18/{r['n_splits']} splits)? {r['rc1_pass']}")


# ---------------------------------------------------------------------------
# Segment 2: RC-2, de-trended (by shard_frac decile bin) lag-1 autocorrelation.
# ---------------------------------------------------------------------------


def run_segment2(rows: list[dict]) -> dict:
    y = np.array([float(r[Y_COL]) for r in rows])
    shard_frac = np.array([float(r["shard_frac"]) for r in rows])
    # Decile bins on shard_frac in [0, 1]: 10 equal-width bins, last edge inclusive.
    bin_idx = np.minimum((shard_frac * 10).astype(int), 9)

    bin_mean = {b: float(y[bin_idx == b].mean()) for b in sorted(set(bin_idx.tolist()))}
    bin_mean_bcast = np.array([bin_mean[b] for b in bin_idx])
    resid = y - bin_mean_bcast
    var_by_shardfrac = float(np.var(bin_mean_bcast) / np.var(y))

    index_of = {(r["trajectory_id"], r["turn"]): i for i, r in enumerate(rows)}
    groups = group_by_key(rows, lambda r: r["trajectory_id"])

    raw_t, raw_t1, dem_t, dem_t1 = [], [], [], []
    for key, traj in groups.items():
        for a, b in zip(traj, traj[1:]):
            if b["turn"] != a["turn"] + 1:
                continue
            ia, ib = index_of[(key, a["turn"])], index_of[(key, b["turn"])]
            raw_t.append(y[ia])
            raw_t1.append(y[ib])
            dem_t.append(resid[ia])
            dem_t1.append(resid[ib])

    lag1_raw, p_raw = stats.pearsonr(raw_t, raw_t1) if len(raw_t) > 1 else (float("nan"), float("nan"))
    lag1_demeaned, p_demeaned = stats.pearsonr(dem_t, dem_t1) if len(dem_t) > 1 else (float("nan"), float("nan"))

    traj_mean_resid = {
        key: float(np.mean([resid[index_of[(key, r["turn"])]] for r in traj])) for key, traj in groups.items()
    }
    traj_bcast = np.array([traj_mean_resid[r["trajectory_id"]] for r in rows])
    stable_traj_share = float(np.var(traj_bcast) / np.var(resid))

    return {
        "n_rows": len(rows),
        "n_trajectories": len(groups),
        "n_lag1_pairs": len(raw_t),
        "var_by_shardfrac": var_by_shardfrac,
        "lag1_raw": float(lag1_raw),
        "lag1_raw_p": float(p_raw),
        "lag1_demeaned": float(lag1_demeaned),
        "lag1_demeaned_p": float(p_demeaned),
        "stable_traj_share": stable_traj_share,
        "rc2_pass": bool(lag1_demeaned > 0 and p_demeaned < 0.05),
    }


def print_segment2(r: dict) -> None:
    print("\n=== segment 2 (RC-2): de-trended (shard_frac decile) lag-1 autocorrelation ===")
    print(
        f"n_rows={r['n_rows']} n_trajectories={r['n_trajectories']} n_lag1_pairs={r['n_lag1_pairs']} "
        f"var_by_shardfrac={r['var_by_shardfrac']*100:.1f}%"
    )
    print(f"lag1_raw={r['lag1_raw']:.4f} (p={r['lag1_raw_p']:.2e})")
    print(f"lag1_demeaned={r['lag1_demeaned']:.4f} (p={r['lag1_demeaned_p']:.2e})")
    print(f"stable_traj_share={r['stable_traj_share']*100:.1f}%")
    print(f"RC-2 pass (lag1_demeaned > 0, p<0.05)? {r['rc2_pass']}")


# ---------------------------------------------------------------------------
# Segment 3: RC-3, input-gain OLS with item fixed effects, with/without u_{t+1}.
# ---------------------------------------------------------------------------


def build_input_gain_transitions(rows: list[dict]) -> tuple[list[dict], list[str]]:
    """2026-09-07 fix (docs/experiments/signal_resolution_plan.md section 0.1):
    `u_t1` must be the action on the SAME row as the predicted target `b`
    (`b[U_COL]`), not a row further ahead. The previous version zipped in a
    third row `c = traj[i+2]` and read `c[U_COL]` -- that is `u_{t+2}`, one
    turn past the row it should have matched, since ERGO's
    `contemporaneous_v=True` means the action that acts on `y_{t+1}` in the
    same turn is `b`'s own `u_reset`. That bug alone flipped RC-3 from
    passing (`u_t` p=0.0296 with the fix) to failing (p=0.6587 with the old
    code) and, chained through RC-B, caused the line to be terminated
    incorrectly -- see the plan's section 0.1 for the side-by-side numbers."""

    groups = group_by_key(rows, lambda r: r["trajectory_id"])
    item_ids = sorted({r["item_id"] for r in rows})
    dummy_items = item_ids[1:]  # drop item_ids[0] as the reference column to avoid the dummy trap.

    transitions = []
    for traj in groups.values():
        for a, b in zip(traj, traj[1:]):
            if b["turn"] != a["turn"] + 1:
                continue
            transitions.append(
                {
                    "y_t": float(a[Y_COL]),
                    "u_t": float(a[U_COL]),
                    "u_t1": float(b[U_COL]),
                    "shard_frac_t": float(a["shard_frac"]),
                    "item_id": a["item_id"],
                    "y_t1": float(b[Y_COL]),
                }
            )
    return transitions, dummy_items


def _design_matrix(transitions: list[dict], dummy_items: list[str], include_u_next: bool) -> tuple[np.ndarray, np.ndarray]:
    rows = [t for t in transitions if t["u_t1"] is not None] if include_u_next else transitions
    X, y = [], []
    for t in rows:
        base = [1.0, t["y_t"], t["u_t"]]
        if include_u_next:
            base.append(t["u_t1"])
        base.append(t["shard_frac_t"])
        base.extend(1.0 if t["item_id"] == item else 0.0 for item in dummy_items)
        X.append(base)
        y.append(t["y_t1"])
    return np.array(X), np.array(y)


def run_segment3(rows: list[dict]) -> dict:
    transitions, dummy_items = build_input_gain_transitions(rows)

    X_with, y_with = _design_matrix(transitions, dummy_items, include_u_next=True)
    fit_with = ols(X_with, y_with)
    X_without, y_without = _design_matrix(transitions, dummy_items, include_u_next=False)
    fit_without = ols(X_without, y_without)

    return {
        "n_transitions_total": len(transitions),
        "n_transitions_with_u_next": int(X_with.shape[0]),
        "n_items_fixed_effects": len(dummy_items),
        "with_u_next": {
            "formula": "y_t1 ~ 1 + y_t + u_t + u_t1 + shard_frac_t + item_FE",
            "intercept": fit_with["beta"][0],
            "y_t_coef": fit_with["beta"][1],
            "y_t_p": fit_with["p"][1],
            "u_t_coef": fit_with["beta"][2],
            "u_t_se": fit_with["se"][2],
            "u_t_p": fit_with["p"][2],
            "u_t1_coef": fit_with["beta"][3],
            "u_t1_se": fit_with["se"][3],
            "u_t1_p": fit_with["p"][3],
            "shard_frac_coef": fit_with["beta"][4],
            "shard_frac_p": fit_with["p"][4],
            "r2": fit_with["r2"],
            "n": fit_with["n"],
        },
        "without_u_next": {
            "formula": "y_t1 ~ 1 + y_t + u_t + shard_frac_t + item_FE",
            "u_t_coef": fit_without["beta"][2],
            "u_t_se": fit_without["se"][2],
            "u_t_p": fit_without["p"][2],
            "r2": fit_without["r2"],
            "n": fit_without["n"],
        },
        "rc3_pass": bool(fit_with["beta"][2] > 0 and fit_with["p"][2] < 0.05),
    }


def print_segment3(r: dict) -> None:
    print("\n=== segment 3 (RC-3): input-gain OLS, item FE, with/without u_{t+1} ===")
    w = r["with_u_next"]
    wo = r["without_u_next"]
    print(f"n_transitions_total={r['n_transitions_total']} n_with_u_next={r['n_transitions_with_u_next']} n_item_FE={r['n_items_fixed_effects']}")
    print(f"with u_t1:    u_t={w['u_t_coef']:+.4f} (se={w['u_t_se']:.4f}, p={w['u_t_p']:.2e})  u_t1={w['u_t1_coef']:+.4f} (se={w['u_t1_se']:.4f}, p={w['u_t1_p']:.2e})  R2={w['r2']:.4f} n={w['n']}")
    print(f"without u_t1: u_t={wo['u_t_coef']:+.4f} (se={wo['u_t_se']:.4f}, p={wo['u_t_p']:.2e})  R2={wo['r2']:.4f} n={wo['n']}")
    print(f"RC-3 pass (with-u_t1 spec: u_t coef > 0, p<0.05)? {r['rc3_pass']}")


# ---------------------------------------------------------------------------
# Segment 4: richer_abs_sign vacuity registration.
# ---------------------------------------------------------------------------


def run_segment4(rows: list[dict], fit_report: dict) -> dict:
    y = np.array([float(r[Y_COL]) for r in rows])
    abs_matches = bool(np.all(np.abs(y) == y))
    sign_matches = bool(np.all(np.sign(y) == y))
    arx_mse = fit_report["arx"]["train_one_step_mse"]
    richer_mse = fit_report["richer_abs_sign"]["train_one_step_mse"]
    mse_diff = abs(arx_mse - richer_mse)
    return {
        "n_rows": len(rows),
        "abs_y_equals_y_for_all_rows": abs_matches,
        "sign_y_equals_y_for_all_rows": sign_matches,
        "arx_train_one_step_mse": arx_mse,
        "richer_train_one_step_mse": richer_mse,
        "train_one_step_mse_abs_diff": mse_diff,
        "vacuous_for_binary_y": bool(abs_matches and sign_matches and mse_diff < 1e-12),
    }


def print_segment4(r: dict) -> None:
    print("\n=== segment 4: richer_abs_sign vacuity registration ===")
    print(f"n_rows={r['n_rows']} abs(y)==y for all rows? {r['abs_y_equals_y_for_all_rows']}  sign(y)==y for all rows? {r['sign_y_equals_y_for_all_rows']}")
    print(f"train_one_step_mse abs diff (arx vs richer) = {r['train_one_step_mse_abs_diff']:.2e}")
    print(f"vacuous_for_binary_y = {r['vacuous_for_binary_y']}")


def main() -> None:
    rows = _add_shard_frac(load_trajectories(ROWS_PATH))
    fit_report = json.loads(FIT_REPORT_PATH.read_text())

    # RC-0 (signal_resolution_plan.md section 0.5): the state readout and
    # the reported evaluation metric are `y_task_success` for both -- same
    # instrument, trivially satisfied by construction (not an empirical test
    # the way it is for a different candidate readout).
    rc0_pass = True
    print("\n=== segment 0 (RC-0): signal consistency ===\nstate readout == reported metric (both y_task_success) -> RC-0 pass (trivial)")

    seg1 = run_segment1(rows)
    print_segment1(seg1)
    seg2 = run_segment2(rows)
    print_segment2(seg2)
    seg3 = run_segment3(rows)
    print_segment3(seg3)
    seg4 = run_segment4(rows, fit_report)
    print_segment4(seg4)

    rc_b_pass = rc0_pass and seg1["rc1_pass"] and seg2["rc2_pass"] and seg3["rc3_pass"]
    print(
        f"\n=== RC-B verdict (all four must pass) ===\n"
        f"RC-0={rc0_pass} RC-1={seg1['rc1_pass']} RC-2={seg2['rc2_pass']} RC-3={seg3['rc3_pass']} -> RC-B={rc_b_pass}"
    )

    report = {
        "rows_path": str(ROWS_PATH),
        "fit_report_path": str(FIT_REPORT_PATH),
        "rc0_pass": rc0_pass,
        "segment1_rc1": seg1,
        "segment2_rc2": seg2,
        "segment3_rc3": seg3,
        "segment4_vacuity": seg4,
        "rc_b_pass": rc_b_pass,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if OUT_PATH.exists():
        raise SystemExit(f"refusing to overwrite existing {OUT_PATH}")
    OUT_PATH.write_text(json.dumps(report, indent=2))
    print(f"\nreport written to {OUT_PATH}")


if __name__ == "__main__":
    main()
