#!/usr/bin/env python3
"""F1 (docs/experiments/signal_resolution_plan.md section two): formalizes
`closeness` -- the SAME extractor/gold-answer comparison
`ergo_math_judge.judge_math_answer` already runs, just not thresholded into
0/1 -- as ERGO's candidate state readout, and checks it against all four
RC gates (RC-0/RC-1/RC-2/RC-3). Also recomputes section 0.3's seven-row
candidate table (`y_task_success`/`closeness`/`coverage`/`reply_len`/
`churn`/`entropy_mean`/`entropy_answer_span`) as the archived evidence for
why `closeness` was chosen over the other de-thresholding candidates.

CPU-only, pure numpy/scipy/re. Needs
outputs/ergo_math_phaseB_random_excite/trajectories.jsonl and (for the two
entropy columns only) entropy_readout.json from
scripts/analyze_ergo_entropy_readout.py (already run, E1 Step 1).
"""

from __future__ import annotations

import json
import pathlib
import random
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

from persona_drift.ergo_math_judge import _normalize, extract_answer_by_regex  # noqa: E402
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
ENTROPY_PATH = pathlib.Path("outputs/ergo_math_phaseB_random_excite/entropy_readout.json")
OUT_PATH = pathlib.Path("outputs/ergo_math_phaseB_random_excite/closeness_readout_state_report.json")

U_COL = "u_reset"
N_SPLITS = 20
N_HELD_OUT_ITEMS = 15
RIDGE = 1e-6
STATE_CONFIG = ReducedStateConfig(nu=1, mu=1, aux_cols=("shard_frac",), contemporaneous_v=True)

NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


# ---------------------------------------------------------------------------
# Readout definitions.
# ---------------------------------------------------------------------------


def _to_number(text: str | None) -> float | None:
    if text is None:
        return None
    try:
        return float(_normalize(text))
    except (ValueError, TypeError):
        return None


def _closeness(row: dict) -> float:
    """Section 2.2's definition, copied verbatim -- do not change the
    normalization. 1.0 iff the extracted answer exactly equals gold;
    0.0 on any parse failure or non-numeric extraction (same convention
    ergo_math_judge's hard 0/1 score already uses for parse failures)."""

    extracted = extract_answer_by_regex(row["agent_message"])
    a = _to_number(extracted) if extracted is not None else None
    g = _to_number(row["gold_answer"])
    if a is None or g is None:
        return 0.0
    return 1.0 / (1.0 + abs(a - g) / max(abs(g), 1.0))


def _add_derived_columns(rows: list[dict]) -> list[dict]:
    for row in rows:
        row["shard_frac"] = row["turn"] / row["num_shards"]
        row["closeness"] = _closeness(row)
        row["reply_len"] = float(len(row["agent_message"].split()))
    # coverage/churn need per-trajectory sequential state (accumulated
    # revealed numbers, previous turn's extracted answer).
    for traj_rows in group_by_trajectory(rows).values():
        revealed: set[str] = set()
        prev_answer = None
        for row in traj_rows:
            revealed |= set(NUM_RE.findall(row.get("shard_text") or ""))
            hit = [n for n in revealed if re.search(r"(?<!\d)" + re.escape(n) + r"(?!\d)", row["agent_message"])]
            row["coverage"] = len(hit) / len(revealed) if revealed else float("nan")

            extracted = extract_answer_by_regex(row["agent_message"])
            row["churn"] = float(prev_answer is None or extracted != prev_answer)
            prev_answer = extracted
    return rows


def _load_entropy_columns(rows: list[dict]) -> None:
    data = json.loads(ENTROPY_PATH.read_text())
    by_key = {(r["trajectory_id"], r["turn"]): r for r in data["rows"]}
    for row in rows:
        entry = by_key[(row["trajectory_id"], row["turn"])]
        row["entropy_mean"] = entry["entropy_mean"]
        row["entropy_answer_span"] = entry["entropy_answer_span"]


# ---------------------------------------------------------------------------
# G-F1-1: closeness distribution gate.
# ---------------------------------------------------------------------------


def run_gate_f1_1(rows: list[dict]) -> dict:
    values = np.array([r["closeness"] for r in rows])
    return {
        "n_rows": len(values),
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=0)),
        "frac_eq_1": float(np.mean(values == 1.0)),
        "frac_eq_0": float(np.mean(values == 0.0)),
    }


def print_gate_f1_1(r: dict) -> None:
    print("\n=== G-F1-1: closeness distribution ===")
    print(f"n_rows={r['n_rows']} mean={r['mean']:.4f} sd={r['sd']:.4f} frac==1.0={r['frac_eq_1']:.4f} frac==0.0={r['frac_eq_0']:.4f}")


# ---------------------------------------------------------------------------
# Shared per-candidate diagnostics (coupling, de-trended lag-1, input gain).
# ---------------------------------------------------------------------------


def _shardfrac_bin_idx(rows: list[dict]) -> np.ndarray:
    shard_frac = np.array([float(r["shard_frac"]) for r in rows])
    return np.minimum((shard_frac * 10).astype(int), 9)


def _detrend(values: np.ndarray, bin_idx: np.ndarray) -> np.ndarray:
    bin_mean = {b: float(values[bin_idx == b].mean()) for b in sorted(set(bin_idx.tolist()))}
    return values - np.array([bin_mean[b] for b in bin_idx])


def ols(X: np.ndarray, y: np.ndarray) -> dict:
    n, k = X.shape
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    rss = float(np.sum(resid**2))
    dof = n - k
    sigma2 = rss / dof
    xtx_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(sigma2 * xtx_inv))
    t = beta / se
    p = 2.0 * stats.t.sf(np.abs(t), dof)
    return {"beta": beta.tolist(), "se": se.tolist(), "p": p.tolist(), "n": n, "dof": dof}


def candidate_row(all_rows: list[dict], col: str) -> dict:
    """One row of section 0.3's seven-row table for readout `col`: rows
    with a null `col` (only entropy_answer_span has any) are dropped for
    every statistic below."""

    usable = [r for r in all_rows if r.get(col) is not None and r[col] == r[col]]  # drop None/NaN
    x = np.array([float(r[col]) for r in usable])
    target = np.array([float(r["y_task_success"]) for r in usable])
    bin_idx = _shardfrac_bin_idx(usable)

    rho_raw, _ = stats.spearmanr(x, target)
    x_resid = _detrend(x, bin_idx)
    target_resid = _detrend(target, bin_idx)
    rho_detrended, _ = stats.spearmanr(x_resid, target_resid)

    # lag-1 (own signal, de-trended) -- same definition as RC-2.
    index_of = {(r["trajectory_id"], r["turn"]): i for i, r in enumerate(usable)}
    groups = group_by_trajectory(usable)
    dem_t, dem_t1 = [], []
    for key, traj in groups.items():
        for a, b in zip(traj, traj[1:]):
            if b["turn"] != a["turn"] + 1:
                continue
            ia, ib = index_of[(key, a["turn"])], index_of[(key, b["turn"])]
            dem_t.append(x_resid[ia])
            dem_t1.append(x_resid[ib])
    lag1_demeaned, lag1_p = stats.pearsonr(dem_t, dem_t1) if len(dem_t) > 1 else (float("nan"), float("nan"))

    # input-gain OLS, item FE, u_{t+1} = predicted-target row's own action.
    # dummy_items must come from the items actually present in the transition
    # set (post turn-contiguity pairing), not from `usable`'s raw rows -- a
    # NaN-dropped readout (coverage/entropy_answer_span) can leave an item
    # with zero surviving transitions, whose all-zero dummy column would
    # make the design matrix singular.
    raw_transitions = []
    for key, traj in groups.items():
        for a, b in zip(traj, traj[1:]):
            if b["turn"] != a["turn"] + 1:
                continue
            raw_transitions.append((a, b))
    item_ids = sorted({a["item_id"] for a, _ in raw_transitions})
    dummy_items = item_ids[1:]
    X_rows, y_rows = [], []
    for a, b in raw_transitions:
        row_vec = [1.0, float(a[col]), float(a[U_COL]), float(b[U_COL]), float(a["shard_frac"])]
        row_vec.extend(1.0 if a["item_id"] == item else 0.0 for item in dummy_items)
        X_rows.append(row_vec)
        y_rows.append(float(b[col]))
    fit = ols(np.array(X_rows), np.array(y_rows))

    return {
        "readout": col,
        "n_usable": len(usable),
        "rho_target": float(rho_raw),
        "rho_target_detrended": float(rho_detrended),
        "lag1_demeaned": float(lag1_demeaned),
        "lag1_demeaned_p": float(lag1_p),
        "u_t_coef": fit["beta"][2],
        "u_t_p": fit["p"][2],
        "n_transitions": fit["n"],
    }


def print_candidate_table(table: list[dict]) -> None:
    print("\n=== section 0.3 candidate table (recomputed) ===")
    print(f"{'readout':<22}{'rho_target':>11}{'rho_detrend':>12}{'lag1_dem':>10}{'u_t':>10}{'p':>10}")
    for r in table:
        print(
            f"{r['readout']:<22}{r['rho_target']:>+11.4f}{r['rho_target_detrended']:>+12.4f}"
            f"{r['lag1_demeaned']:>+10.4f}{r['u_t_coef']:>+10.4f}{r['u_t_p']:>10.2e}"
        )


# ---------------------------------------------------------------------------
# RC-1/RC-2/RC-3 for closeness specifically (same protocol as
# analyze_ergo_readout_state.py, parameterized on Y_COL="closeness").
# ---------------------------------------------------------------------------


def rollout_output_error_aux_truth(predictor, rows: list[dict], config: ReducedStateConfig, y_col: str) -> float:
    squared_errors: list[float] = []
    for traj_rows in group_by_trajectory(rows, id_col="trajectory_id").values():
        pairs = build_reduced_state_pairs(traj_rows, config, y_col=y_col, u_col=U_COL)
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


def _null_mses(train_rows: list[dict], held_out_rows: list[dict], y_col: str) -> dict:
    train_y = np.array([r[y_col] for r in train_rows])
    held_y = np.array([r[y_col] for r in held_out_rows])

    global_mean = float(train_y.mean())
    mse_constant = float(np.mean((held_y - global_mean) ** 2))

    turn_mean: dict[int, float] = {}
    for t in sorted({r["turn"] for r in train_rows}):
        turn_mean[t] = float(np.mean([r[y_col] for r in train_rows if r["turn"] == t]))
    pred_turn = np.array([turn_mean.get(r["turn"], global_mean) for r in held_out_rows])
    mse_turn_mean = float(np.mean((held_y - pred_turn) ** 2))

    X_train = np.array([[1.0, r["shard_frac"], float(r[U_COL])] for r in train_rows])
    fit = ols(X_train, train_y)
    beta = np.array(fit["beta"])
    X_held = np.array([[1.0, r["shard_frac"], float(r[U_COL])] for r in held_out_rows])
    mse_stateless_ols = float(np.mean((held_y - X_held @ beta) ** 2))

    return {"constant": mse_constant, "turn_mean": mse_turn_mean, "stateless_ols": mse_stateless_ols}


def run_rc1(rows: list[dict], y_col: str) -> dict:
    items = sorted({r["item_id"] for r in rows})
    per_split = []
    for seed in range(N_SPLITS):
        rng = random.Random(seed)
        ids = items[:]
        rng.shuffle(ids)
        held_out_items = set(ids[:N_HELD_OUT_ITEMS])
        train_rows = [r for r in rows if r["item_id"] not in held_out_items]
        held_out_rows = [r for r in rows if r["item_id"] in held_out_items]

        nulls = _null_mses(train_rows, held_out_rows, y_col)
        best_null = min(nulls.values())

        dataset = build_identification_dataset(train_rows, STATE_CONFIG, y_col=y_col, u_col=U_COL)
        model = KoopmanSurrogate(extra_features_fn=no_extra_features, ridge=RIDGE).fit(dataset)
        mse_naive = rollout_output_error(model, held_out_rows, STATE_CONFIG, y_col=y_col, u_col=U_COL)
        mse_aux_truth = rollout_output_error_aux_truth(model, held_out_rows, STATE_CONFIG, y_col)

        per_split.append(
            {
                "seed": seed,
                **nulls,
                "best_null": best_null,
                "arx_naive": mse_naive,
                "arx_aux_truth": mse_aux_truth,
                "aux_truth_beats_best_null": bool(mse_aux_truth < best_null),
            }
        )

    def _mean(key: str) -> float:
        return float(np.mean([s[key] for s in per_split]))

    aux_truth_pass_count = sum(1 for s in per_split if s["aux_truth_beats_best_null"])
    skills = [1.0 - s["arx_aux_truth"] / s["best_null"] for s in per_split]
    return {
        "n_splits": N_SPLITS,
        "mean_arx_aux_truth": _mean("arx_aux_truth"),
        "mean_best_null": float(np.mean([s["best_null"] for s in per_split])),
        "mean_skill": float(np.mean(skills)),
        "aux_truth_pass_count": aux_truth_pass_count,
        "rc1_pass": aux_truth_pass_count >= 18,
    }


def run_rc2(rows: list[dict], y_col: str) -> dict:
    y = np.array([float(r[y_col]) for r in rows])
    bin_idx = _shardfrac_bin_idx(rows)
    resid = _detrend(y, bin_idx)

    index_of = {(r["trajectory_id"], r["turn"]): i for i, r in enumerate(rows)}
    groups = group_by_trajectory(rows)
    dem_t, dem_t1 = [], []
    for key, traj in groups.items():
        for a, b in zip(traj, traj[1:]):
            if b["turn"] != a["turn"] + 1:
                continue
            ia, ib = index_of[(key, a["turn"])], index_of[(key, b["turn"])]
            dem_t.append(resid[ia])
            dem_t1.append(resid[ib])
    lag1_demeaned, p = stats.pearsonr(dem_t, dem_t1) if len(dem_t) > 1 else (float("nan"), float("nan"))
    return {"lag1_demeaned": float(lag1_demeaned), "p": float(p), "rc2_pass": bool(lag1_demeaned > 0 and p < 0.05)}


def run_rc3(rows: list[dict], y_col: str) -> dict:
    row = candidate_row(rows, y_col)
    return {
        "u_t_coef": row["u_t_coef"],
        "u_t_p": row["u_t_p"],
        "n": row["n_transitions"],
        "rc3_pass": bool(row["u_t_coef"] > 0 and row["u_t_p"] < 0.05),
    }


def main() -> None:
    rows = _add_derived_columns(load_trajectories(ROWS_PATH))
    _load_entropy_columns(rows)

    gate1 = run_gate_f1_1(rows)
    print_gate_f1_1(gate1)

    table = [candidate_row(rows, col) for col in
             ("y_task_success", "closeness", "coverage", "reply_len", "churn", "entropy_mean", "entropy_answer_span")]
    print_candidate_table(table)

    rc0_pass = True  # same extractor + same gold answer as y_task_success -- constructively RC-0.
    rc1 = run_rc1(rows, "closeness")
    rc2 = run_rc2(rows, "closeness")
    rc3 = run_rc3(rows, "closeness")

    print("\n=== closeness: RC-0/RC-1/RC-2/RC-3 ===")
    print("RC-0 pass (same instrument as y_task_success, constructive)? True")
    print(f"RC-1: mean aux_truth mse={rc1['mean_arx_aux_truth']:.4f} mean best_null={rc1['mean_best_null']:.4f} mean skill={rc1['mean_skill']:.4f} pass_count={rc1['aux_truth_pass_count']}/{rc1['n_splits']} -> {rc1['rc1_pass']}")
    print(f"RC-2: lag1_demeaned={rc2['lag1_demeaned']:.4f} p={rc2['p']:.4e} -> {rc2['rc2_pass']}")
    print(f"RC-3: u_t={rc3['u_t_coef']:+.4f} p={rc3['u_t_p']:.4e} n={rc3['n']} -> {rc3['rc3_pass']}")

    rc_b_pass = rc0_pass and rc1["rc1_pass"] and rc2["rc2_pass"] and rc3["rc3_pass"]
    print(f"\n=== closeness RC-B verdict === RC-0={rc0_pass} RC-1={rc1['rc1_pass']} RC-2={rc2['rc2_pass']} RC-3={rc3['rc3_pass']} -> RC-B={rc_b_pass}")

    # Section 2.4's "not worse than the binary version on any of the four criteria" check.
    y_task_success_row = table[0]
    closeness_row = table[1]
    not_worse = (
        rc1["mean_skill"] >= 0.128 - 1e-9  # y_task_success's own skill from section 0.2 (sanity floor)
        and rc2["lag1_demeaned"] >= 0.2578 - 1e-9
        and rc3["u_t_p"] <= 0.05
    )
    print(f"\ncloseness not worse than y_task_success on all four criteria (section 2.4)? {not_worse}")

    report = {
        "rows_path": str(ROWS_PATH),
        "gate_f1_1": gate1,
        "candidate_table": table,
        "closeness_rc0_pass": rc0_pass,
        "closeness_rc1": rc1,
        "closeness_rc2": rc2,
        "closeness_rc3": rc3,
        "closeness_rc_b_pass": rc_b_pass,
        "closeness_not_worse_than_binary": not_worse,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if OUT_PATH.exists():
        raise SystemExit(f"refusing to overwrite existing {OUT_PATH}")
    OUT_PATH.write_text(json.dumps(report, indent=2))
    print(f"\nreport written to {OUT_PATH}")


if __name__ == "__main__":
    main()
