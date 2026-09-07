#!/usr/bin/env python3
"""E1 Step 2 (docs/experiments/backup/readout_controllability_gate_plan.md section
5.2): runs the exact same RC-2/RC-3 diagnostics
scripts/analyze_ergo_readout_state.py's segments 2/3 ran on `y_task_success`
(section 4.2), but on the two token-entropy columns
scripts/analyze_ergo_entropy_readout.py produced instead
(`entropy_mean`/`entropy_answer_span`) -- same de-trending (shard_frac
decile bin, not raw turn), same lag-1 pairing, same input-gain OLS spec
(`y_t1 ~ 1 + y_t + u_t + u_t1 + shard_frac_t + item_FE`, `u_{t+1}` included
per the plan's section 0.2(ii) lesson).

One judgment call the plan's section 5.2 doesn't pin down: RC-3 was defined
for `y_task_success` as "u_t coefficient significantly POSITIVE" (higher
success is the good direction). For an entropy readout the mechanistically
meaningful direction is the opposite sign (reset should, if anything,
LOWER uncertainty) -- this script applies the literal ">0, p<0.05" test
`rc3_pass_literal` (for exact criterion parity) AND reports
`rc3_pass_significant_either_sign` (two-sided significance regardless of
sign) so a human isn't stuck with only the wrong-direction reading if the
coefficient comes back significant and negative. See this script's final
printed section -- flag this as an open question in the report, don't
silently pick one.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

DEFAULT_ENTROPY_PATH = pathlib.Path("outputs/ergo_math_phaseB_random_excite/entropy_readout.json")
DEFAULT_OUT_PATH = pathlib.Path("outputs/ergo_math_phaseB_random_excite/entropy_readout_state_report.json")
U_COL = "u_reset"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--entropy-path", type=pathlib.Path, default=DEFAULT_ENTROPY_PATH)
    parser.add_argument("--out-path", type=pathlib.Path, default=DEFAULT_OUT_PATH)
    return parser.parse_args()


def group_by_key(rows: list[dict], key_fn) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(key_fn(row), []).append(row)
    for traj_rows in groups.values():
        traj_rows.sort(key=lambda r: r["turn"])
    return groups


def ols(X: np.ndarray, y: np.ndarray) -> dict:
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


def run_segment2(rows: list[dict], y_col: str) -> dict:
    usable = [r for r in rows if r.get(y_col) is not None]
    y = np.array([float(r[y_col]) for r in usable])
    shard_frac = np.array([r["turn"] / r["num_shards"] for r in usable])
    bin_idx = np.minimum((shard_frac * 10).astype(int), 9)

    bin_mean = {b: float(y[bin_idx == b].mean()) for b in sorted(set(bin_idx.tolist()))}
    bin_mean_bcast = np.array([bin_mean[b] for b in bin_idx])
    resid = y - bin_mean_bcast
    var_by_shardfrac = float(np.var(bin_mean_bcast) / np.var(y))

    index_of = {(r["trajectory_id"], r["turn"]): i for i, r in enumerate(usable)}
    groups = group_by_key(usable, lambda r: r["trajectory_id"])

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
    traj_bcast = np.array([traj_mean_resid[r["trajectory_id"]] for r in usable])
    stable_traj_share = float(np.var(traj_bcast) / np.var(resid))

    return {
        "y_col": y_col,
        "n_rows_usable": len(usable),
        "n_rows_dropped_null": len(rows) - len(usable),
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


def build_input_gain_transitions(rows: list[dict], y_col: str) -> tuple[list[dict], list[str]]:
    usable = [r for r in rows if r.get(y_col) is not None]
    groups = group_by_key(usable, lambda r: r["trajectory_id"])
    item_ids = sorted({r["item_id"] for r in usable})
    dummy_items = item_ids[1:]  # drop item_ids[0] as the reference column to avoid the dummy trap.

    transitions = []
    for traj in groups.values():
        for a, b, c in zip(traj, traj[1:], traj[2:] + [None]):
            if b["turn"] != a["turn"] + 1:
                continue
            u_next = float(c[U_COL]) if c is not None and c["turn"] == b["turn"] + 1 else None
            transitions.append(
                {
                    "y_t": float(a[y_col]),
                    "u_t": float(a[U_COL]),
                    "u_t1": u_next,
                    "shard_frac_t": a["turn"] / a["num_shards"],
                    "item_id": a["item_id"],
                    "y_t1": float(b[y_col]),
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


def run_segment3(rows: list[dict], y_col: str) -> dict:
    transitions, dummy_items = build_input_gain_transitions(rows, y_col)
    X_with, y_with = _design_matrix(transitions, dummy_items, include_u_next=True)
    fit_with = ols(X_with, y_with)
    X_without, y_without = _design_matrix(transitions, dummy_items, include_u_next=False)
    fit_without = ols(X_without, y_without)

    u_t_coef, u_t_p = fit_with["beta"][2], fit_with["p"][2]
    return {
        "y_col": y_col,
        "n_transitions_total": len(transitions),
        "n_transitions_with_u_next": int(X_with.shape[0]),
        "n_items_fixed_effects": len(dummy_items),
        "with_u_next": {
            "formula": "y_t1 ~ 1 + y_t + u_t + u_t1 + shard_frac_t + item_FE",
            "y_t_coef": fit_with["beta"][1],
            "y_t_p": fit_with["p"][1],
            "u_t_coef": u_t_coef,
            "u_t_se": fit_with["se"][2],
            "u_t_p": u_t_p,
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
        "rc3_pass_literal": bool(u_t_coef > 0 and u_t_p < 0.05),
        "rc3_pass_significant_either_sign": bool(u_t_p < 0.05),
    }


def print_report(col: str, seg2: dict, seg3: dict) -> None:
    print(f"\n=== {col} ===")
    print(
        f"segment 2 (RC-2): n_usable={seg2['n_rows_usable']} (dropped {seg2['n_rows_dropped_null']} null) "
        f"n_lag1_pairs={seg2['n_lag1_pairs']} var_by_shardfrac={seg2['var_by_shardfrac']*100:.1f}%"
    )
    print(f"  lag1_demeaned={seg2['lag1_demeaned']:.4f} (p={seg2['lag1_demeaned_p']:.2e})  RC-2 pass? {seg2['rc2_pass']}")
    w = seg3["with_u_next"]
    print(
        f"segment 3 (RC-3): with u_t1: u_t={w['u_t_coef']:+.4f} (se={w['u_t_se']:.4f}, p={w['u_t_p']:.2e})  "
        f"u_t1={w['u_t1_coef']:+.4f} (se={w['u_t1_se']:.4f}, p={w['u_t1_p']:.2e})  R2={w['r2']:.4f} n={w['n']}"
    )
    print(
        f"  RC-3 pass (literal, u_t>0)? {seg3['rc3_pass_literal']}   "
        f"significant either sign? {seg3['rc3_pass_significant_either_sign']}"
    )


def main() -> None:
    args = parse_args()
    data = json.loads(args.entropy_path.read_text())
    rows = data["rows"]

    report = {"entropy_path": str(args.entropy_path), "readouts": {}}
    for col in ("entropy_mean", "entropy_answer_span"):
        seg2 = run_segment2(rows, col)
        seg3 = run_segment3(rows, col)
        print_report(col, seg2, seg3)
        rc_b_literal = seg2["rc2_pass"] and seg3["rc3_pass_literal"]
        rc_b_either_sign = seg2["rc2_pass"] and seg3["rc3_pass_significant_either_sign"]
        print(f"  RC-B ({col}), literal criteria = {rc_b_literal}; RC-2 + any-sign-significant RC-3 = {rc_b_either_sign}")
        report["readouts"][col] = {"segment2_rc2": seg2, "segment3_rc3": seg3, "rc_b_literal": rc_b_literal, "rc_b_either_sign": rc_b_either_sign}

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.out_path.exists():
        raise SystemExit(f"refusing to overwrite existing {args.out_path}")
    args.out_path.write_text(json.dumps(report, indent=2))
    print(f"\nreport written to {args.out_path}")


if __name__ == "__main__":
    main()
