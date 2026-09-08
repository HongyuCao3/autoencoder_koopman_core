#!/usr/bin/env python3
"""EK-A gates G-EKA-1 / G-EKA-2 / G-EKA-4
(docs/experiments/ergo_fidelity_restoration_plan.md section 4), on the
counterfactual-branch arm produced by scripts/run_ergo_branch_arm.py.

Each turn of that arm was generated twice from a byte-identical prefix under
opposite actions, so `delta_c = closeness(u=1) - closeness(u=0)` is a direct
measurement of that turn's one-step causal gain. No regression has to control
for the prefix -- which is the whole reason this arm exists: EK0 showed the
observational `random_excite p=0.5` arm's endpoint MDE is +0.49 against an
effect of +0.181, so its "u does nothing" numbers ruled out nothing.

CPU only. Inference is item-level bootstrap throughout (an item contributes
several turns and several seeds; those are not independent draws).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

N_BOOTSTRAP = 4000
BOOTSTRAP_SEED = 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--branch-dir", type=pathlib.Path, required=True,
                   help="run_ergo_branch_arm.py output dir (trajectories.jsonl + counterfactual_pairs.jsonl)")
    p.add_argument("--out-path", type=pathlib.Path, required=True, help="JSON report; must not already exist.")
    return p.parse_args()


def load_pairs(branch_dir: pathlib.Path) -> list[dict]:
    """Join each base turn to its counterfactual and orient by ACTION, not by
    which branch happened to be the base: `c_reset` is always the u=1 side."""

    base = [json.loads(l) for l in (branch_dir / "trajectories.jsonl").open() if l.strip()]
    cf = [json.loads(l) for l in (branch_dir / "counterfactual_pairs.jsonl").open() if l.strip()]
    base_by = {(r["trajectory_id"], r["turn"]): r for r in base}
    prev_c = {(r["trajectory_id"], r["turn"]): None for r in base}
    for r in base:
        prev = base_by.get((r["trajectory_id"], r["turn"] - 1))
        prev_c[(r["trajectory_id"], r["turn"])] = None if prev is None else prev["closeness"]

    pairs = []
    for c in cf:
        key = (c["base_trajectory_id"], c["branch_turn"])
        b = base_by.get(key)
        if b is None:
            raise SystemExit(f"counterfactual {c['trajectory_id']} has no base row at {key}")
        if b["shard_text"] != c["shard_text"] or b["u_reset"] == c["u_reset"]:
            raise SystemExit(f"integrity: {key} is not a flipped same-turn pair")
        reset_row, plain_row = (c, b) if c["u_reset"] else (b, c)
        pairs.append({
            "item_id": b["item_id"],
            "seed": b["seed"],
            "turn": b["turn"],
            "num_shards": b["num_shards"],
            "shard_frac": b["turn"] / b["num_shards"],
            "c_prev": prev_c[key],
            "c_reset": reset_row["closeness"],
            "c_plain": plain_row["closeness"],
            "delta_c": reset_row["closeness"] - plain_row["closeness"],
            "delta_y": float(reset_row["y_task_success"]) - float(plain_row["y_task_success"]),
            "reply_reset": reset_row["agent_message"],
            "reply_plain": plain_row["agent_message"],
        })
    return pairs


def boot_ci(values: np.ndarray, groups: np.ndarray, stat) -> tuple[float, float]:
    uniq = np.unique(groups)
    idx = {g: np.flatnonzero(groups == g) for g in uniq}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = []
    for _ in range(N_BOOTSTRAP):
        sel = np.concatenate([idx[g] for g in rng.choice(uniq, len(uniq), replace=True)])
        try:
            draws.append(stat(sel))
        except (np.linalg.LinAlgError, ZeroDivisionError):
            pass
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(lo), float(hi)


def ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def gate_eka1(pairs: list[dict]) -> dict:
    d = np.array([p["delta_c"] for p in pairs])
    dy = np.array([p["delta_y"] for p in pairs])
    g = np.array([p["item_id"] for p in pairs])
    lo, hi = boot_ci(d, g, lambda sel: float(d[sel].mean()))
    ylo, yhi = boot_ci(dy, g, lambda sel: float(dy[sel].mean()))
    return {
        "n_pairs": len(pairs), "n_items": int(len(np.unique(g))),
        "mean_delta_c": float(d.mean()), "ci_delta_c": [lo, hi],
        "mean_delta_y": float(dy.mean()), "ci_delta_y": [ylo, yhi],
        "share_delta_c_positive": float((d > 0).mean()),
        "share_delta_c_zero": float((d == 0).mean()),
        "pass": bool(lo * hi > 0),
    }


def gate_eka2(pairs: list[dict]) -> dict:
    """delta_c ~ 1 + c_prev. Turn 1 has no prior state and is excluded (recorded)."""
    usable = [p for p in pairs if p["c_prev"] is not None]
    d = np.array([p["delta_c"] for p in usable])
    x = np.array([p["c_prev"] for p in usable])
    g = np.array([p["item_id"] for p in usable])
    X = np.c_[np.ones(len(x)), x]
    beta = ols(X, d)
    lo, hi = boot_ci(d, g, lambda sel: float(ols(np.c_[np.ones(len(sel)), x[sel]], d[sel])[1]))
    return {
        "n_pairs_used": len(usable), "n_pairs_dropped_turn1": len(pairs) - len(usable),
        "intercept": float(beta[0]), "slope_on_c_prev": float(beta[1]), "ci_slope": [lo, hi],
        "pass": bool(lo * hi > 0),
        "expected_sign": "negative (reset matters less the closer the agent already is)",
        "scatter": [{"c_prev": p["c_prev"], "delta_c": p["delta_c"], "turn": p["turn"],
                     "item_id": p["item_id"], "seed": p["seed"]} for p in usable],
    }


def _jaccard(a: str, b: str) -> float:
    sa, sb = set(a.split()), set(b.split())
    return len(sa & sb) / len(sa | sb) if (sa | sb) else 0.0


def record_items(pairs: list[dict], branch_dir: pathlib.Path) -> dict:
    """G-EKA-4: does the 4B failure look like ECHO (copying its own previous
    reply) or DRIFT (carrying an early wrong assumption forward)? Measured on
    the base branch, where consecutive turns exist."""

    base = [json.loads(l) for l in (branch_dir / "trajectories.jsonl").open() if l.strip()]
    by = {}
    for r in base:
        by.setdefault(r["trajectory_id"], []).append(r)
    exact, jacc = [], []
    for rs in by.values():
        rs.sort(key=lambda r: r["turn"])
        for a, b in zip(rs, rs[1:]):
            exact.append(float(a["agent_message"] == b["agent_message"]))
            jacc.append(_jaccard(a["agent_message"], b["agent_message"]))
    d = np.array([p["delta_c"] for p in pairs])
    turns = np.array([p["turn"] for p in pairs])
    solved = np.array([(p["c_prev"] is not None and p["c_prev"] >= 0.99) for p in pairs])
    return {
        "echo_exact_duplicate_rate": float(np.mean(exact)) if exact else None,
        "echo_token_jaccard_mean": float(np.mean(jacc)) if jacc else None,
        "delta_c_by_turn": {int(t): float(d[turns == t].mean()) for t in sorted(set(turns.tolist()))},
        "delta_c_from_solved": float(d[solved].mean()) if solved.any() else None,
        "delta_c_from_unsolved": float(d[~solved].mean()) if (~solved).any() else None,
        "n_from_solved": int(solved.sum()),
    }


def main() -> None:
    args = parse_args()
    if args.out_path.exists():
        raise SystemExit(f"refusing to overwrite existing {args.out_path}")
    pairs = load_pairs(args.branch_dir)
    eka1, eka2, rec = gate_eka1(pairs), gate_eka2(pairs), record_items(pairs, args.branch_dir)

    print(f"=== G-EKA-1 executor authority (same-prefix pairs) ===")
    print(f"n_pairs={eka1['n_pairs']} over {eka1['n_items']} items")
    print(f"mean delta_c = {eka1['mean_delta_c']:+.4f}  CI=[{eka1['ci_delta_c'][0]:+.4f}, {eka1['ci_delta_c'][1]:+.4f}]"
          f"  -> {'PASS' if eka1['pass'] else 'FAIL'}")
    print(f"mean delta_y = {eka1['mean_delta_y']:+.4f}  CI=[{eka1['ci_delta_y'][0]:+.4f}, {eka1['ci_delta_y'][1]:+.4f}] (record)")
    print(f"delta_c > 0 in {eka1['share_delta_c_positive']*100:.1f}% of pairs, exactly 0 in {eka1['share_delta_c_zero']*100:.1f}%")

    print(f"\n=== G-EKA-2 state dependence (delta_c ~ c_prev) ===")
    print(f"n={eka2['n_pairs_used']} (dropped {eka2['n_pairs_dropped_turn1']} turn-1 pairs, no prior state)")
    print(f"slope = {eka2['slope_on_c_prev']:+.4f}  CI=[{eka2['ci_slope'][0]:+.4f}, {eka2['ci_slope'][1]:+.4f}]"
          f"  intercept={eka2['intercept']:+.4f}  -> {'PASS' if eka2['pass'] else 'FAIL'}")

    print(f"\n=== G-EKA-4 record items ===")
    print(f"echo: exact-duplicate reply rate={rec['echo_exact_duplicate_rate']:.4f}  "
          f"token Jaccard with previous reply={rec['echo_token_jaccard_mean']:.4f}")
    print(f"delta_c from already-solved states={rec['delta_c_from_solved']} (n={rec['n_from_solved']}), "
          f"from unsolved={rec['delta_c_from_unsolved']:.4f}")
    print(f"delta_c by turn: " + "  ".join(f"t{t}={v:+.4f}" for t, v in rec["delta_c_by_turn"].items()))

    report = {
        "branch_dir": str(args.branch_dir),
        "n_bootstrap": N_BOOTSTRAP, "bootstrap_seed": BOOTSTRAP_SEED,
        "g_eka_1": eka1, "g_eka_2": eka2, "g_eka_4_record": rec,
    }
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_path.write_text(json.dumps(report, indent=2))
    print(f"\nreport written to {args.out_path}")


if __name__ == "__main__":
    main()
