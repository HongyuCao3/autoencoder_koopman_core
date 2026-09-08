#!/usr/bin/env python3
"""EK-A gate G-EKA-3 (docs/experiments/ergo_fidelity_restoration_plan.md
section 4): fit the controlled operator on the counterfactual-branch arm and
score it against the three trivial nulls on item-disjoint folds.

Why a separate fitter rather than scripts/fit_koopman_ergo_closeness.py: this
arm's base branch is a single controller (u is constant along it), so the
observational construction that script uses would have no input variation to
identify `B` from. The identifying variation lives BETWEEN the two branches --
each turn contributes two transitions out of the SAME prior state,
(c_prev, u=0) -> c and (c_prev, u=1) -> c, which is a perfectly balanced
design for `B`. The null definitions are copied verbatim from that script
(constant / per-turn mean / stateless OLS on [1, shard_frac, u]) so the
numbers stay comparable to Phase B's.

CPU only.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

# Pre-registered BEFORE the arm ran (2026-09-08).
#
# G-EKA-3's PASS criterion is the item-bootstrap CI on skill (best-null MSE minus
# ARX MSE), not the fold vote. Calibration on synthetic data with a large planted
# state effect returned only 16/20 folds: with ~45 items a 20-fold vote puts ~2
# items in each fold, so the vote is a low-powered decision rule and an 18/20
# threshold would reject a true effect. That is EK0's mistake in a new costume --
# judging with an instrument too blunt for the effect. The fold vote is kept as a
# RECORD item so the numbers stay comparable to RC-1's and K2's.
DEFAULT_PASS_FOLDS = 18
N_FOLDS = 20
FOLD_SEED = 0
N_BOOTSTRAP = 4000
BOOTSTRAP_SEED = 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--branch-dir", type=pathlib.Path, required=True)
    p.add_argument("--out-path", type=pathlib.Path, required=True, help="JSON report; must not already exist.")
    p.add_argument("--ridge", type=float, default=1e-6)
    p.add_argument("--pass-folds", type=int, default=DEFAULT_PASS_FOLDS,
                   help=f"RECORD ONLY: folds the operator beats the best null on, out of {N_FOLDS}. "
                        f"Not the pass criterion -- see the module docstring.")
    p.add_argument("--controllability-horizon", type=int, default=5)
    return p.parse_args()


def build_transitions(branch_dir: pathlib.Path) -> list[dict]:
    """One row per (turn, action). Turn 1 has no prior state and is dropped."""
    base = [json.loads(l) for l in (branch_dir / "trajectories.jsonl").open() if l.strip()]
    cf = [json.loads(l) for l in (branch_dir / "counterfactual_pairs.jsonl").open() if l.strip()]
    base_by = {(r["trajectory_id"], r["turn"]): r for r in base}

    out = []
    for c in cf:
        tid, k = c["base_trajectory_id"], c["branch_turn"]
        b = base_by.get((tid, k))
        prev = base_by.get((tid, k - 1))
        if b is None:
            raise SystemExit(f"counterfactual {c['trajectory_id']} has no base row")
        if prev is None:
            continue
        for row in (b, c):
            out.append({
                "item_id": b["item_id"], "seed": b["seed"], "turn": k,
                "c_prev": prev["closeness"], "shard_frac": k / b["num_shards"],
                "u": float(row["u_reset"]), "c_next": row["closeness"],
            })
    return out


def _design(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    X = np.array([[1.0, r["c_prev"], r["u"], r["shard_frac"]] for r in rows])
    y = np.array([r["c_next"] for r in rows])
    return X, y


def _ridge_beta(X: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    penalty = ridge * np.eye(X.shape[1])
    penalty[0, 0] = 0.0  # never shrink the intercept
    return np.linalg.solve(X.T @ X + penalty, X.T @ y)


def _null_mses(train: list[dict], held: list[dict]) -> dict:
    """Copied from scripts/fit_koopman_ergo_closeness.py::_null_mses."""
    train_y = np.array([r["c_next"] for r in train])
    held_y = np.array([r["c_next"] for r in held])
    global_mean = float(train_y.mean())
    mse_constant = float(np.mean((held_y - global_mean) ** 2))
    turn_mean = {t: float(np.mean([r["c_next"] for r in train if r["turn"] == t]))
                 for t in sorted({r["turn"] for r in train})}
    pred_turn = np.array([turn_mean.get(r["turn"], global_mean) for r in held])
    mse_turn_mean = float(np.mean((held_y - pred_turn) ** 2))
    Xtr = np.array([[1.0, r["shard_frac"], r["u"]] for r in train])
    beta = np.linalg.lstsq(Xtr, train_y, rcond=None)[0]
    Xh = np.array([[1.0, r["shard_frac"], r["u"]] for r in held])
    mse_stateless = float(np.mean((held_y - Xh @ beta) ** 2))
    return {"constant": mse_constant, "turn_mean": mse_turn_mean, "stateless_ols": mse_stateless}


def main() -> None:
    args = parse_args()
    if args.out_path.exists():
        raise SystemExit(f"refusing to overwrite existing {args.out_path}")
    rows = build_transitions(args.branch_dir)
    items = sorted({r["item_id"] for r in rows})
    print(f"{len(rows)} transitions from {len(items)} items "
          f"({len(rows)//2} same-prefix pairs; turn-1 pairs dropped, no prior state)")

    # --- full-sample operator + B's CI (item bootstrap)
    X, y = _design(rows)
    beta = _ridge_beta(X, y, args.ridge)
    names = ["const", "a (c_prev)", "B (u)", "g (shard_frac)"]
    grp = np.array([r["item_id"] for r in rows])
    uniq = np.unique(grp)
    idx = {g: np.flatnonzero(grp == g) for g in uniq}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = []
    for _ in range(N_BOOTSTRAP):
        sel = np.concatenate([idx[g] for g in rng.choice(uniq, len(uniq), replace=True)])
        try:
            draws.append(_ridge_beta(X[sel], y[sel], args.ridge))
        except np.linalg.LinAlgError:
            pass
    D = np.array(draws)
    coefs = {}
    print("\n=== operator (c_next = const + a*c_prev + B*u + g*shard_frac) ===")
    for j, nm in enumerate(names):
        lo, hi = np.percentile(D[:, j], [2.5, 97.5])
        coefs[nm] = {"point": float(beta[j]), "ci": [float(lo), float(hi)]}
        print(f"  {nm:15s} {beta[j]:+.4f}  CI=[{lo:+.4f}, {hi:+.4f}]"
              + ("  <-- CI excludes 0" if lo * hi > 0 else ""))

    # --- G-EKA-3: item-disjoint folds against the three nulls
    fold_rng = np.random.default_rng(FOLD_SEED)
    shuffled = list(items)
    fold_rng.shuffle(shuffled)
    folds = np.array_split(np.array(shuffled, dtype=object), N_FOLDS)
    per_fold, wins = [], 0
    for f in folds:
        held_items = set(f.tolist())
        train = [r for r in rows if r["item_id"] not in held_items]
        held = [r for r in rows if r["item_id"] in held_items]
        if not held or not train:
            continue
        Xtr, ytr = _design(train)
        Xh, yh = _design(held)
        b = _ridge_beta(Xtr, ytr, args.ridge)
        arx = float(np.mean((yh - Xh @ b) ** 2))
        nulls = _null_mses(train, held)
        best = min(nulls.values())
        wins += int(arx < best)
        per_fold.append({"n_held": len(held), "arx": arx, **nulls})
    n_folds = len(per_fold)
    skill = np.array([min(f["constant"], f["turn_mean"], f["stateless_ols"]) - f["arx"] for f in per_fold])
    frng = np.random.default_rng(BOOTSTRAP_SEED)
    sk_draws = [float(skill[frng.integers(0, n_folds, n_folds)].mean()) for _ in range(N_BOOTSTRAP)]
    sk_lo, sk_hi = (float(v) for v in np.percentile(sk_draws, [2.5, 97.5]))
    print(f"\n=== G-EKA-3: {n_folds} item-disjoint folds, scalar one-step MSE ===")
    for k in ("constant", "turn_mean", "stateless_ols"):
        print(f"  null {k:14s} mean={np.mean([f[k] for f in per_fold]):.6f}")
    print(f"  arx              mean={np.mean([f['arx'] for f in per_fold]):.6f}"
          f"   beats best null in {wins}/{n_folds} folds")
    passed = sk_lo > 0
    print(f"  mean skill (best null - arx) = {skill.mean():+.6f}  CI=[{sk_lo:+.6f}, {sk_hi:+.6f}]")
    print(f"  G-EKA-3 pass criterion is the skill CI excluding 0 -> {'PASS' if passed else 'FAIL'}")
    print(f"  (record only, for comparability with RC-1/K2: {wins}/{n_folds} vs a {args.pass_folds}/{N_FOLDS} vote)")

    # --- non-degeneracy, same shape the defense line reports
    a = float(beta[1])
    Bc = float(beta[2])
    ctrb = np.array([[Bc * a**i for i in range(args.controllability_horizon)]])
    gram = float(sum((a**i * Bc) ** 2 for i in range(args.controllability_horizon)))
    print(f"\n=== non-degeneracy ===")
    print(f"  spectral radius |a| = {abs(a):.4f}   controllability rank = {int(np.linalg.matrix_rank(ctrb))}"
          f"   finite-horizon Gramian = {gram:.6e}")

    report = {
        "branch_dir": str(args.branch_dir), "n_transitions": len(rows), "n_items": len(items),
        "ridge": args.ridge, "coefficients": coefs,
        "g_eka_3": {"n_folds": n_folds, "mean_skill": float(skill.mean()), "ci_skill": [sk_lo, sk_hi],
                    "pass": bool(passed), "criterion": "item-bootstrap CI on skill excludes 0",
                    "arx_wins_record_only": wins, "fold_vote_reference": args.pass_folds,
                    "per_fold": per_fold},
        "non_degeneracy": {"spectral_radius": abs(a), "gramian": gram,
                           "controllability_horizon": args.controllability_horizon},
        "n_bootstrap": N_BOOTSTRAP, "bootstrap_seed": BOOTSTRAP_SEED, "fold_seed": FOLD_SEED,
    }
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_path.write_text(json.dumps(report, indent=2))
    print(f"\nreport written to {args.out_path}")


if __name__ == "__main__":
    main()
