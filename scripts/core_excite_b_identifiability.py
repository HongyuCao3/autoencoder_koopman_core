#!/usr/bin/env python3
"""#4b analysis: is the command effect B identified on the excited core data?

Spec: docs/experiments/command_channel_identifiability_plan.md section 3.2.
Gate (signed before collection): the main element's 95% interval excludes 0
on every seed AND its sign agrees across the three seeds. Sign stability
>= 0.95 is reported beside it, as in #4a.

Fit (fixed before the first run, 2026-11-05): Table 1's state for
sentence_length_t10, lag=3 -> z_t = [y_t, y_(t-1), y_(t-2), y_(t-3)], and
the per-turn command c_(t+1) = `command_norm` of row t+1 (the word count the
prompt for reply t+1 asked for):

    z_(t+1) = A z_t + B c_(t+1) + b,    main element = C B = B[0].

Ridge alpha 1e-6, as AugmentedKoopmanModel. Resampling unit: trajectory,
2000 resamples, default_rng(0). A lag=0 column is written as a robustness
note and does not enter the verdict. K, B and controllability_diagnostics
(H=6, Table 1's horizon) of the full-data lag=3 fit are written for Umesh.

Zero GPU; reads persona_drift_control/outputs/core_excite_sentence_length_<s>/,
writes results/core_excite_b_identifiability/ and refuses to overwrite.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from koopman_ae.core import controllability_diagnostics  # noqa: E402

SEEDS = (0, 1, 2)
LAG = 3
ROBUST_LAG = 0
HORIZON = 6
ALPHA = 1e-6
N_RESAMPLES = 2000
BOOTSTRAP_SEED = 0
SIGN_STABILITY_MIN = 0.95
IN_TEMPLATE = "persona_drift_control/outputs/core_excite_sentence_length_{seed}/trajectories.jsonl"
OUT_DIR = REPO_ROOT / "results" / "core_excite_b_identifiability"


def transitions(rows: list[dict], lag: int) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Per trajectory: Z_t (n, lag+1), c (n,), Z_next (n, lag+1)."""
    by_traj: dict[str, dict[int, dict]] = {}
    for r in rows:
        by_traj.setdefault(r["trajectory_id"], {})[int(r["turn"])] = r
    out = {}
    for tid, turns in by_traj.items():
        y = {t: float(r["normalized_output"]) for t, r in turns.items()}
        Z, C, Zn = [], [], []
        for t in sorted(turns):
            if t - lag not in y or t + 1 not in turns:
                continue
            Z.append([y[t - k] for k in range(lag + 1)])
            Zn.append([y[t + 1 - k] for k in range(lag + 1)])
            C.append(float(turns[t + 1]["command_norm"]))
        if Z:
            out[tid] = (np.array(Z), np.array(C), np.array(Zn))
    return out


def fit(units: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    Z = np.vstack([u[0] for u in units])
    c = np.concatenate([u[1] for u in units])
    Zn = np.vstack([u[2] for u in units])
    X = np.column_stack([Z, c, np.ones(len(Z))])
    theta = np.linalg.solve(X.T @ X + ALPHA * np.eye(X.shape[1]), X.T @ Zn)
    d = Z.shape[1]
    return theta[:d].T, theta[d : d + 1].T, theta[-1]


def identifiability(units) -> dict:
    """Rank of [Z | c | 1] and R^2 of c on [Z | 1] -- is B defined at all?"""
    Z = np.vstack([u[0] for u in units])
    c = np.concatenate([u[1] for u in units])
    X = np.column_stack([Z, c, np.ones(len(Z))])
    W = np.column_stack([Z, np.ones(len(Z))])
    resid = c - W @ np.linalg.lstsq(W, c, rcond=None)[0]
    r2 = 1.0 - float(resid @ resid) / float(((c - c.mean()) ** 2).sum())
    return {"rank": int(np.linalg.matrix_rank(X)), "n_columns": int(X.shape[1]), "r2_command_on_state": r2}


def main_element(units, rng) -> dict:
    _, B, _ = fit(units)
    point = float(B[0, 0])
    n = len(units)
    draws = np.empty(N_RESAMPLES)
    for i in range(N_RESAMPLES):
        idx = rng.integers(0, n, n)
        draws[i] = fit([units[j] for j in idx])[1][0, 0]
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"point": point, "ci95": [float(lo), float(hi)],
            "sign_stability": float(np.mean(np.sign(draws) == np.sign(point))),
            "n_trajectories": n, "n_transitions": int(sum(len(u[1]) for u in units))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", type=pathlib.Path, default=OUT_DIR)
    args = ap.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"{args.out_dir} exists; refusing to overwrite")

    per_seed = {}
    for seed in SEEDS:
        rows = [json.loads(l) for l in open(REPO_ROOT / IN_TEMPLATE.format(seed=seed))]
        if {r["command_policy"] for r in rows} != {"random"}:
            raise ValueError(f"seed {seed}: not a random-command collection")
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        units = list(transitions(rows, LAG).values())
        robust_units = list(transitions(rows, ROBUST_LAG).values())
        A, B, b = fit(units)
        per_seed[seed] = {
            "identifiability": identifiability(units),
            "main_element_CB": main_element(units, rng),
            "robust_lag0_CB": main_element(robust_units, np.random.default_rng(BOOTSTRAP_SEED)),
            "K": A.tolist(), "B": B[:, 0].tolist(), "b": b.tolist(),
            "controllability": {k: v for k, v in controllability_diagnostics(A, B, HORIZON).items()
                                if k != "controllability_matrix"},
        }
        print(f"seed {seed}: CB={per_seed[seed]['main_element_CB']}")

    me = [per_seed[s]["main_element_CB"] for s in SEEDS]
    excludes0 = all(m["ci95"][0] > 0 or m["ci95"][1] < 0 for m in me)
    same_sign = len({np.sign(m["point"]) for m in me}) == 1
    stable = all(m["sign_stability"] >= SIGN_STABILITY_MIN for m in me)
    verdict = {"interval_excludes_zero_all_seeds": excludes0, "sign_agrees_across_seeds": same_sign,
               "sign_stability_ge_0.95_all_seeds": stable, "gate_pass": bool(excludes0 and same_sign)}
    args.out_dir.mkdir(parents=True)
    (args.out_dir / "summary.json").write_text(json.dumps(
        {"spec": "docs/experiments/command_channel_identifiability_plan.md#3.2",
         "lag": LAG, "horizon": HORIZON, "alpha": ALPHA, "n_resamples": N_RESAMPLES,
         "bootstrap_seed": BOOTSTRAP_SEED, "per_seed": per_seed, "verdict": verdict}, indent=2))
    print(json.dumps(verdict))


if __name__ == "__main__":
    main()
