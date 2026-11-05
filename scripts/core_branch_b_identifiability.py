#!/usr/bin/env python3
"""#4b v2 analysis: the core command effect from same-prefix forks.

Spec: docs/experiments/command_channel_identifiability_plan.md section 3.5.
Fixed before any data exists (2026-11-05).

ESTIMATOR. Each fork holds one prefix (same prompt, same replies 1..k-1, same
feedback template) fixed and changes only the number asked for. For siblings
j of one fork,

    y_k^(j) = K z_(k-1) + a_prompt + CB c^(j) + e^(j),

so within-fork deviations from the fork mean cancel K z and the prompt level:
    CB = sum_f sum_j dc dy / sum_f sum_j dc^2   (dc, dy demeaned within fork).
Resampling unit: trajectory (one fork each), 2000 resamples, default_rng(0).

RANK CHECK on all transitions, with `paper_side_experiments._identifiability`.
State z_t = [y_t, y_(t-1), r]: the column space of the headroom state
(output_memory = input_memory = 2, [y_t, y_(t-1), r-y_t, r-y_(t-1)]) without its
internal dependency r-y_(t-1) = (r-y_t) + y_t - y_(t-1), which alone would make
[Z|V|1] rank-deficient whatever the commands were. Command V = normalised ask
for reply t+1. Main-trajectory transitions ask for r every turn, so on them
alone V is a column of Z -- reported as `main_only`, expected unidentified.
Forks at k = 2 have no y_0 and enter the paired estimate only.

GATE (section 3.5, signed before collection): rank full and R^2(V on [Z|1])
< 0.99 on every seed, AND the paired CB interval excludes 0 on every seed,
AND the sign agrees across the three seeds.

Zero GPU; reads persona_drift_control/outputs/core_branch_sentence_length_<s>/,
writes results/core_branch_b_identifiability/ and refuses to overwrite.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "_paper_side_experiments", REPO_ROOT / "scripts" / "paper_side_experiments.py")
pse = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(pse)

SEEDS = (0, 1, 2)
N_RESAMPLES = 2000
BOOTSTRAP_SEED = 0
R2_MAX = 0.99
IN_TEMPLATE = "persona_drift_control/outputs/core_branch_sentence_length_{seed}/{name}.jsonl"
OUT_DIR = REPO_ROOT / "results" / "core_branch_b_identifiability"


def fork_units(points: list[dict]) -> list[tuple[np.ndarray, np.ndarray]]:
    """One (asks, replies) pair per fork, both normalised."""
    return [(np.array([b["command_norm"] for b in p["branches"]], dtype=float),
             np.array([b["normalized_output"] for b in p["branches"]], dtype=float)) for p in points]


def paired_cb(units: list[tuple[np.ndarray, np.ndarray]]) -> float:
    num = sum(float((c - c.mean()) @ (y - y.mean())) for c, y in units)
    den = sum(float((c - c.mean()) @ (c - c.mean())) for c, _ in units)
    return num / den


def paired_interval(units, rng) -> dict:
    point = paired_cb(units)
    n = len(units)
    draws = np.array([paired_cb([units[j] for j in rng.integers(0, n, n)]) for _ in range(N_RESAMPLES)])
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"point": point, "ci95": [float(lo), float(hi)],
            "sign_stability": float(np.mean(np.sign(draws) == np.sign(point))),
            "n_forks": n, "n_siblings": int(sum(len(c) for c, _ in units))}


def _z(y_t: float, y_tm1: float, r: float) -> list[float]:
    return [y_t, y_tm1, r]


def all_transitions(rows: list[dict], points: list[dict]) -> dict[str, np.ndarray]:
    by_traj: dict[str, dict[int, dict]] = {}
    for r in rows:
        by_traj.setdefault(r["trajectory_id"], {})[int(r["turn"])] = r
    Z, V, src = [], [], []
    for turns in by_traj.values():
        for t in sorted(turns):
            if t - 1 in turns and t + 1 in turns:
                y, r = turns[t]["normalized_output"], turns[t]["command_norm"]
                Z.append(_z(y, turns[t - 1]["normalized_output"], r))
                V.append(turns[t + 1]["command_norm"])
                src.append("main")
    for p in points:
        pre = p["prefix_state"]
        if pre["y_k_minus_2"] is None:
            continue
        z = _z(pre["y_k_minus_1"]["norm"], pre["y_k_minus_2"]["norm"], p["target_norm"])
        for b in p["branches"]:
            Z.append(z)
            V.append(b["command_norm"])
            src.append("branch")
    return {"Z": np.array(Z, dtype=float), "V": np.array(V, dtype=float)[:, None], "src": np.array(src)}


def identifiability(T: dict[str, np.ndarray]) -> dict:
    ident = pse._identifiability(T["Z"], T["V"])
    ident = {k: v for k, v in ident.items() if k != "singular_values"}
    ident["passes_rank_and_r2"] = bool(not ident["rank_deficient"] and max(ident["r_on_state_r2"]) < R2_MAX)
    ident["n_transitions"] = int(len(T["V"]))
    return ident


def subset(T: dict[str, np.ndarray], src: str) -> dict[str, np.ndarray]:
    m = T["src"] == src
    return {k: v[m] for k, v in T.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", type=pathlib.Path, default=OUT_DIR)
    args = ap.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"{args.out_dir} exists; refusing to overwrite")

    per_seed = {}
    for seed in SEEDS:
        def load(name: str) -> list[dict]:
            return [json.loads(l) for l in open(REPO_ROOT / IN_TEMPLATE.format(seed=seed, name=name))]
        rows, points = load("trajectories"), load("branch_points")
        if {r["command_policy"] for r in rows} | {p["command_policy"] for p in points} != {"branch"}:
            raise ValueError(f"seed {seed}: not a branch collection")
        if not all(p["disable_thinking"] for p in points):
            raise ValueError(f"seed {seed}: thinking was on")
        T = all_transitions(rows, points)
        per_seed[seed] = {
            "paired_CB": paired_interval(fork_units(points), np.random.default_rng(BOOTSTRAP_SEED)),
            "identifiability_all": identifiability(T),
            "identifiability_main_only": identifiability(subset(T, "main")),
            "git_sha": sorted({p["git_sha"] for p in points}),
        }
        print(f"seed {seed}: {per_seed[seed]['paired_CB']}")

    me = [per_seed[s]["paired_CB"] for s in SEEDS]
    ident_ok = all(per_seed[s]["identifiability_all"]["passes_rank_and_r2"] for s in SEEDS)
    excludes0 = all(m["ci95"][0] > 0 or m["ci95"][1] < 0 for m in me)
    same_sign = len({np.sign(m["point"]) for m in me}) == 1
    verdict = {"rank_full_and_r2_lt_0.99_all_seeds": ident_ok,
               "interval_excludes_zero_all_seeds": excludes0, "sign_agrees_across_seeds": same_sign,
               "gate_pass": bool(ident_ok and excludes0 and same_sign)}
    args.out_dir.mkdir(parents=True)
    (args.out_dir / "summary.json").write_text(json.dumps(
        {"spec": "docs/experiments/command_channel_identifiability_plan.md#3.5",
         "n_resamples": N_RESAMPLES, "bootstrap_seed": BOOTSTRAP_SEED, "r2_max": R2_MAX,
         "per_seed": per_seed, "verdict": verdict}, indent=2))
    print(json.dumps(verdict))


if __name__ == "__main__":
    main()
