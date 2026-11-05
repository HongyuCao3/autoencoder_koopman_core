#!/usr/bin/env python3
"""Reachable reply length under a bounded word-count request (#4b follow-up).

Exploratory, no gate (docs/experiments/core_excite_b_identifiability_results.md,
section "探索性补充"). Fixes the in-session computation of 2026-11-05 as code.

Per seed, on the random-command collection (job 19443006):
  * fit z_(t+1) = K z_t + B c_(t+1) + b on the `train` topics (lag=3, same
    fit as core_excite_b_identifiability.py);
  * steady-state reachable mean length for a held request c in [0, 1]:
    y* = (CB c + b0) / (1 - sum K[0, :]), reported in words (5 + 95 y);
  * held-out check: one-step predictions on the `test` topics against the
    observed next reading, mean per request bin, in words;
  * empirical delivery per request bin, all seeds' turns >= 2.

Zero GPU; writes results/core_excite_reachability/ and refuses to overwrite.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "_core_excite_b", REPO_ROOT / "scripts" / "core_excite_b_identifiability.py")
ident = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ident)

BINS = ((5, 30), (30, 60), (60, 80), (80, 101))
OUT_DIR = REPO_ROOT / "results" / "core_excite_reachability"


def words(y):
    return 5.0 + 95.0 * np.asarray(y, dtype=float)


def steady_state_range(K: np.ndarray, B: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    gain = 1.0 - float(K[0].sum())
    return float(words(b[0] / gain)), float(words((B[0, 0] + b[0]) / gain))


def binned(requested_words: np.ndarray, values_words: np.ndarray) -> dict[str, dict]:
    out = {}
    for lo, hi in BINS:
        m = (requested_words >= lo) & (requested_words < hi)
        out[f"{lo}-{hi - 1}"] = {"mean": float(values_words[m].mean()), "n": int(m.sum())}
    return out


def seed_report(rows: list[dict]) -> dict:
    split = {r["trajectory_id"]: r["topic_split"] for r in rows}
    units = ident.transitions(rows, ident.LAG)
    train = [u for t, u in units.items() if split[t] == "train"]
    test = [u for t, u in units.items() if split[t] == "test"]
    K, B, b = ident.fit(train)
    Z = np.vstack([u[0] for u in test])
    c = np.concatenate([u[1] for u in test])
    y_next = np.vstack([u[2] for u in test])[:, 0]
    pred = Z @ K[0] + B[0, 0] * c + b[0]
    lo, hi = steady_state_range(K, B, b)
    req = words(c)
    late = [r for r in rows if r["turn"] >= 2]
    return {
        "n_train_trajectories": len(train), "n_test_trajectories": len(test),
        "train_fit": {"CB": float(B[0, 0]), "b0": float(b[0]), "sum_K_row0": float(K[0].sum())},
        "steady_state_words": {"c=0 (5 words)": lo, "c=1 (100 words)": hi},
        "heldout_one_step": {"predicted": binned(req, words(pred)), "observed": binned(req, words(y_next))},
        "empirical_delivery_turn_ge2": binned(
            np.array([r["command_raw_count"] for r in late], dtype=float),
            np.array([r["measured_raw_count"] for r in late], dtype=float)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", type=pathlib.Path, default=OUT_DIR)
    args = ap.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"{args.out_dir} exists; refusing to overwrite")
    per_seed = {}
    for seed in ident.SEEDS:
        rows = [json.loads(l) for l in open(REPO_ROOT / ident.IN_TEMPLATE.format(seed=seed))]
        if {r["command_policy"] for r in rows} != {"random"}:
            raise ValueError(f"seed {seed}: not a random-command collection")
        per_seed[seed] = seed_report(rows)

    def agg(get):
        v = np.array([get(per_seed[s]) for s in ident.SEEDS])
        return {"mean": float(v.mean()), "std": float(v.std(ddof=1)), "n_seeds": len(v)}

    summary = {
        "ceiling_words_c1": agg(lambda r: r["steady_state_words"]["c=1 (100 words)"]),
        **{f"heldout_{kind}_{k}": agg(lambda r, k=k, kind=kind: r["heldout_one_step"][kind][k]["mean"])
           for kind in ("predicted", "observed") for k in per_seed[0]["heldout_one_step"]["predicted"]},
        **{f"delivered_{k}": agg(lambda r, k=k: r["empirical_delivery_turn_ge2"][k]["mean"])
           for k in per_seed[0]["empirical_delivery_turn_ge2"]},
    }
    args.out_dir.mkdir(parents=True)
    (args.out_dir / "summary.json").write_text(json.dumps(
        {"lag": ident.LAG, "alpha": ident.ALPHA, "bins_words": BINS,
         "per_seed": per_seed, "across_seeds": summary}, indent=2))
    for k, v in summary.items():
        print(f"{k}: {v['mean']:.1f} ± {v['std']:.1f} (n={v['n_seeds']})")


if __name__ == "__main__":
    main()
