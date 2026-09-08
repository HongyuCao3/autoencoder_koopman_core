#!/usr/bin/env python3
"""E3 Step 3 (docs/experiments/two_task_success_plan.md, section 2, ERGO
line): fits the state-action interaction regression

    c_{t+1} = a*c_t + b0*u_{t+1} + b2*u_{t+1}*c_t + g*shard_frac_{t+1} + const

on ERGO trajectories' `closeness` readout (F1/F2's de-thresholded state --
`scripts/fit_koopman_ergo_closeness.py`'s `Y_COL`). The question this is
for: whether reset's marginal effect on closeness depends on how close the
agent already is (`b2` is that dependence -- G-E3-2's gate is `b2`'s 95% CI
excluding 0 with a negative sign, i.e. reset matters less the closer you
already are).

`closeness` is not a column that exists on `trajectories.jsonl` rows -- it
is the extractor + gold-answer formula in
`scripts/fit_koopman_ergo_closeness.py::_add_derived_columns`, duplicated
here verbatim (`_add_closeness` below), matching that script's own
documented convention of duplicating this formula per analysis script
rather than importing it across scripts.

`u_{t+1}` alignment (this is the one thing this script must not get wrong --
see `scripts/analyze_ergo_readout_state.py`, `build_input_gain_transitions`'s
docstring, lines ~296-310, for the exact bug this guards against): for a
consecutive turn pair `(a, b)` with `b["turn"] == a["turn"] + 1`, `u_{t+1}`
is `b`'s OWN `u_reset` field -- the action recorded on the SAME row as the
`c_{t+1}` target, matching `contemporaneous_v=True` semantics (the action
that acts on turn t+1's outcome is turn t+1's own logged action). The
pre-fix version of `analyze_ergo_readout_state.py` instead zipped in a
THIRD row `c = traj[i+2]` and read `c["u_reset"]`, which is `u_{t+2}` -- one
turn past where it belongs. This script only ever zips two adjacent rows
(`zip(traj, traj[1:])`); there is no third row anywhere in this file.

Fit split: `modeling.dataset.split_by_system_prompt_id` on `item_id` --
the exact same call `fit_koopman_ergo_closeness.py` uses
(`held_out_frac=0.25`, `split_seed=0`; 60 items -> 45 train / 15 held out).
This regression fits (and bootstraps) ONLY on the 45 train items, matching
that script's arx/richer fits (which also fit on `train_rows` only, never
on `held_out_rows`). The 15 held-out items are loaded and counted for the
report but never enter this regression -- there is no held-out MSE here;
the point of reusing the split is to reuse the identical item partition,
not to validate on it.

Bootstrap: per the task spec, 1000 resamples, item-level (resample the
train item_ids with replacement, pool all their transition rows, refit
OLS each time), `np.random.default_rng(seed)`, percentile 95% CI -- the
same recipe `scripts/analyze_ergo_phaseC_comparison.py::_paired_bootstrap`
uses for paired-difference bootstraps, adapted here to a 5-coefficient OLS
fit instead of a scalar difference.

Missing/NaN `closeness`: a transition (a, b) is dropped (and counted) if
either `a["closeness"]` or `b["closeness"]` is `None` or NaN. In practice
`_add_closeness`'s own formula never produces NaN (it falls back to 0.0
when the answer can't be parsed) -- this guard exists for input rows that
already carry an explicit missing/NaN `closeness` value (e.g. a future
caller that pre-populates the column), and is exercised by a synthetic
test rather than by real Phase B data.

CPU-only, pure numpy. Run directly (no sbatch). Reads `--trajectories-dir`
only; never writes into it.

Usage:
    python scripts/analyze_ergo_state_action_interaction.py \\
        --trajectories-dir outputs/ergo_math_phaseB_random_excite \\
        --out-path /path/to/interaction_report.json
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from persona_drift.ergo_math_judge import _normalize, extract_answer_by_regex  # noqa: E402
from persona_drift.modeling.dataset import (  # noqa: E402
    group_by_trajectory,
    load_trajectories,
    split_by_system_prompt_id,
)

Y_COL = "closeness"
U_COL = "u_reset"
AUX_COL = "shard_frac"
COEF_NAMES = ("a", "b0", "b2", "g", "const")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--trajectories-dir",
        type=pathlib.Path,
        required=True,
        help="directory containing trajectories.jsonl (e.g. outputs/ergo_math_phaseB_random_excite); read-only.",
    )
    parser.add_argument("--held-out-frac", type=float, default=0.25)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--out-path", type=pathlib.Path, required=True, help="JSON report output path (must not already exist).")
    return parser.parse_args()


def _to_number(text: str | None) -> float | None:
    if text is None:
        return None
    try:
        return float(_normalize(text))
    except (ValueError, TypeError):
        return None


def _add_closeness(rows: list[dict]) -> list[dict]:
    """Verbatim copy of fit_koopman_ergo_closeness.py::_add_derived_columns
    (same per-domain-script duplication convention that file documents)."""
    for row in rows:
        row[AUX_COL] = row["turn"] / row["num_shards"]
        extracted = extract_answer_by_regex(row["agent_message"])
        a = _to_number(extracted) if extracted is not None else None
        g = _to_number(row.get("gold_answer"))
        row[Y_COL] = 0.0 if a is None or g is None else 1.0 / (1.0 + abs(a - g) / max(abs(g), 1.0))
    return rows


def _is_missing(value) -> bool:
    if value is None:
        return True
    try:
        return bool(math.isnan(float(value)))
    except (TypeError, ValueError):
        return True


def build_transitions(rows: list[dict]) -> tuple[list[dict], dict]:
    """Consecutive-turn transitions `(a, b)` with `b["turn"] == a["turn"] + 1`,
    `u_next` read off `b`'s OWN `u_reset` (see module docstring's alignment
    note). Drops (and counts) non-consecutive pairs and pairs with missing/
    NaN `closeness` on either side."""

    groups = group_by_trajectory(rows)
    transitions: list[dict] = []
    n_dropped_turn_gap = 0
    n_dropped_missing_closeness = 0
    for traj in groups.values():
        for a, b in zip(traj, traj[1:]):
            if b["turn"] != a["turn"] + 1:
                n_dropped_turn_gap += 1
                continue
            if _is_missing(a.get(Y_COL)) or _is_missing(b.get(Y_COL)):
                n_dropped_missing_closeness += 1
                continue
            transitions.append(
                {
                    "item_id": a["item_id"],
                    "c_t": float(a[Y_COL]),
                    "u_next": float(b[U_COL]),
                    "shard_frac_next": float(b[AUX_COL]),
                    "c_t1": float(b[Y_COL]),
                }
            )
    return transitions, {
        "n_dropped_turn_gap": n_dropped_turn_gap,
        "n_dropped_missing_closeness": n_dropped_missing_closeness,
    }


def _design_matrix(transitions: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Column order matches COEF_NAMES: [c_t, u_next, u_next*c_t, shard_frac_next, 1.0]."""
    X = np.array(
        [[t["c_t"], t["u_next"], t["u_next"] * t["c_t"], t["shard_frac_next"], 1.0] for t in transitions]
    )
    y = np.array([t["c_t1"] for t in transitions])
    return X, y


def _ols_beta(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def _bootstrap_by_item(transitions: list[dict], n_resamples: int, seed: int) -> np.ndarray:
    by_item: dict[str, list[dict]] = {}
    for t in transitions:
        by_item.setdefault(t["item_id"], []).append(t)
    item_ids = sorted(by_item)
    rng = np.random.default_rng(seed)
    betas = np.zeros((n_resamples, len(COEF_NAMES)))
    for i in range(n_resamples):
        sampled_items = rng.choice(item_ids, size=len(item_ids), replace=True)
        resampled: list[dict] = []
        for item in sampled_items:
            resampled.extend(by_item[item])
        X, y = _design_matrix(resampled)
        betas[i] = _ols_beta(X, y)
    return betas


def fit_interaction_model(transitions: list[dict], n_bootstrap: int, bootstrap_seed: int) -> dict:
    X, y = _design_matrix(transitions)
    point_beta = _ols_beta(X, y)
    boot_betas = _bootstrap_by_item(transitions, n_resamples=n_bootstrap, seed=bootstrap_seed)
    coefficients = {}
    for i, name in enumerate(COEF_NAMES):
        coefficients[name] = {
            "point_estimate": float(point_beta[i]),
            "ci_low": float(np.percentile(boot_betas[:, i], 2.5)),
            "ci_high": float(np.percentile(boot_betas[:, i], 97.5)),
        }
    return {
        "coefficients": coefficients,
        "n_pairs": len(transitions),
        "n_items": len({t["item_id"] for t in transitions}),
        "n_bootstrap": n_bootstrap,
        "bootstrap_seed": bootstrap_seed,
    }


def main() -> None:
    args = parse_args()
    trajectories_path = args.trajectories_dir / "trajectories.jsonl"
    rows = _add_closeness(load_trajectories(trajectories_path))

    split = split_by_system_prompt_id(
        rows,
        train_frac=1.0 - args.held_out_frac,
        val_frac=0.0,
        seed=args.split_seed,
        split_col="item_id",
    )
    train_rows = split["train"]
    held_out_rows = split["test"]
    n_train_items = len({r["item_id"] for r in train_rows})
    n_held_out_items = len({r["item_id"] for r in held_out_rows})

    transitions, drop_info = build_transitions(train_rows)
    if not transitions:
        raise SystemExit("no transitions survived filtering -- cannot fit the interaction model")

    fit = fit_interaction_model(transitions, n_bootstrap=args.n_bootstrap, bootstrap_seed=args.bootstrap_seed)

    report = {
        "config": {
            "trajectories_dir": str(args.trajectories_dir),
            "held_out_frac": args.held_out_frac,
            "split_seed": args.split_seed,
            "n_bootstrap": args.n_bootstrap,
            "bootstrap_seed": args.bootstrap_seed,
            "y_col": Y_COL,
            "u_col": U_COL,
            "aux_col": AUX_COL,
            "formula": "c_t1 ~ a*c_t + b0*u_next + b2*u_next*c_t + g*shard_frac_next + const",
        },
        "n_train_items": n_train_items,
        "n_held_out_items": n_held_out_items,
        "n_rows_train": len(train_rows),
        "n_rows_held_out": len(held_out_rows),
        **drop_info,
        **fit,
    }

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.out_path.exists():
        raise SystemExit(f"refusing to overwrite existing {args.out_path}")
    args.out_path.write_text(json.dumps(report, indent=2))

    print(f"n_train_items={n_train_items} n_held_out_items={n_held_out_items}")
    print(f"n_pairs={fit['n_pairs']} n_items_in_fit={fit['n_items']}")
    print(f"dropped: {drop_info}")
    for name in COEF_NAMES:
        c = fit["coefficients"][name]
        print(f"{name}: point={c['point_estimate']:+.6f}  CI=[{c['ci_low']:+.6f}, {c['ci_high']:+.6f}]")
    print(f"report written to {args.out_path}")


if __name__ == "__main__":
    main()
