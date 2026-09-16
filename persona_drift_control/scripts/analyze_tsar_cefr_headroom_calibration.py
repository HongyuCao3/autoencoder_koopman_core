#!/usr/bin/env python3
"""Candidate D: calibrate the planning-headroom criterion on the real tree.

Zero GPU. Reads the 67,660 rows of GPU-2a's depth-4 tree and re-scores them
under a cost that carries a PLANTED, KNOWN-SIZE step-dependent preference. No
model is loaded, no text is generated, nothing already on disk is overwritten.

WHY THIS EXISTS, AND WHAT IT IS NOT.

Candidate B's probe reported, on the CONSTANT schedule, a leave-one-out
G_plan of -0.0217 with a 95% CI of [-0.0450, -0.0011] -- negative, and
excluding zero -- on a problem whose true headroom is believed to be near zero
(D-2.5 puts the closed-loop supremum 0.0130 above the hand-written rule on the
terminal metric). A CI excluding zero on the wrong side of a near-zero truth is
the signature of a BIASED estimator, not of a small effect, and the suspected
bias (about 0.02) is the same size as the effects the criterion is asked to
detect. Every null this project has reported with this instrument rests on the
estimator not having that bias.

So this script does NOT ask "can the criterion ever read non-zero" -- the unit
test `test_detects_planted_gap` already answers that on a fixture, and the
2xMDE detection threshold is fixed by construction (MDE is computed, not
discovered). It asks:

    when the TRUE headroom is delta, what does the LOO estimator report?

and reports the gap between the two as the bias. That number is what tells a
reader how to read the five nulls.

THE PLANTED TERM IS SYMMETRIC BETWEEN THE TWO CLASSES. It is a function of
(step, decision bin, action) and is added to the one cost array both classes
are scored on. The myopic class cannot follow it across steps because it is
step-invariant BY DEFINITION; the planting takes nothing away from it that it
could otherwise have used. `pistar` cycles the preferred action with the step
index so that no step-invariant table can match it at every step.

TRUTH IS EXACT, NOT ESTIMATED. Both classes are solved over the WHOLE
population (all cells, all sources) by the same exact solvers the probe uses --
brute force over 4^3 step-invariant tables, prefix enumeration for the
step-dependent class. That population optimum IS the true G_plan at this delta;
it is not a fit that has to generalise. The LOO column beside it is the
estimator under test.

CALIBER. Same as the probe: this is a GATE verdict. Uncertainty is over sources
(bootstrap by `source_id`), not over seeds, so per `.claude/global.md` it may
not be quoted as a cross-arm headline in mean +- std (n) form. The detection
threshold read off this curve is specific to this tree, this sample size and
this policy class, and does not transfer to the other ten columns.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from persona_drift.run_provenance import provenance  # noqa: E402
from analyze_tsar_cefr_d25 import (  # noqa: E402
    BIN_SCHEMES,
    N_ACTIONS,
    load_tree,
)
from analyze_tsar_cefr_moving_reference import (  # noqa: E402
    SCHEDULES,
    MovingRefGroup,
    enumerate_prefixes_running,
    plant_step_dependent_cost,
    run_schedule,
    solve_anticipating,
    solve_myopic,
)

DELTA_GRID = (0.0, 0.01, 0.02, 0.05, 0.10, 0.20)
SCHEDULE = "constant"


def pistar(t: int, b: int) -> int:
    """Preferred action at step `t` (1-based) for bin `b`.

    The step index enters through the shift, so the preferred action for a given
    bin differs across steps and no step-invariant table can follow it
    everywhere. That is the whole point: the planted headroom has to be visible
    only to a class that can see `t`.
    """
    return (b + t - 1) % N_ACTIONS


def true_headroom(cells_by_target, delta: float, nbins: int) -> dict:
    """Population optimum of both classes at this delta. Exact, not fitted."""
    out = {}
    for target, cells in sorted(cells_by_target.items()):
        g = MovingRefGroup(cells, nbins, SCHEDULES[SCHEDULE])
        if delta:
            plant_step_dependent_cost(g, delta, pistar)
        all_idx = np.arange(g.n)
        pos3, dec = enumerate_prefixes_running(g)
        _, j_my = solve_myopic(g, all_idx)
        _, j_an = solve_anticipating(g, pos3, dec, all_idx)
        out[target] = {"myopic": j_my, "anticipating": j_an, "G_plan": j_my - j_an,
                       "n_cells": g.n}
    total = sum(v["n_cells"] for v in out.values())
    out["pooled_G_plan"] = sum(v["G_plan"] * v["n_cells"] for k, v in out.items()
                               if k != "pooled_G_plan") / total
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tree", type=pathlib.Path,
                    default=pathlib.Path("outputs/tsar_cefr_gpu2a_branch/tree.jsonl"))
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--bins", type=int, default=3, choices=sorted(BIN_SCHEMES))
    ap.add_argument("--deltas", type=float, nargs="+", default=list(DELTA_GRID))
    args = ap.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")

    cells = load_tree(args.tree)
    by_target = collections.defaultdict(list)
    for c in cells.values():
        by_target[c.target].append(c)

    report = {
        "provenance": provenance(),
        "tree": str(args.tree),
        "schedule": SCHEDULE,
        "planted_policy": "pistar(t, b) = (b + t - 1) mod 4, step-dependent by construction",
        "metric": "J = mean_t (level_expected_t - r_t)^2 + delta * 1[a_t != pistar(t, bin_t)]",
        "question": ("what does the LOO estimator report when the TRUE headroom is a known "
                     "size; the gap between the two columns is the bias"),
        "not_asking": ("whether the criterion can ever read non-zero (the fixture unit test "
                       "answers that), nor where the 2xMDE threshold sits (MDE is computed)"),
        "reporting_caliber": ("gate verdict; uncertainty over sources, not seeds. Per "
                              ".claude/global.md may not be quoted as a cross-arm headline. "
                              "The threshold read off this curve is specific to this tree, "
                              "this sample size and this policy class"),
        "bins": args.bins,
        "cells": len(cells),
        "rows": [],
    }
    for delta in args.deltas:
        truth = true_headroom(by_target, delta, args.bins)
        plant = None if delta == 0.0 else (delta, pistar)
        est = run_schedule(by_target, SCHEDULE, args.bins, verify=False, plant=plant)
        loo = est["loo"]
        row = {
            "delta": delta,
            "true_G_plan": truth["pooled_G_plan"],
            "true_per_target": {k: v for k, v in truth.items() if k != "pooled_G_plan"},
            "loo_G_plan": loo["G_plan"],
            "ci95": loo["ci95"],
            "mde": loo["mde"],
            "bias": loo["G_plan"] - truth["pooled_G_plan"],
            "fires": loo["passes_2x_mde"],
        }
        report["rows"].append(row)
        print(f"delta={delta:<5} true={row['true_G_plan']:+.4f} "
              f"loo={row['loo_G_plan']:+.4f} bias={row['bias']:+.4f} "
              f"CI=[{loo['ci95'][0]:+.4f},{loo['ci95'][1]:+.4f}] "
              f"MDE={loo['mde']:.4f} {'FIRES' if row['fires'] else '-'}", flush=True)

    # The curve is read against TRUE headroom, never against delta. J_myopic and
    # J_anticipating are each a min over tables, hence concave in delta, and a
    # difference of two concave functions need not be monotone -- the fixture test
    # `test_true_headroom_grows_with_delta` caught exactly that. delta is the knob;
    # the exactly-solved population gap is the axis.
    by_truth = sorted(report["rows"], key=lambda r: r["true_G_plan"])
    fired = [r["true_G_plan"] for r in by_truth if r["fires"]]
    report["verdict"] = {
        "detection_threshold_true_headroom": min(fired) if fired else None,
        "bias_at_smallest_truth": by_truth[0]["bias"],
        "true_headroom_at_zero_delta": next(
            (r["true_G_plan"] for r in report["rows"] if r["delta"] == 0.0), None),
        "monotone_in_true_headroom": all(
            by_truth[i]["loo_G_plan"] <= by_truth[i + 1]["loo_G_plan"] + 1e-12
            for i in range(len(by_truth) - 1)),
        "curve": [[r["true_G_plan"], r["loo_G_plan"], r["fires"]] for r in by_truth],
        "reading": ("a detection threshold with a curve monotone in TRUE headroom calibrates "
                    "the five nulls ('headroom below this'); a non-monotone curve means the "
                    "estimator is not calibratable at this sample size, and per the gate's "
                    "second row no threshold may be reported"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["verdict"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
