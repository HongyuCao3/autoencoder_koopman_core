#!/usr/bin/env python3
"""Verdict for the upstream-fidelity arm (2026-09-09). CPU only.

Reads one scored readout covering all four variants and answers ONE question:
does any harness variant bring our retention curve down to the benchmark's?

PRE-REGISTERED (before the arm ran; ledger + screening doc section 10):

  A variant CLOSES THE GAP if
    (a) its mean upstream-comparable retention over t1..t20 is at least 0.10
        BELOW the baseline variant (`turn1_greedy`, i.e. what the S0-0 arm
        did), with an item-paired bootstrap 95% CI on the difference that
        excludes 0, AND
    (b) its turn-1 retention is <= 0.60 (upstream reads ~0.50 for this model
        id on tuples/3).

  Pass  -> re-run the S0-0 branch arm under that variant, recompute K1/K2/K3.
  Fail (no variant) -> the readout has no range on this task with this model;
        the pre-registered closure of the line stands.

The readout compared is `y_binary` (all k constraints held), because that is
upstream's own aggregation -- the graded `y` is this project's choice and has
no published counterpart. The graded curve is reported alongside, never as the
fidelity criterion.

Pairing is by item: every variant ran the SAME 12 dialogues with the same
turns, so the difference is measured within item and the between-item spread
(which is large -- constraint sets differ wildly) never enters the comparison.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics

import numpy as np

N_BOOTSTRAP = 10000
GAP_MARGIN = 0.10       # (a) how far below the baseline a variant must land
TURN1_CEILING = 0.60    # (b) upstream reads ~0.50 at turn 1 on tuples/3
BASELINE = "turn1_greedy"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--readout", type=pathlib.Path, required=True,
                   help="score_sequor_trajectories.py report over the fidelity arm.")
    p.add_argument("--out-path", type=pathlib.Path, required=True, help="Must not already exist.")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def curve(rows: list[dict], field: str) -> dict[int, float]:
    by_turn: dict[int, list[float]] = {}
    for row in rows:
        if row[field] is not None:
            by_turn.setdefault(row["turn"], []).append(float(row[field]))
    return {turn: statistics.fmean(vals) for turn, vals in sorted(by_turn.items())}


def item_means(rows: list[dict], field: str) -> dict[str, float]:
    by_item: dict[str, list[float]] = {}
    for row in rows:
        if row[field] is not None:
            by_item.setdefault(row["item_id"], []).append(float(row[field]))
    return {item: statistics.fmean(vals) for item, vals in by_item.items()}


def paired_difference(variant: dict[str, float], baseline: dict[str, float], seed: int) -> dict:
    """Item-paired difference (variant - baseline) with a bootstrap CI over items."""

    shared = sorted(set(variant) & set(baseline))
    diffs = np.array([variant[i] - baseline[i] for i in shared])
    rng = np.random.default_rng(seed)
    draws = [
        float(np.mean(diffs[rng.integers(0, len(diffs), len(diffs))]))
        for _ in range(N_BOOTSTRAP)
    ]
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {
        "n_items": len(shared), "mean_difference": float(np.mean(diffs)),
        "ci": [float(lo), float(hi)], "bootstrap_sd": float(np.std(draws, ddof=1)),
        "per_item": {i: variant[i] - baseline[i] for i in shared},
    }


def main() -> None:
    args = parse_args()
    if args.out_path.exists():
        raise SystemExit(f"refusing to overwrite existing {args.out_path}")
    report = json.loads(args.readout.read_text())
    if report.get("mode") == "debug":
        raise SystemExit(f"{args.readout}: arm mode=debug; debug artifacts are not evidence")
    if report["judge_kind"] != "independent":
        raise SystemExit(f"{args.readout}: judge_kind={report['judge_kind']!r}; the fidelity "
                         f"comparison to upstream's published curve needs the reporting judge")

    rows_by_variant: dict[str, list[dict]] = {}
    for row in report["rows"]:
        variant = row.get("variant")
        if variant is None:
            raise SystemExit("readout rows carry no `variant`: this is not a fidelity arm")
        rows_by_variant.setdefault(variant, []).append(row)
    if BASELINE not in rows_by_variant:
        raise SystemExit(f"the baseline variant {BASELINE!r} is absent; nothing to pair against")

    baseline_items = item_means(rows_by_variant[BASELINE], "y_binary")
    out = {
        "readout": str(args.readout), "judge_model": report["judge_model"],
        "criterion": {
            "gap_margin_below_baseline": GAP_MARGIN, "turn1_ceiling": TURN1_CEILING,
            "baseline_variant": BASELINE,
            "upstream_reference": "tuples/3, Qwen3-4B-Instruct-2507, read off upstream's figure: "
                                  "~0.50 at turn 1, <0.12 by turn 50",
        },
        "variants": {},
    }

    for variant, rows in sorted(rows_by_variant.items()):
        binary = curve(rows, "y_binary")
        graded = curve(rows, "y_graded")
        overall = statistics.fmean([v for v in binary.values()])
        turn1 = binary.get(1)
        record = {
            "n_rows": len(rows),
            "mean_y_binary_over_turns": overall,
            "y_binary_turn1": turn1, "y_binary_turn20": binary.get(max(binary)),
            "y_binary_by_turn": binary, "y_graded_by_turn": graded,
            "mean_y_graded_over_turns": statistics.fmean([v for v in graded.values()]),
            "n_rows_unusable": sum(1 for r in rows if r["y_graded"] is None),
        }
        if variant != BASELINE:
            diff = paired_difference(item_means(rows, "y_binary"), baseline_items, args.seed)
            record["paired_vs_baseline"] = diff
            record["closes_gap"] = bool(
                diff["mean_difference"] <= -GAP_MARGIN
                and diff["ci"][1] < 0
                and turn1 is not None and turn1 <= TURN1_CEILING
            )
        out["variants"][variant] = record

    closers = [v for v, r in out["variants"].items() if r.get("closes_gap")]
    out["variants_closing_the_gap"] = closers
    out["verdict"] = "GAP CLOSED" if closers else "GAP NOT CLOSED"

    print(f"judge {report['judge_model']}  (upstream-comparable readout: all k constraints held)\n")
    print(f"{'variant':16s} {'mean':>7s} {'t1':>6s} {'t20':>6s}   paired vs {BASELINE}")
    for variant, r in out["variants"].items():
        line = (f"{variant:16s} {r['mean_y_binary_over_turns']:7.3f} {r['y_binary_turn1']:6.2f} "
                f"{r['y_binary_turn20']:6.2f}")
        if "paired_vs_baseline" in r:
            d = r["paired_vs_baseline"]
            line += (f"   {d['mean_difference']:+.4f}  CI [{d['ci'][0]:+.4f}, {d['ci'][1]:+.4f}]"
                     f"  {'CLOSES' if r['closes_gap'] else ''}")
        print(line)
    print(f"\nverdict: {out['verdict']}"
          + (f"  ({', '.join(closers)})" if closers else ""))
    if closers:
        print("-> re-run the S0-0 branch arm under that variant and recompute K1/K2/K3.")
    else:
        print("-> no harness variant we control brings the curve to upstream's range: the readout "
              "has no range on this task with this model, and closing the line stands.")

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nreport written to {args.out_path}")


if __name__ == "__main__":
    main()
