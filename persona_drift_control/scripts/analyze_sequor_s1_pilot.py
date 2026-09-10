#!/usr/bin/env python3
"""S1 sizing from the pilot arm (job 15768661). CPU only, no verdict.

THE QUESTION: what is S1a's MDE, and therefore what N does S1 need?

Plan section 4 sized S1a on an ASSUMPTION (per-item difference sd about 0.30 ->
MDE about 0.13 at n=40) and required that it be recomputed from a real
SUSTAINED arm before submitting. The pilot is that arm. This script does the
recomputation, and nothing else:

  1. the S1a contrast     per-item, paired: mean `y` under `constant_remind`
                          minus mean `y` under `zero_control`, over a stated
                          window of late turns;
  2. variance components  sd ACROSS items and sd ACROSS seeds within an item,
                          separately -- they enter the MDE differently, and
                          only the second one is bought by adding seeds;
  3. the sizing table     MDE(n items, s seeds) and the n required to resolve
                          a stated effect, so the N ruling is arithmetic;
  4. late-turn range      per arm, per turn: does `y` still move at turn 20
                          under a sustained reminder, and what share sits on
                          the ceiling -- the branch arm cannot answer this;
  5. G-S1 replay          the plan's admission checks (u balance, one reminded
                          side per cell, per-turn spread) on real rows.

WHAT THIS IS NOT: not a gate, and not a result. n=12 against a plan written for
n=40, and no K gate reads it. The pre-registered consequence of a small pilot
effect is to re-size S1 -- never to conclude the executor lacks authority,
which K3 already answered on one-step pairs (+0.1233, CI [+0.0595, +0.1898]).
The contrast printed here is a VARIANCE ESTIMATE that happens to have a mean.

Two windows are reported side by side (`late_window` and `turn_20_only`)
because the plan says "endpoint" without saying which functional, and they
imply materially different MDEs. Picking the smaller one here would be
choosing the estimand after seeing the variance. S1 must pre-register one.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import pathlib
import statistics

import numpy as np

POWER_Z = 2.80  # two-sided alpha=0.05 at 80% power, as in the S0-0 gates
N_BOOTSTRAP = 10000
MIN_DISTINCT_VALUES = 3  # plan section 4's G-S1 clause on per-turn spread
SIZING_N_ITEMS = (12, 16, 24, 40)
SIZING_N_SEEDS = (1, 3, 5)


def _gates_module():
    """The response-cap guard lives in the S0-0 gate script. Import it rather
    than restating the criterion: two copies of a threshold drift apart, and
    the guard exists precisely so an arm whose cap bound cannot be analysed
    without the exclusion being named in the artifact."""

    path = pathlib.Path(__file__).with_name("analyze_sequor_s0_0_gates.py")
    spec = importlib.util.spec_from_file_location("_s0_0_gates", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--independent-readout", type=pathlib.Path, required=True,
                   help="score_sequor_trajectories.py report with judge_kind=independent. "
                        "The reporting readout is the only one any number may come from.")
    p.add_argument("--out-path", type=pathlib.Path, required=True, help="Must not already exist.")
    p.add_argument("--late-from", type=int, default=15,
                   help="First turn of the late window (inclusive).")
    p.add_argument("--target-effect", type=float, default=0.10,
                   help="The effect S1 must be able to resolve, for the required-n column.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--exclude-item", action="append", default=[], metavar="ITEM_ID",
                   help="Drop this item, the only way past the response-cap guard. Sizing is "
                        "reported both with and without the exclusion, since an exclusion that "
                        "moves the MDE must be visible next to it.")
    return p.parse_args()


def per_item_contrast(rows: list[dict], treated: str, control: str, turns: range) -> dict:
    """`d[item][seed]` = mean `y` under `treated` minus mean `y` under
    `control`, over `turns`. Paired within (item, seed): under sampled decoding
    a pair across seeds compares two trajectories rather than one policy."""

    means: dict[tuple[str, str, int], list[float]] = {}
    for row in rows:
        if row["y_graded"] is None or row["turn"] not in turns:
            continue
        means.setdefault((row["branch"], row["item_id"], row["seed"]), []).append(row["y_graded"])
    out: dict[str, dict[int, float]] = {}
    for (branch, item, seed), values in means.items():
        if branch != treated:
            continue
        other = means.get((control, item, seed))
        if not other:
            continue
        out.setdefault(item, {})[seed] = statistics.mean(values) - statistics.mean(other)
    return out


def variance_components(contrast: dict[str, dict[int, float]]) -> dict:
    """Split the per-item difference variance into a between-item part and a
    within-item across-seed part.

    They are not interchangeable. Adding SEEDS shrinks only the second, by 1/s;
    adding ITEMS shrinks the whole thing by 1/n. Reporting a single pooled sd
    would let someone buy seeds expecting an MDE that only items can deliver --
    the same class of error as ERGO's one-step MDE excusing an endpoint gate.
    """

    per_item_means = {item: statistics.mean(by_seed.values()) for item, by_seed in contrast.items()}
    n_seeds = {len(by_seed) for by_seed in contrast.values()}
    within = [statistics.variance(list(by_seed.values()))
              for by_seed in contrast.values() if len(by_seed) > 1]
    var_seed = statistics.mean(within) if within else 0.0
    var_of_item_means = statistics.variance(list(per_item_means.values()))
    s = min(n_seeds) if n_seeds else 1
    var_item = var_of_item_means - var_seed / s if s else var_of_item_means
    return {
        "n_items": len(per_item_means),
        "n_seeds_per_item": sorted(n_seeds),
        "mean_contrast": statistics.mean(per_item_means.values()),
        "sd_of_item_means": math.sqrt(var_of_item_means),
        "sd_between_items": math.sqrt(var_item) if var_item > 0 else 0.0,
        "sd_within_item_across_seeds": math.sqrt(var_seed),
        "between_item_variance_is_negative": var_item <= 0,
        "per_item_means": per_item_means,
        "per_seed_means": {
            str(seed): statistics.mean([by_seed[seed] for by_seed in contrast.values() if seed in by_seed])
            for seed in sorted({s for by_seed in contrast.values() for s in by_seed})
        },
    }


def sizing_table(components: dict, target_effect: float) -> dict:
    """MDE(n, s) = z * sqrt((var_item + var_seed / s) / n), and the n that
    resolves `target_effect` at each s."""

    var_item = components["sd_between_items"] ** 2
    var_seed = components["sd_within_item_across_seeds"] ** 2
    mde = {}
    required_n = {}
    for s in SIZING_N_SEEDS:
        per_item_var = var_item + var_seed / s
        mde[f"s{s}"] = {f"n{n}": POWER_Z * math.sqrt(per_item_var / n) for n in SIZING_N_ITEMS}
        required_n[f"s{s}"] = math.ceil(POWER_Z ** 2 * per_item_var / target_effect ** 2)
    return {"target_effect": target_effect, "mde_by_n_and_seeds": mde, "required_n_items": required_n,
            "power_z": POWER_Z,
            "note": "Adding seeds shrinks only the within-item term. If sd_between_items dominates, "
                    "more seeds buy almost nothing and only N moves the MDE."}


def bootstrap_contrast(contrast: dict[str, dict[int, float]], seed: int) -> dict:
    """Resample ITEMS with replacement: a dialogue is the independent unit."""

    rng = np.random.default_rng(seed)
    items = sorted(contrast)
    values = [statistics.mean(contrast[item].values()) for item in items]
    draws = [float(np.mean(rng.choice(values, size=len(values), replace=True)))
             for _ in range(N_BOOTSTRAP)]
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"point": float(np.mean(values)), "ci": [float(lo), float(hi)],
            "bootstrap_sd": float(np.std(draws, ddof=1)), "n_items": len(items)}


def range_by_turn(rows: list[dict], arms: list[str]) -> dict:
    """Per arm, per turn: spread, distinct values, ceiling and floor share.

    The ceiling share is the number that matters for S3: a controller reading a
    state that sits at 1.0 has nothing to feed back, and the S0-0 stratification
    already showed the one-step gain collapsing to +0.0026 in that layer.
    """

    grouped: dict[tuple[str, int], list[float]] = {}
    for row in rows:
        if row["y_graded"] is None:
            continue
        grouped.setdefault((row["branch"], row["turn"]), []).append(row["y_graded"])
    out = {}
    for arm in arms:
        turns = sorted(t for (a, t) in grouped if a == arm)
        per_turn = {}
        for turn in turns:
            values = grouped[(arm, turn)]
            per_turn[str(turn)] = {
                "n": len(values), "mean": statistics.mean(values),
                "sd": statistics.pstdev(values), "n_distinct": len(set(values)),
                "ceiling_share": sum(1 for v in values if v == 1.0) / len(values),
                "floor_share": sum(1 for v in values if v == 0.0) / len(values),
            }
        failing = [t for t, rec in per_turn.items()
                   if rec["sd"] == 0 or rec["n_distinct"] < MIN_DISTINCT_VALUES]
        out[arm] = {"by_turn": per_turn, "turns_failing_g_s1_spread": failing}
    return out


def g_s1_replay(arm_report: dict, rows: list[dict], arms: list[str]) -> dict:
    """The plan's G-S1 admission checks, recomputed from the rows that exist.

    On the pilot this is a check of the CRITERION, not of S1: the balance and
    cell clauses are properties of the schedule and carry over, the per-turn
    spread clause is measured at n=12 and will be re-measured at S1's N.
    """

    item_sets = {arm: sorted({r["item_id"] for r in rows if r["branch"] == arm}) for arm in arms}
    reference = item_sets[arms[0]]
    spread = range_by_turn(rows, arms)
    return {
        "item_sets_identical_across_arms": all(v == reference for v in item_sets.values()),
        "schedule_checks": arm_report["schedule_checks"],
        "u_mean_in_band": {
            seed: 0.45 <= check["bernoulli_u_mean"] <= 0.55
            for seed, check in arm_report["schedule_checks"].items()
        },
        "cells_are_matched": arm_report["cell_coverage"]["cells_are_matched"],
        "n_cells_with_exactly_one_reminder": arm_report["cell_coverage"]["n_cells_with_exactly_one_reminder"],
        "turns_failing_spread_by_arm": {arm: spread[arm]["turns_failing_g_s1_spread"] for arm in arms},
        "gold_coverage": arm_report["gold_coverage"],
    }


def sizing_for(rows: list[dict], late_from: int, n_turns: int, target_effect: float, seed: int) -> dict:
    windows = {
        "late_window": range(late_from, n_turns + 1),
        "turn_20_only": range(n_turns, n_turns + 1),
    }
    out = {}
    for name, turns in windows.items():
        contrast = per_item_contrast(rows, "constant_remind", "zero_control", turns)
        if not contrast:
            raise SystemExit(f"{name}: no paired items; the arms do not share an item set")
        components = variance_components(contrast)
        out[name] = {
            "turns": [min(turns), max(turns)],
            "contrast": bootstrap_contrast(contrast, seed),
            "variance_components": components,
            "sizing": sizing_table(components, target_effect),
        }
    return out


def main() -> None:
    args = parse_args()
    if args.out_path.exists():
        raise SystemExit(f"refusing to overwrite existing {args.out_path}")
    gates = _gates_module()
    readout = gates.load_readout(args.independent_readout, "independent")
    arm_dir = pathlib.Path(readout["arm_dir"])
    arm_report = json.loads((arm_dir / "arm_report.json").read_text())
    cap_record = gates.refuse_if_the_cap_bound(arm_report, args.exclude_item)

    arms = list(arm_report["arms"])
    n_turns = arm_report["n_turns"]
    all_rows = readout["rows"]
    kept = [r for r in all_rows if r["item_id"] not in args.exclude_item]

    report = {
        "arm_dir": str(arm_dir), "agent_model": readout["agent_model"],
        "judge_model": readout["judge_model"], "judge_kind": readout["judge_kind"],
        "n_rows": len(kept), "arms": arms, "n_turns": n_turns,
        "seeds": arm_report["seeds"], "late_from": args.late_from,
        "response_cap": cap_record,
        "s1a_sizing": sizing_for(kept, args.late_from, n_turns, args.target_effect, args.seed),
        "sensitivity_with_capped_items": (
            sizing_for(all_rows, args.late_from, n_turns, args.target_effect, args.seed)
            if args.exclude_item else None
        ),
        "range_by_turn": range_by_turn(kept, arms),
        "g_s1_replay": g_s1_replay(arm_report, kept, arms),
        "not_a_verdict": (
            "Sizing only. No K gate reads this file. The pre-registered consequence of a small "
            "pilot contrast is to re-size S1, never to rule on executor authority."
        ),
    }

    late = report["s1a_sizing"]["late_window"]
    end = report["s1a_sizing"]["turn_20_only"]
    print(f"{len(kept)} rows, {late['variance_components']['n_items']} items, "
          f"seeds {arm_report['seeds']}, excluded {args.exclude_item or 'none'}\n")
    for name, block in report["s1a_sizing"].items():
        c, v = block["contrast"], block["variance_components"]
        print(f"{name} (t{block['turns'][0]}..t{block['turns'][1]}): contrast {c['point']:+.4f} "
              f"CI [{c['ci'][0]:+.4f}, {c['ci'][1]:+.4f}]")
        print(f"    sd of item means {v['sd_of_item_means']:.4f} = between-items "
              f"{v['sd_between_items']:.4f} (+) within-item across seeds "
              f"{v['sd_within_item_across_seeds']:.4f}")
        print("    per-seed contrast " + " ".join(f"{k}:{val:+.4f}" for k, val in v["per_seed_means"].items()))
        mde = block["sizing"]["mde_by_n_and_seeds"]
        for s_key, row in mde.items():
            print(f"    MDE {s_key}: " + "  ".join(f"{n}={val:.4f}" for n, val in row.items()))
        print(f"    n required for {args.target_effect:+.2f}: {block['sizing']['required_n_items']}")
    print()
    for arm in arms:
        rec = report["range_by_turn"][arm]["by_turn"][str(n_turns)]
        print(f"turn {n_turns} {arm:<16} mean {rec['mean']:.3f} sd {rec['sd']:.3f} "
              f"distinct {rec['n_distinct']} ceiling {rec['ceiling_share']:.3f} floor {rec['floor_share']:.3f}")
    g = report["g_s1_replay"]
    print(f"\nG-S1 replay: item sets identical {g['item_sets_identical_across_arms']}; "
          f"u in [0.45,0.55] {g['u_mean_in_band']}; cells matched {g['cells_are_matched']} "
          f"({g['n_cells_with_exactly_one_reminder']} cells)")
    print(f"  turns failing the spread clause: {g['turns_failing_spread_by_arm']}")
    print(f"  constraints outside the judge's calibration set: "
          f"{g['gold_coverage']['share_outside_calibration_set']:.4f}")
    print("\nTwo windows, both printed: the plan says 'endpoint' without naming the functional, "
          "and they imply different MDEs.\nS1 must pre-register one BEFORE submission -- choosing "
          "after seeing the variance is choosing the estimand from the noise.")

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_path.write_text(json.dumps(report, indent=2))
    print(f"\nreport written to {args.out_path}")


if __name__ == "__main__":
    main()
