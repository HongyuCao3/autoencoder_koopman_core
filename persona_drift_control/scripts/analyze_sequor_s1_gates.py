#!/usr/bin/env python3
"""S1's admission gate and its pre-registered S1a estimand
(docs/experiments/constraint_retention_plan.md section 4;
 docs/experiments/constraint_signal_screening.md section 10 item 9). CPU only.

    G-S1   admission        row counts, identical item sets, `u` balance,
                            both actions at every turn, and -- PER ARM --
                            `y` spread at every turn. Fails -> report, never
                            re-sample.
    S1a    premise (i)      `constant_remind` minus `zero_control` over the
                            LATE WINDOW t15..t20, paired by (item, seed), on
                            the independent judge's readout. Estimand signed
                            2026-09-10 BEFORE the arm was submitted; turn-20
                            only is reported beside it as the secondary
                            quantity and may not be promoted afterwards.

S1a is a measurement, not a kill gate. The pre-registered consequence of a
contrast below its own MDE is to REPORT AND HAND BACK: one-step authority was
already answered by K3 (+0.1233, CI [+0.0595, +0.1898], 1.30x MDE), so a small
endpoint contrast re-sizes S1 rather than closing the line (screening section
10 item 9). The verdict vocabulary here encodes that -- there is no FAIL branch
that closes anything.

Every number is reported with the MDE recomputed from THIS arm's own variance
components, in the same units as the quantity tested. The share of judged
constraints outside the judge's calibration set (65% on this item set) rides
along in the report because it must appear beside any number these rows
produce.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import pathlib
import statistics

MIN_DISTINCT_VALUES = 3  # plan section 4's G-S1 spread clause
U_BAND = (0.45, 0.55)  # plan section 4's G-S1 balance clause
PRE_REGISTERED_MDE = {"late_window": 0.0636, "turn_20_only": 0.0991}  # pilot sizing, s1_sizing_report.json


def _sibling_module(name: str):
    """Import a sibling analysis script rather than restating its criteria.

    The response-cap guard and the readout loader live in the S0-0 gate
    script; the contrast, the variance decomposition and the sizing table live
    in the pilot script and were used to size this very arm. Two copies of a
    threshold drift apart, and re-deriving the sizing arithmetic here would
    mean the MDE that justified N=40 and the MDE the verdict is read against
    could stop being the same function.
    """

    path = pathlib.Path(__file__).with_name(name)
    spec = importlib.util.spec_from_file_location(f"_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--independent-readout", type=pathlib.Path, required=True,
                   help="score_sequor_trajectories.py report with judge_kind=independent. "
                        "The reporting readout is the only one a number may come from.")
    p.add_argument("--out-path", type=pathlib.Path, required=True, help="Must not already exist.")
    p.add_argument("--late-from", type=int, default=15,
                   help="First turn of the pre-registered late window (inclusive). The estimand "
                        "was signed at t15..t20 before submission; changing it after the fact is "
                        "choosing the estimand from the noise.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--exclude-item", action="append", default=[], metavar="ITEM_ID",
                   help="Drop this item, the only way past the response-cap guard. Every "
                        "exclusion is written into the report beside the numbers it changed.")
    return p.parse_args()


def gate_g_s1(arm_report: dict, rows: list[dict], arms: list[str], n_turns: int) -> dict:
    """The plan's admission checks, computed on the rows that exist.

    The spread clause is evaluated PER ARM and never on arms merged together
    (screening section 10 item 9, clause 2): the pilot's `constant_remind` had
    2 distinct values at t2 while the pooled set had more, and pooling would
    hide exactly the saturation the clause exists to catch.

    A failure here is reported and handed back. Re-sampling until an admission
    check passes would make the check a formality.
    """

    item_sets = {arm: sorted({r["item_id"] for r in rows if r["branch"] == arm}) for arm in arms}
    reference = item_sets[arms[0]]
    n_items = len(reference)

    by_arm_turn: dict[tuple[str, int], list[float]] = {}
    actions_by_turn: dict[int, set[int]] = {}
    for row in rows:
        if row["branch"] in ("bernoulli", "antithetic") and row["turn"] >= 2:
            actions_by_turn.setdefault(row["turn"], set()).add(int(row["u_remind"]))
        if row["y_graded"] is None:
            continue
        by_arm_turn.setdefault((row["branch"], row["turn"]), []).append(row["y_graded"])

    spread: dict[str, dict] = {}
    for arm in arms:
        per_turn = {}
        for turn in range(1, n_turns + 1):
            values = by_arm_turn.get((arm, turn), [])
            if not values:
                continue
            per_turn[str(turn)] = {
                "n": len(values), "mean": statistics.fmean(values),
                "sd": statistics.pstdev(values), "n_distinct": len(set(values)),
                "ceiling_share": sum(1 for v in values if v == 1.0) / len(values),
                "floor_share": sum(1 for v in values if v == 0.0) / len(values),
            }
        spread[arm] = {
            "by_turn": per_turn,
            "turns_failing": [t for t, rec in per_turn.items()
                              if rec["sd"] == 0.0 or rec["n_distinct"] < MIN_DISTINCT_VALUES],
        }

    u_mean_in_band = {
        str(seed): U_BAND[0] <= check["bernoulli_u_mean"] <= U_BAND[1]
        for seed, check in arm_report["schedule_checks"].items()
    }
    turns_missing_an_action = sorted(t for t, seen in actions_by_turn.items() if seen != {0, 1})
    # The plan writes this clause as "arms x 40 x 20": it predates the ruling that
    # moved the line from one greedy pass to 3 sampled seeds (screening section 10
    # item 5). Seeds multiply the rows; dropping the factor would make the clause
    # fail on every arm the line has actually run.
    n_seeds = len(arm_report["seeds"])
    expected_rows = len(arms) * n_items * n_turns * n_seeds
    checks = {
        "row_count_matches_design": len(rows) == expected_rows,
        "item_sets_identical_across_arms": all(v == reference for v in item_sets.values()),
        "u_mean_in_band_per_seed": all(u_mean_in_band.values()),
        "both_actions_at_every_turn": not turns_missing_an_action,
        "cells_are_matched": bool(arm_report["cell_coverage"]["cells_are_matched"]),
        "spread_at_every_turn_per_arm": all(not s["turns_failing"] for s in spread.values()),
    }
    return {
        "criterion": "rows = arms x items x turns x seeds; identical item sets; bernoulli u mean in "
                     f"[{U_BAND[0]}, {U_BAND[1]}] per seed; both actions at every turn t>=2; "
                     f"per ARM, every turn has sd > 0 and >= {MIN_DISTINCT_VALUES} distinct y",
        "n_rows": len(rows), "n_rows_expected": expected_rows, "n_items": n_items, "n_seeds": n_seeds,
        "checks": checks,
        "u_mean_by_seed": {s: c["bernoulli_u_mean"] for s, c in arm_report["schedule_checks"].items()},
        "u_mean_in_band": u_mean_in_band,
        "antithetic_is_exact_complement": {s: c["antithetic_is_exact_complement"]
                                           for s, c in arm_report["schedule_checks"].items()},
        "turns_missing_an_action": turns_missing_an_action,
        "turns_failing_spread_by_arm": {arm: spread[arm]["turns_failing"] for arm in arms},
        "spread_by_arm": spread,
        "verdict": "PASS" if all(checks.values()) else "FAIL",
        "on_failure": "report and hand back; the plan forbids re-sampling to pass admission",
    }


def s1a_contrast(pilot, rows: list[dict], turns: range, seed: int) -> dict:
    """The signed estimand on one window: paired by (item, seed), bootstrap CI
    over items, and the MDE recomputed from this arm's own variance.

    Two aggregations of the same paired differences are reported and they
    answer different questions. The bootstrap CI resamples ITEMS (the
    independent unit) and is what the contrast's precision should be read off.
    The `mean +/- sd (n = 3 seeds)` line is the house reporting format
    (.claude/global.md), and its sd is a between-seed spread, not the
    contrast's standard error -- three seeds of one item set cannot estimate
    item-to-item variation.
    """

    contrast = pilot.per_item_contrast(rows, "constant_remind", "zero_control", turns)
    if not contrast:
        raise SystemExit(f"t{min(turns)}..t{max(turns)}: no paired items; arms do not share an item set")
    components = pilot.variance_components(contrast)
    boot = pilot.bootstrap_contrast(contrast, seed)

    n_items = components["n_items"]
    n_seeds = min(components["n_seeds_per_item"])
    var_item = components["sd_between_items"] ** 2
    var_seed = components["sd_within_item_across_seeds"] ** 2
    mde = pilot.POWER_Z * math.sqrt((var_item + var_seed / n_seeds) / n_items)

    per_seed = [components["per_seed_means"][str(s)] for s in sorted(int(k) for k in components["per_seed_means"])]
    point = boot["point"]
    return {
        "turns": [min(turns), max(turns)],
        "paired_by": "(item_id, seed)",
        "judge_kind": "independent",
        "contrast": point,
        "ci": boot["ci"],
        "bootstrap_sd": boot["bootstrap_sd"],
        "n_items": n_items,
        "n_seeds": n_seeds,
        "per_seed_contrast": {str(s): v for s, v in enumerate(per_seed)},
        "mean_over_seeds": statistics.fmean(per_seed),
        "sd_over_seeds": statistics.stdev(per_seed) if len(per_seed) > 1 else 0.0,
        "reportable": f"{statistics.fmean(per_seed):+.4f} +/- "
                      f"{statistics.stdev(per_seed) if len(per_seed) > 1 else 0.0:.4f} (n={len(per_seed)} seeds)",
        "variance_components": components,
        "mde_at_80pct_this_arm": mde,
        "mde_at_80pct_pre_registered": PRE_REGISTERED_MDE.get(
            "late_window" if min(turns) < max(turns) else "turn_20_only"),
        "effect_over_mde": abs(point) / mde if mde else None,
        "ci_excludes_zero": bool(boot["ci"][0] * boot["ci"][1] > 0),
        "resolved": bool(boot["ci"][0] * boot["ci"][1] > 0 and abs(point) >= mde),
    }


def s1a_verdict(primary: dict) -> dict:
    """Premise (i) at the endpoint, with the pre-registered consequence of a
    small contrast written into the vocabulary.

    RESOLVED requires both halves -- a CI excluding zero AND an effect at or
    above the design's own MDE -- because S0-0's K3 passed the first half and
    failed the second (0.84x MDE) under the old harness, and a "significant"
    number under its own MDE is a point estimate likely inflated by selection
    on noise.

    There is deliberately no branch that closes the line. Authority on the
    one-step quantity is already banked by K3 under the fidelity harness
    (+0.1233, 1.30x MDE); an endpoint contrast this design cannot resolve is a
    statement about N, not about the executor (screening section 10 item 9).
    """

    if primary["resolved"]:
        verdict, reason = "RESOLVED", (
            "the CI excludes 0 and the contrast is at or above this arm's own MDE: "
            "premise (i) holds at the endpoint under this harness")
    elif primary["ci_excludes_zero"]:
        verdict, reason = "REPORT_AND_HAND_BACK", (
            "the CI excludes 0 but the contrast sits below this arm's 80%-power MDE -- "
            "significant at ~1.96 sigma, detected at under 80% power, point estimate likely "
            "inflated (the S0-0 K3 shape). Re-size, do not promote")
    else:
        verdict, reason = "REPORT_AND_HAND_BACK", (
            "the CI crosses 0 at this N. The pre-registered consequence is to re-size S1, NOT "
            "to conclude the executor lacks authority -- K3 answered that on one-step pairs")
    return {"verdict": verdict, "reason": reason,
            "closes_the_line": False,
            "note": "S1a has no line-closing branch by design (screening section 10 item 9)."}


def main() -> None:
    args = parse_args()
    if args.out_path.exists():
        raise SystemExit(f"refusing to overwrite existing {args.out_path}")
    gates = _sibling_module("analyze_sequor_s0_0_gates.py")
    pilot = _sibling_module("analyze_sequor_s1_pilot.py")

    readout = gates.load_readout(args.independent_readout, "independent")
    arm_dir = pathlib.Path(readout["arm_dir"])
    arm_report = json.loads((arm_dir / "arm_report.json").read_text())
    cap_record = gates.refuse_if_the_cap_bound(arm_report, args.exclude_item)

    arms = list(arm_report["arms"])
    n_turns = arm_report["n_turns"]
    rows = [r for r in readout["rows"] if r["item_id"] not in args.exclude_item]

    admission = gate_g_s1(arm_report, rows, arms, n_turns)
    windows = {
        "late_window": range(args.late_from, n_turns + 1),
        "turn_20_only": range(n_turns, n_turns + 1),
    }
    s1a = {name: s1a_contrast(pilot, rows, turns, args.seed) for name, turns in windows.items()}
    verdict = s1a_verdict(s1a["late_window"])

    report = {
        "arm_dir": str(arm_dir),
        "provenance": readout.get("provenance"),
        "agent_model": readout["agent_model"], "judge_model": readout["judge_model"],
        "judge_kind": readout["judge_kind"], "arms": arms, "seeds": arm_report["seeds"],
        "n_turns": n_turns, "late_from": args.late_from,
        "response_cap": cap_record,
        "n_rows_unusable": readout.get("n_rows_unusable"),
        "g_s1": admission,
        "s1a": s1a,
        "s1a_estimand": {
            "primary": "late_window", "secondary": "turn_20_only",
            "signed": "2026-09-10, before submission (screening section 10 item 9)",
            "rule": "the secondary quantity may not be promoted after the fact",
        },
        "s1a_verdict": verdict,
        "gold_coverage": arm_report["gold_coverage"],
        "caveat": (
            f"{arm_report['gold_coverage']['n_constraints_judged'] - arm_report['gold_coverage']['n_constraints_in_calibration_set']}"
            f"/{arm_report['gold_coverage']['n_constraints_judged']} "
            f"({arm_report['gold_coverage']['share_outside_calibration_set']:.0%}) of the judged "
            "constraints lie outside the judge's calibration set; its accuracy there was never "
            "measured. That noise biases STATE coefficients toward zero (S2), and must be "
            "reported beside every number from these rows (screening section 10 item 9)."
        ),
    }

    print(f"G-S1 {admission['verdict']}: " + "  ".join(
        f"{k}={'ok' if v else 'FAIL'}" for k, v in admission["checks"].items()))
    print(f"  u mean by seed {admission['u_mean_by_seed']}; "
          f"turns failing spread {admission['turns_failing_spread_by_arm']}")
    print()
    for name, block in s1a.items():
        print(f"S1a {name} (t{block['turns'][0]}..t{block['turns'][1]}): {block['reportable']}, "
              f"bootstrap CI [{block['ci'][0]:+.4f}, {block['ci'][1]:+.4f}] over {block['n_items']} items")
        print(f"    per-seed " + " ".join(f"s{k}:{v:+.4f}" for k, v in block["per_seed_contrast"].items())
              + f"   MDE(this arm) {block['mde_at_80pct_this_arm']:.4f} "
                f"(pre-registered {block['mde_at_80pct_pre_registered']})  "
                f"effect/MDE {block['effect_over_mde']:.2f}")
    print(f"\nS1a verdict: {verdict['verdict']} -- {verdict['reason']}")
    print(f"\n{report['caveat']}")

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_path.write_text(json.dumps(report, indent=2))
    print(f"\nreport written to {args.out_path}")


if __name__ == "__main__":
    main()
