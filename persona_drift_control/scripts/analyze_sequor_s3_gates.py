#!/usr/bin/env python3
"""S3's admission gate and its pre-registered estimand -- the RQ3 closed-loop
table (docs/experiments/constraint_retention_plan.md section 6, arm table
signed 2026-09-12). CPU only.

    G-S3   admission   row counts, identical item sets, turn-1 action-free,
                       per-ARM per-turn spread, no arm over its budget, and
                       the ACTION-SET check: an open-loop arm that ever played
                       a targeted action was reading the state.
    RQ3    estimand    late-window (t15..t20) mean `y` on the INDEPENDENT
                       judge, `koopman_mpc` minus `best_fixed_schedule`,
                       paired by (item, seed), bootstrapped over items.

THE SECONDARY THAT DECIDES HOW A NULL READS. `minus greedy_targeted` is what
the fitted operator bought BEYOND merely reacting to the state, and it is the
difference between two papers. If the primary resolves and this one does not,
the closed loop works and the operator is not why. The plan's own clause is
the mirror image: a primary CI containing 0 while the arms beat stored
zero_control is a clean negative -- "closed-loop control adds nothing over the
best fixed schedule" -- and is publishable as such. Neither reading is allowed
to be chosen after the numbers are seen; both are written here in advance.

WHAT THIS SCRIPT REFUSES TO DO. It never reads `y_inloop`. Those verdicts came
from the agent's own model and they DROVE the controller, so they are a
selection signal; using them for a reported number would be selecting and
reporting on one observation (.claude/global.md). They appear in the report
only as diagnostics: how often the controller was blind, and how far its view
of the state drifted from the reporting judge's.

MDE 0.0617 at 50 items x 3 seeds, carried from S1's own paired late-window
variance components (G-S2-4, signed 2026-09-10), and recomputed here from THIS
arm's variance so the verdict is read against the precision the run actually
had. A contrast below the MDE is `UNDECIDABLE` (screening section 10 item 10),
never "the closed loop is worthless" -- and the fitted model predicted the
primary at +0.0531, i.e. 0.86x this MDE, so UNDECIDABLE was the single most
likely outcome BEFORE the job was submitted. That prediction is in the report
next to the measurement, because a design known in advance to be a coin flip
must not be read afterwards as if it had been powered.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import pathlib
import statistics

PRE_REGISTERED_MDE = 0.0617
PREDICTED_BY_THE_MODEL = {"koopman_mpc_minus_best_fixed_schedule": 0.0531,
                          "koopman_mpc_minus_greedy_targeted": 0.0430}
MIN_DISTINCT_VALUES = 3
BLIND_TURN_CEILING = 0.15  # the in-loop judge measured 5% unparsed (15739196); 3x that is a fault
TREATMENT = "koopman_mpc"
OPEN_LOOP = ("best_fixed_schedule", "equal_cost_random")
ZERO_CONTROL = "zero_control"


def _sibling_module(name: str):
    path = pathlib.Path(__file__).with_name(name)
    spec = importlib.util.spec_from_file_location(f"_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--independent-readout", type=pathlib.Path, required=True,
                   help="score_sequor_trajectories.py report over the S3 arm, judge_kind=independent.")
    p.add_argument("--zero-control-readout", type=pathlib.Path, default=None,
                   help="The S1 arm's independent readout, for the stored zero_control comparison. "
                        "Its items are the first 40 of S3's 50 (both use select_bank_items), so the "
                        "contrast is computed on the intersection and the count is reported.")
    p.add_argument("--out-path", type=pathlib.Path, required=True, help="Must not already exist.")
    p.add_argument("--late-from", type=int, default=15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--exclude-item", action="append", default=[], metavar="ITEM_ID")
    return p.parse_args()


def gate_g_s3(arm_report: dict, run_config: dict, rows: list[dict], arms: list[str],
              n_turns: int) -> dict:
    """The design's own properties, checked on the rows that exist.

    The action-set clause is the one that could not be checked before this arm
    existed. An open-loop arm has no way to name the broken constraints, so a
    row where it named a proper subset of k would mean the runner leaked the
    state into a schedule -- and the whole comparison would be between two
    closed loops. It is checked on rows, not on the policy code, because the
    policy being right does not prove the runner used it.
    """

    item_sets = {arm: sorted({r["item_id"] for r in rows if r["branch"] == arm}) for arm in arms}
    reference = item_sets[arms[0]]
    k = len(rows[0]["constraints"])

    by_arm_turn: dict[tuple[str, int], list[float]] = {}
    for row in rows:
        if row["y_graded"] is not None:
            by_arm_turn.setdefault((row["branch"], row["turn"]), []).append(row["y_graded"])
    spread = {}
    for arm in arms:
        per_turn = {}
        for turn in range(1, n_turns + 1):
            values = by_arm_turn.get((arm, turn), [])
            if not values:
                continue
            per_turn[str(turn)] = {
                "n": len(values), "mean": statistics.fmean(values), "sd": statistics.pstdev(values),
                "n_distinct": len(set(values)),
                "ceiling_share": sum(1 for v in values if v == 1.0) / len(values),
            }
        spread[arm] = {"by_turn": per_turn,
                       "turns_failing": [t for t, rec in per_turn.items()
                                         if rec["sd"] == 0.0 or rec["n_distinct"] < MIN_DISTINCT_VALUES]}

    turn1_actions = sorted({row["branch"] for row in rows if row["turn"] == 1 and row["n_named"]})
    leaked = sorted({row["branch"] for row in rows
                     if row["branch"] in OPEN_LOOP and 0 < row["n_named"] < k})
    targeted_outside_state = sorted({
        row["trajectory_id"] for row in rows
        if row["branch"] not in OPEN_LOOP and row["n_named"] and row["state_before_action"] is not None
        and not set(row["action_named"]) <= {j for j, ok in enumerate(row["state_before_action"]) if not ok}
        and row["n_named"] != k})

    spend = arm_report["budget_accounting"]
    overspent = {arm: stats["overspent_items"] for arm, stats in spend.items()
                 if stats["overspent_items"]}
    blind = {arm: stats["blind_turn_share"] for arm, stats in spend.items()}

    n_seeds = len(arm_report["seeds"])
    declared_items = arm_report["n_items"]
    expected_rows = len(arms) * declared_items * n_turns * n_seeds
    checks = {
        "row_count_matches_design": len(rows) == expected_rows,
        "item_sets_identical_across_arms": all(v == reference for v in item_sets.values()),
        "turn_1_action_free_in_every_arm": not turn1_actions,
        "open_loop_arms_never_targeted": not leaked,
        "closed_loop_targets_were_broken_constraints": not targeted_outside_state,
        "no_arm_overspent_its_budget": not overspent,
        "blind_turn_share_under_ceiling": all(v <= BLIND_TURN_CEILING for v in blind.values()),
        "spread_at_every_turn_per_arm": all(not s["turns_failing"] for s in spread.values()),
    }
    return {
        "criterion": f"rows = arms x items x turns x seeds; identical item sets; no action at turn 1; "
                     f"open-loop arms play only none/blanket; closed-loop targets are constraints the "
                     f"in-loop judge called broken; no arm over budget; blind-turn share "
                     f"<= {BLIND_TURN_CEILING:.0%}; per ARM every turn has sd > 0 and "
                     f">= {MIN_DISTINCT_VALUES} distinct y",
        "n_rows": len(rows), "n_rows_expected": expected_rows, "n_items": len(reference),
        "n_seeds": n_seeds, "checks": checks,
        "arms_acting_at_turn_1": turn1_actions,
        "open_loop_arms_that_targeted": leaked,
        "closed_loop_rows_targeting_an_unbroken_constraint": targeted_outside_state[:20],
        "overspent": overspent,
        "budget_share_spent": {arm: stats["share_of_budget_spent"] for arm, stats in spend.items()},
        "budget_share": run_config["budget_share"],
        "blind_turn_share": blind,
        "actions_by_arm": {arm: {"total": stats["n_actions"], "targeted": stats["n_targeted_actions"],
                                 "blanket": stats["n_blanket_actions"]}
                           for arm, stats in spend.items()},
        "turns_failing_spread_by_arm": {arm: spread[arm]["turns_failing"] for arm in arms},
        "spread_by_arm": spread,
        "verdict": "PASS" if all(checks.values()) else "FAIL",
        "on_failure": "report and hand back; the plan forbids re-running an arm until it admits",
    }


def contrast(pilot, rows: list[dict], treated: str, control: str, turns: range, seed: int) -> dict:
    """The signed estimand on one pair of arms: paired by (item, seed),
    bootstrapped over items, with the MDE recomputed from this arm's own
    variance components in the same units as the quantity tested."""

    paired = pilot.per_item_contrast(rows, treated, control, turns)
    if not paired:
        raise SystemExit(f"{treated} vs {control}: no paired items; the arms do not share an item set")
    components = pilot.variance_components(paired)
    boot = pilot.bootstrap_contrast(paired, seed)

    n_items = components["n_items"]
    n_seeds = min(components["n_seeds_per_item"])
    mde = pilot.POWER_Z * math.sqrt(
        (components["sd_between_items"] ** 2 + components["sd_within_item_across_seeds"] ** 2 / n_seeds)
        / n_items)
    per_seed = [components["per_seed_means"][str(s)]
                for s in sorted(int(k) for k in components["per_seed_means"])]
    sd_seeds = statistics.stdev(per_seed) if len(per_seed) > 1 else 0.0
    point = boot["point"]
    return {
        "treated": treated, "control": control, "turns": [min(turns), max(turns)],
        "paired_by": "(item_id, seed)", "judge_kind": "independent",
        "contrast": point, "ci": boot["ci"], "bootstrap_sd": boot["bootstrap_sd"],
        "n_items": n_items, "n_seeds": n_seeds,
        "per_seed_contrast": {str(s): v for s, v in enumerate(per_seed)},
        "reportable": f"{statistics.fmean(per_seed):+.4f} +/- {sd_seeds:.4f} (n={len(per_seed)} seeds)",
        "variance_components": components,
        "mde_at_80pct_this_arm": mde,
        "mde_at_80pct_pre_registered": PRE_REGISTERED_MDE,
        "abs_effect_over_mde": abs(point) / mde if mde else None,
        "ci_excludes_zero": bool(boot["ci"][0] * boot["ci"][1] > 0),
        "resolved": bool(boot["ci"][0] * boot["ci"][1] > 0 and abs(point) >= mde),
    }


def arm_levels(rows: list[dict], arms: list[str], turns: range) -> dict:
    """The late-window level of each arm. A contrast of +0.01 reads differently
    at 0.55 than at 0.95, and the primary alone does not say which."""

    out = {}
    for arm in arms:
        values = [r["y_graded"] for r in rows
                  if r["branch"] == arm and r["turn"] in turns and r["y_graded"] is not None]
        if values:
            out[arm] = {"n": len(values), "mean": statistics.fmean(values),
                        "sd": statistics.pstdev(values),
                        "ceiling_share": sum(1 for v in values if v == 1.0) / len(values)}
    return out


def inloop_vs_independent(rows: list[dict]) -> dict:
    """How far the controller's view of the state sat from the reporting
    judge's, on the same rows.

    A diagnostic and nothing more: the controller could only ever act on the
    in-loop verdicts, so a large divergence does not invalidate the arm -- it
    prices what a cheap in-loop judge costs, which is the number the zero-GPU
    targeting-agreement check (48.6% on the branch arm) estimated in advance.
    """

    agree, total, both_parsed = 0, 0, 0
    for row in rows:
        inloop, independent = row.get("followed_inloop"), row.get("followed")
        total += 1
        if inloop is None or independent is None:
            continue
        if any(b is None for b in inloop) or any(b is None for b in independent):
            continue
        both_parsed += 1
        agree += sum(1 for a, b in zip(inloop, independent) if bool(a) == bool(b))
    k = len(rows[0]["constraints"])
    return {
        "n_rows": total, "n_rows_both_parsed": both_parsed,
        "per_constraint_agreement": agree / (both_parsed * k) if both_parsed else None,
        "reading": "the share of constraint verdicts on which the controller's judge and the "
                   "reporting judge agreed. The controller acted on the former; every reported "
                   "number comes from the latter.",
    }


def verdict(primary: dict, operator_margin: dict, versus_zero: dict | None) -> dict:
    """RQ3's three readings, all written before the numbers were seen.

    PASS needs both halves -- a CI excluding zero AND an effect at or above
    this arm's own MDE -- because a contrast significant at ~1.96 sigma but
    under the design's 80%-power MDE is a point estimate detected at less than
    80% power and likely inflated by the noise that made it detectable. That is
    the K3 shape, and this line has already paid for it once.
    """

    if primary["resolved"] and primary["contrast"] > 0:
        operator = ("and the operator earned it: the margin over greedy_targeted also resolves"
                    if operator_margin["resolved"] and operator_margin["contrast"] > 0
                    else "but the margin over greedy_targeted does not resolve, so this design "
                         "cannot say the FITTED OPERATOR earned it rather than reacting to the "
                         "state at all")
        return {"verdict": "RQ3_POSITIVE", "closes_the_line": False,
                "reason": f"at equal token cost the closed loop beats the best fixed schedule "
                          f"by {primary['contrast']:+.4f} ({primary['abs_effect_over_mde']:.2f}x "
                          f"this arm's MDE), {operator}",
                "consequence": "fill the RQ3 table; the operator margin decides whether the claim "
                               "is about Koopman planning or about feedback in general"}
    if primary["resolved"]:
        return {"verdict": "RQ3_NEGATIVE_RESOLVED", "closes_the_line": True,
                "reason": f"the closed loop is resolvably WORSE than the best fixed schedule "
                          f"({primary['contrast']:+.4f}). Reading the state and paying for it cost "
                          f"more than it bought",
                "consequence": "report as a negative result; `koopman_mpc` does not go in the paper "
                               "as a method"}
    beat_zero = versus_zero is not None and versus_zero["resolved"] and versus_zero["contrast"] > 0
    return {
        "verdict": "UNDECIDABLE", "closes_the_line": None,
        "reason": (f"|{primary['contrast']:+.4f}| against this arm's MDE "
                   f"{primary['mde_at_80pct_this_arm']:.4f}"
                   + ("" if primary["ci_excludes_zero"] else ", CI crosses 0")
                   + f". The fitted model predicted "
                     f"{PREDICTED_BY_THE_MODEL['koopman_mpc_minus_best_fixed_schedule']:+.4f} "
                     f"= 0.86x MDE BEFORE submission, so this outcome was the most likely one and "
                     f"is a statement about the design's resolution, not about the closed loop"),
        "consequence": (
            "the plan's signed reading applies: a primary CI containing 0 while the arms beat "
            "stored zero_control is a CLEAN NEGATIVE -- closed-loop control adds nothing over the "
            "best fixed schedule at this budget -- and is publishable as such"
            if beat_zero else
            "the arms do not resolvably beat stored zero_control either, so this run says nothing "
            "about the closed loop; report the budget curve and hand back"),
        "arms_beat_zero_control": beat_zero,
    }


def main() -> None:
    args = parse_args()
    if args.out_path.exists():
        raise SystemExit(f"refusing to overwrite existing {args.out_path}")
    gates = _sibling_module("analyze_sequor_s0_0_gates.py")
    pilot = _sibling_module("analyze_sequor_s1_pilot.py")

    readout = gates.load_readout(args.independent_readout, "independent")
    arm_dir = pathlib.Path(readout["arm_dir"])
    arm_report = json.loads((arm_dir / "arm_report.json").read_text())
    run_config = json.loads((arm_dir / "run_config.json").read_text())
    cap_record = gates.refuse_if_the_cap_bound(arm_report, args.exclude_item)

    arms = list(arm_report["arms"])
    n_turns = arm_report["n_turns"]
    by_id = {row["trajectory_id"]: row for row in readout["rows"]}
    with (arm_dir / "trajectories.jsonl").open() as handle:
        for line in handle:
            raw = json.loads(line)
            row = by_id.get(raw["trajectory_id"])
            if row is not None and row["turn"] == raw["turn"]:
                for field in ("n_named", "action_named", "state_before_action", "followed_inloop",
                              "y_inloop", "budget", "tokens_left_after"):
                    row[field] = raw[field]
    rows = [row for row in readout["rows"] if row["item_id"] not in args.exclude_item]

    admission = gate_g_s3(arm_report, run_config, rows, arms, n_turns)
    turns = range(args.late_from, n_turns + 1)

    primary = contrast(pilot, rows, TREATMENT, "best_fixed_schedule", turns, args.seed)
    operator_margin = contrast(pilot, rows, TREATMENT, "greedy_targeted", turns, args.seed)
    secondary = {"koopman_mpc_minus_greedy_targeted": operator_margin}
    for arm in arms:
        if arm not in (TREATMENT, "best_fixed_schedule", "greedy_targeted"):
            secondary[f"{TREATMENT}_minus_{arm}"] = contrast(pilot, rows, TREATMENT, arm, turns, args.seed)

    versus_zero = None
    if args.zero_control_readout:
        stored = gates.load_readout(args.zero_control_readout, "independent")
        zero_rows = [r for r in stored["rows"]
                     if r["branch"] == ZERO_CONTROL and r["item_id"] not in args.exclude_item]
        shared = sorted({r["item_id"] for r in rows} & {r["item_id"] for r in zero_rows})
        merged = ([r for r in rows if r["item_id"] in shared]
                  + [r for r in zero_rows if r["item_id"] in shared])
        versus_zero = contrast(pilot, merged, TREATMENT, ZERO_CONTROL, turns, args.seed)
        versus_zero["n_shared_items"] = len(shared)
        versus_zero["rng_paired"] = False
        versus_zero["note"] = (
            "the stored zero_control rows come from job 15772956 under the same harness, items and "
            "seeds, but they were sampled at another time and are NOT RNG-paired with these arms; "
            "read for direction and for whether the arms beat doing nothing at all")
        secondary[f"{TREATMENT}_minus_{ZERO_CONTROL}"] = versus_zero

    ruling = verdict(primary, operator_margin, versus_zero)
    gold = arm_report["gold_coverage"]

    report = {
        "arm_dir": str(arm_dir), "provenance": readout.get("provenance"),
        "agent_model": readout["agent_model"], "judge_model": readout["judge_model"],
        "judge_kind": readout["judge_kind"], "arms": arms, "seeds": arm_report["seeds"],
        "n_turns": n_turns, "late_from": args.late_from,
        "budget_share": run_config["budget_share"],
        "response_cap": cap_record, "n_rows_unusable": readout.get("n_rows_unusable"),
        "g_s3": admission,
        "estimand": {
            "primary": f"{TREATMENT} minus best_fixed_schedule, late window "
                       f"t{args.late_from}..t{n_turns}, independent judge, paired by (item, seed)",
            "secondary": list(secondary),
            "signed": "2026-09-12, before submission (plan section 6)",
            "rule": "no secondary quantity may be promoted to primary after the fact",
        },
        "arm_levels": arm_levels(rows, arms, turns),
        "primary": primary, "secondary": secondary,
        "predicted_before_submission": PREDICTED_BY_THE_MODEL,
        "inloop_vs_independent": inloop_vs_independent(rows),
        "verdict": ruling,
        "caveat": (
            f"{gold['n_constraints_judged'] - gold['n_constraints_in_calibration_set']}"
            f"/{gold['n_constraints_judged']} ({gold['share_outside_calibration_set']:.0%}) of the "
            "judged constraints lie outside the judge's calibration set; its accuracy there was "
            "never measured, and that noise attenuates every contrast toward zero."
        ),
    }

    print(f"G-S3 {admission['verdict']}: " + "  ".join(
        f"{k}={'ok' if v else 'FAIL'}" for k, v in admission["checks"].items()))
    print(f"  budget spent {admission['budget_share_spent']}")
    print(f"  blind turns {admission['blind_turn_share']}   actions {admission['actions_by_arm']}")
    print("\nlate-window levels:")
    for arm, level in report["arm_levels"].items():
        print(f"  {arm:22s} {level['mean']:.4f}  (ceiling share {level['ceiling_share']:.3f})")
    print(f"\nPRIMARY  {primary['treated']} - {primary['control']}  t{primary['turns'][0]}..{primary['turns'][1]}")
    print(f"  {primary['reportable']}   CI [{primary['ci'][0]:+.4f}, {primary['ci'][1]:+.4f}] "
          f"over {primary['n_items']} items")
    print(f"  MDE(this arm) {primary['mde_at_80pct_this_arm']:.4f} "
          f"(pre-registered {PRE_REGISTERED_MDE})  |effect|/MDE {primary['abs_effect_over_mde']:.2f}  "
          f"(model predicted {PREDICTED_BY_THE_MODEL['koopman_mpc_minus_best_fixed_schedule']:+.4f})")
    for name, block in secondary.items():
        print(f"SECONDARY  {name}: {block['contrast']:+.4f}  "
              f"CI [{block['ci'][0]:+.4f}, {block['ci'][1]:+.4f}]  "
              f"|effect|/MDE {block['abs_effect_over_mde']:.2f}")
    print(f"\nin-loop vs reporting judge agreement: "
          f"{report['inloop_vs_independent']['per_constraint_agreement']}")
    print(f"\nVERDICT {ruling['verdict']} -- {ruling['reason']}")
    print(f"  consequence: {ruling['consequence']}")
    print(f"\n{report['caveat']}")

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_path.write_text(json.dumps(report, indent=2))
    print(f"\nreport written to {args.out_path}")


if __name__ == "__main__":
    main()
