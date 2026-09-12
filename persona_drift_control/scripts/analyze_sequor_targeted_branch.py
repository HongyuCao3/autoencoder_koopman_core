#!/usr/bin/env python3
"""The targeted-reminder pilot's pre-registered gate
(docs/experiments/constraint_signal_screening.md section 10 item 13, signed
2026-09-11 BEFORE job 15792569 was submitted). CPU only.

WHAT IT DECIDES. Options (a) and (b) both returned zero: under a scalar
operator, under a per-constraint binary state, and under a thresholded
objective with priced reminders, the best SCHEDULE is the same for every
trajectory. The mechanism is that the action is one switch, so the state can
only scale what a reminder is worth. This arm changes the ACTION -- restate
only the constraint that broke -- which is the one thing a fixed plan cannot
imitate, because which constraint breaks at turn 12 is not knowable at turn 1.

    PASS   -> rewrite the plan's section 6 arm table around a targeted
              executor; S3 is the main-line action, not another gate.
    FAIL   -> cut the `koopman_mpc` arm; S3 keeps only the open-loop
              comparison. No fourth re-specification.

PRIMARY QUANTITY (signed, single): of the constraints the readout called
violated at t-1, the share called followed at t -- targeted minus blanket,
paired at (item, seed, turn, constraint), bootstrapped over ITEMS.

MDE 0.0525, measured not assumed: the same recovery contrast on the S1b arm
had a between-item sd of 0.1185 over 28 items, and 2.80 * 0.1185 / sqrt(40) =
0.0525. Blanket reminding buys +0.0589 there, so this design resolves only
"targeting roughly doubles the executor" and nothing finer. BELOW THE MDE IS
`UNDECIDABLE`, which is not "targeting does not work" -- and it is not a
ruling this script is allowed to make, because item 13 wrote down only the
pass and fail branches.

SECONDARY QUANTITIES, reported beside the primary and never promoted to it:
the retention of constraints that were ALREADY held at t-1 (a targeted
reminder does not mention them, and the model may read the silence as
permission -- that would be trading one constraint for another), the net
`y_graded` contrast, each branch against the archived u=0 row, and the
insertion cost that makes an equal-token reading possible.

SELECTION VS REPORT. The branch point is chosen on the t-1 verdict and the
outcome is read on t: two observations, never the same one. All three branches
are selected by the identical condition and share a prefix, so conditioning
moves the levels and not the contrast. Only the independent judge is read here
-- the 4B in-loop judge's discount was already measured at zero GPU and rides
along in the report.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import pathlib
import statistics

BLANKET = "blanket_remind"
TARGETED = "targeted_remind"
BASE_ARM = "zero_control"
PRE_REGISTERED_MDE = 0.0525  # 2.80 * 0.1185 / sqrt(40); screening section 10 item 13
POWER_Z = 2.80  # two-sided alpha=0.05 at 80% power, as everywhere else on this line


def _sibling_module(name: str):
    """Import a sibling analysis script rather than restating its criteria.

    The response-cap guard, the readout loader and the item bootstrap live in
    the S0-0 gate script and have been read against every arm on this line.
    A second copy of a threshold is a threshold that can drift.
    """

    path = pathlib.Path(__file__).with_name(name)
    spec = importlib.util.spec_from_file_location(f"_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--independent-readout", type=pathlib.Path, required=True,
                   help="score_sequor_trajectories.py report over the targeted-branch arm, "
                        "judge_kind=independent. The reporting readout is the only source of a number.")
    p.add_argument("--base-readout", type=pathlib.Path, required=True,
                   help="The S1 arm's independent readout. Supplies the t-1 verdicts the targets "
                        "are checked against, and the archived u=0 row for the secondary contrast.")
    p.add_argument("--out-path", type=pathlib.Path, required=True, help="Must not already exist.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--exclude-item", action="append", default=[], metavar="ITEM_ID",
                   help="Drop this item, the only way past the response-cap guard. Every exclusion "
                        "is written into the report beside the numbers it changed.")
    return p.parse_args()


def branch_metadata(arm_dir: pathlib.Path) -> dict[str, dict]:
    """`violated_indices` and the base trajectory, per branch row.

    These live in the trajectory artifact and not in the readout, because the
    readout only knows what the judge was shown. Reading them back from the
    artifact is also what lets the next function check the runner's targeting
    against the readout that was supposed to have produced it.
    """

    out: dict[str, dict] = {}
    with (arm_dir / "trajectories.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            out[row["trajectory_id"]] = {
                "item_id": row["item_id"], "seed": row["seed"], "turn": row["turn"],
                "branch": row["branch"], "violated_indices": row["violated_indices"],
                "base_trajectory_id": row["base_trajectory_id"],
                "n_violated_at_t_minus_1": row["n_violated_at_t_minus_1"],
            }
    return out


def verify_targets_against_base(meta: dict[str, dict], base_verdicts: dict[tuple[str, int], list]) -> dict:
    """Every recorded target must equal the set the base readout calls broken
    at t-1.

    The runner computed the targets from this same readout, so a mismatch does
    not mean the targeting was wrong -- it means the artifact and the readout
    on disk are not the pair that produced each other, and every number below
    would be computed against the wrong counterfactual. Refusing here is
    cheaper than discovering it in the contrast.
    """

    mismatches = []
    for trajectory_id, record in sorted(meta.items()):
        followed = base_verdicts.get((record["base_trajectory_id"], record["turn"] - 1))
        expected = (None if followed is None or any(b is None for b in followed)
                    else [i for i, b in enumerate(followed) if not b])
        if expected != record["violated_indices"]:
            mismatches.append({"trajectory_id": trajectory_id, "recorded": record["violated_indices"],
                               "base_readout_says": expected})
    if mismatches:
        raise SystemExit(
            f"{len(mismatches)} branch row(s) record a target the base readout does not produce, "
            f"e.g. {mismatches[0]}. The arm and the readout are not a matched pair.")
    return {"n_branch_rows_checked": len(meta),
            "criterion": "recorded violated_indices == constraints the base readout calls broken at t-1",
            "n_mismatches": 0}


def paired_units(rows: list[dict], meta: dict[str, dict], which: str) -> list[dict]:
    """One record per (item, seed, turn, constraint), carrying both branches.

    `which="violated"` selects the constraints broken at t-1 -- the primary
    quantity's unit, the thing a targeted reminder actually names. `which="held"`
    selects the complement, which the targeted reminder is silent about.

    A branch point where either branch failed to parse is dropped whole rather
    than half: keeping the parsed side would compare a measured branch against
    a missing one. The count of these is reported.
    """

    by_key: dict[tuple[str, int, int], dict[str, dict]] = {}
    for row in rows:
        record = meta[row["trajectory_id"]]
        key = (record["item_id"], record["seed"], record["turn"])
        by_key.setdefault(key, {})[record["branch"]] = {"row": row, "meta": record}

    units, dropped_unparsed, dropped_incomplete = [], 0, 0
    for (item_id, seed, turn), branches in sorted(by_key.items()):
        if BLANKET not in branches or TARGETED not in branches:
            dropped_incomplete += 1
            continue
        followed = {name: branches[name]["row"]["followed"] for name in (BLANKET, TARGETED)}
        if any(f is None or any(b is None for b in f) for f in followed.values()):
            dropped_unparsed += 1
            continue
        violated = branches[TARGETED]["meta"]["violated_indices"]
        indices = (violated if which == "violated"
                   else [i for i in range(len(followed[TARGETED])) if i not in violated])
        for index in indices:
            units.append({
                "item_id": item_id, "seed": seed, "turn": turn, "constraint_index": index,
                "n_violated_at_t_minus_1": len(violated),
                "targeted": float(followed[TARGETED][index]),
                "blanket": float(followed[BLANKET][index]),
                "delta": float(followed[TARGETED][index]) - float(followed[BLANKET][index]),
            })
    return units, {"branch_points_dropped_unparsed": dropped_unparsed,
                   "branch_points_dropped_incomplete": dropped_incomplete}


def contrast(gates, units: list[dict], field: str, seed: int) -> dict:
    """The paired difference, bootstrapped over items, with the MDE recomputed
    from this arm's own between-item spread.

    The point estimate is the mean over UNITS and the bootstrap resamples
    ITEMS, exactly as on every other contrast on this line: a dialogue is the
    independent unit, the turns and constraints inside one are not.
    """

    if not units:
        raise SystemExit(f"no paired units for {field}: nothing to contrast")

    def statistic(sample: list[dict]) -> float:
        return statistics.fmean(unit[field] for unit in sample) if sample else None

    point, ci, boot_sd = gates._item_bootstrap(units, statistic, seed)

    by_item: dict[str, list[float]] = {}
    for unit in units:
        by_item.setdefault(unit["item_id"], []).append(unit[field])
    item_means = {item: statistics.fmean(values) for item, values in by_item.items()}
    sd_items = statistics.stdev(item_means.values()) if len(item_means) > 1 else 0.0
    mde = POWER_Z * sd_items / math.sqrt(len(item_means)) if item_means else None

    by_seed: dict[int, list[float]] = {}
    for unit in units:
        by_seed.setdefault(unit["seed"], []).append(unit[field])
    per_seed = [statistics.fmean(by_seed[s]) for s in sorted(by_seed)]
    sd_seeds = statistics.stdev(per_seed) if len(per_seed) > 1 else 0.0

    return {
        "point": point, "ci": ci, "bootstrap_sd": boot_sd,
        "n_units": len(units), "n_items": len(item_means), "n_seeds": len(per_seed),
        "paired_by": "(item_id, seed, turn, constraint_index)",
        "bootstrap_over": "item_id",
        "per_seed": {str(s): v for s, v in zip(sorted(by_seed), per_seed)},
        "reportable": f"{statistics.fmean(per_seed):+.4f} +/- {sd_seeds:.4f} (n={len(per_seed)} seeds)",
        "sd_between_items": sd_items,
        "mde_at_80pct_this_arm": mde,
        "ci_excludes_zero": bool(ci[0] * ci[1] > 0),
        "abs_effect_over_mde_this_arm": abs(point) / mde if mde else None,
    }


def rate_by_branch(units: list[dict]) -> dict:
    """The two levels the contrast is a difference of. A contrast near zero
    reads differently at 0.95 than at 0.30, and the primary number alone does
    not say which arm of a ceiling we are on."""

    return {
        "targeted": statistics.fmean(unit["targeted"] for unit in units),
        "blanket": statistics.fmean(unit["blanket"] for unit in units),
        "n_units": len(units),
    }


def net_y_contrast(gates, rows: list[dict], meta: dict[str, dict], base_rows: list[dict], seed: int) -> dict:
    """`y_graded` at turn t on the three branches: targeted, blanket, and the
    archived u=0 row the branches hang off.

    The u=0 comparison is secondary by construction and says so here. The two
    new branches share a sampling seed (common random numbers); the archived
    row was drawn at another time under sampled decoding and is NOT RNG-paired
    with them, so it can show direction and whether the blanket effect
    reproduces, but its precision is not the primary's.
    """

    base_y = {(row["trajectory_id"], row["turn"]): row["y_graded"] for row in base_rows}
    by_key: dict[tuple[str, int, int], dict[str, dict]] = {}
    for row in rows:
        record = meta[row["trajectory_id"]]
        by_key.setdefault((record["item_id"], record["seed"], record["turn"]), {})[record["branch"]] = (row, record)

    units = []
    for (item_id, seed_id, turn), branches in sorted(by_key.items()):
        if BLANKET not in branches or TARGETED not in branches:
            continue
        y_targeted = branches[TARGETED][0]["y_graded"]
        y_blanket = branches[BLANKET][0]["y_graded"]
        if y_targeted is None or y_blanket is None:
            continue
        y_zero = base_y.get((branches[TARGETED][1]["base_trajectory_id"], turn))
        units.append({
            "item_id": item_id, "seed": seed_id, "turn": turn,
            "targeted_minus_blanket": y_targeted - y_blanket,
            "targeted_minus_zero": None if y_zero is None else y_targeted - y_zero,
            "blanket_minus_zero": None if y_zero is None else y_blanket - y_zero,
        })

    out = {"n_branch_points": len(units),
           "targeted_minus_blanket": contrast(gates, units, "targeted_minus_blanket", seed)}
    for field in ("targeted_minus_zero", "blanket_minus_zero"):
        paired = [unit for unit in units if unit[field] is not None]
        out[field] = contrast(gates, paired, field, seed)
        out[field]["rng_paired"] = False
        out[field]["note"] = ("the archived u=0 row was sampled at another time and is not RNG-paired "
                              "with the branches; direction and reproduction only")
    out["targeted_minus_blanket"]["rng_paired"] = True
    return out


def cost_block(rows: list[dict], meta: dict[str, dict], primary: dict) -> dict:
    """Insertion cost per branch, and the equal-token reading item 13 asks for
    when the recovery rates tie."""

    tokens: dict[str, list[int]] = {}
    for row in rows:
        tokens.setdefault(meta[row["trajectory_id"]]["branch"], []).append(row["inserted_tokens"])
    median = {branch: statistics.median(values) for branch, values in tokens.items()}
    ratio = (median[BLANKET] / median[TARGETED]) if median.get(TARGETED) else None
    return {
        "inserted_tokens_median": median,
        "inserted_tokens_mean": {branch: statistics.fmean(values) for branch, values in tokens.items()},
        "blanket_over_targeted": ratio,
        "equal_token_reading": (
            f"a targeted reminder costs {median.get(TARGETED)} inserted tokens against "
            f"{median.get(BLANKET)} for a blanket one ({ratio:.2f}x). At a recovery contrast of "
            f"{primary['point']:+.4f} the targeted action buys "
            + ("the same recovery for fewer tokens" if abs(primary["point"]) < primary["mde_at_80pct_this_arm"]
               else "a different recovery rate, so the cost reading is secondary to the contrast")
            if ratio else "insertion costs were not recorded"),
    }


def by_n_violated(gates, violated_units: list[dict], held_units: list[dict], seed: int) -> dict:
    """How the two contrasts move with the number of constraints broken at t-1.

    EXPLORATORY, and labelled as such in the report: it is not the signed
    estimand and no verdict reads it. It exists because S3's arm table has to
    say what the controller optimises over, and that depends on whether the
    choice between a targeted and a blanket reminder is a function of the
    state at all. At 3 broken constraints the targeted block IS the blanket
    block (a unit test pins that), so this cell is a built-in null: a contrast
    that is not ~0 there would mean the two actions differ by something other
    than which constraints they name.
    """

    out = {}
    for n in sorted({unit["n_violated_at_t_minus_1"] for unit in violated_units}):
        cell = {"n_violated_at_t_minus_1": n}
        for name, units in (("recovery_of_violated", violated_units), ("retention_of_held", held_units)):
            subset = [unit for unit in units if unit["n_violated_at_t_minus_1"] == n]
            cell[name] = (contrast(gates, subset, "delta", seed) if subset
                          else {"point": None, "n_units": 0,
                                "note": "no units: a targeted reminder naming all k constraints "
                                        "leaves no constraint unnamed"})
        out[str(n)] = cell
    out["status"] = ("EXPLORATORY -- design input for the section 6 arm table, not a reported "
                     "result and not promotable to the primary (screening section 10 item 13)")
    return out


def verdict(primary: dict) -> dict:
    """Item 13's branches, and the one case it did not write down.

    PASS needs both halves -- a CI excluding zero AND an effect at or above the
    MDE -- for the reason K3 taught this line: significance under the design's
    own MDE is a point estimate detected at less than 80% power, and it is
    likely inflated by the noise that made it detectable.

    A contrast under the MDE is pre-registered as UNDECIDABLE and explicitly
    NOT as "targeting does not work". Item 13 wrote consequences for pass and
    for fail only, so the consequence of UNDECIDABLE is a ruling this script
    does not have; it says so rather than defaulting one way and calling it
    pre-registered.
    """

    mde = primary["mde_at_80pct_this_arm"]
    resolved = primary["ci_excludes_zero"] and abs(primary["point"]) >= mde
    if resolved and primary["point"] > 0:
        return {"verdict": "PASS", "closes_koopman_mpc": False,
                "reason": "the CI excludes 0, the contrast is positive and at or above this arm's own "
                          "MDE: a targeted reminder recovers broken constraints better than a blanket "
                          "one, which no fixed schedule can imitate",
                "consequence": "rewrite the plan's section 6 arm table around a targeted executor; "
                               "S3 is the main-line action"}
    if resolved:
        return {"verdict": "FAIL", "closes_koopman_mpc": True,
                "reason": "the CI excludes 0 and the contrast is at or above the MDE, but NEGATIVE: "
                          "targeting is resolvably worse than blanket reminding on the constraints it "
                          "names. The action dimension does not rescue the closed loop",
                "consequence": "cut the `koopman_mpc` arm; S3 keeps only the open-loop comparison"}
    return {"verdict": "UNDECIDABLE", "closes_koopman_mpc": None,
            "reason": (f"|{primary['point']:+.4f}| sits below this arm's 80%-power MDE "
                       f"{mde:.4f}" + ("" if primary["ci_excludes_zero"] else " and the CI crosses 0")
                       + ". Pre-registered reading: this design resolves only a doubling of the "
                         "executor, so a smaller true effect is invisible here. NOT evidence that "
                         "targeting does not work"),
            "consequence": "NEEDS A USER RULING: item 13 signed consequences for PASS and FAIL only. "
                           "The default in .claude/global.md (a null closes the line) and the clause's "
                           "own refusal to read a sub-MDE number as a negative point in opposite "
                           "directions, and this script is not entitled to pick between them"}


def main() -> None:
    args = parse_args()
    if args.out_path.exists():
        raise SystemExit(f"refusing to overwrite existing {args.out_path}")
    gates = _sibling_module("analyze_sequor_s0_0_gates.py")

    readout = gates.load_readout(args.independent_readout, "independent")
    base = gates.load_readout(args.base_readout, "independent")
    arm_dir = pathlib.Path(readout["arm_dir"])
    arm_report = json.loads((arm_dir / "arm_report.json").read_text())
    cap_record = gates.refuse_if_the_cap_bound(arm_report, args.exclude_item)

    meta = branch_metadata(arm_dir)
    base_verdicts = {(row["trajectory_id"], row["turn"]): row["followed"] for row in base["rows"]}
    targeting_check = verify_targets_against_base(meta, base_verdicts)

    rows = [row for row in readout["rows"] if row["item_id"] not in args.exclude_item]
    base_rows = [row for row in base["rows"]
                 if row["branch"] == BASE_ARM and row["item_id"] not in args.exclude_item]

    violated_units, drops = paired_units(rows, meta, "violated")
    held_units, _ = paired_units(rows, meta, "held")

    primary = contrast(gates, violated_units, "delta", args.seed)
    primary["mde_at_80pct_pre_registered"] = PRE_REGISTERED_MDE
    primary["abs_effect_over_mde_pre_registered"] = abs(primary["point"]) / PRE_REGISTERED_MDE
    primary["levels"] = rate_by_branch(violated_units)

    secondary_held = contrast(gates, held_units, "delta", args.seed)
    secondary_held["levels"] = rate_by_branch(held_units)

    ruling = verdict(primary)

    base_arm_report_path = pathlib.Path(base["arm_dir"]) / "arm_report.json"
    gold = json.loads(base_arm_report_path.read_text()).get("gold_coverage", {})

    report = {
        "arm_dir": str(arm_dir),
        "base_arm_dir": base["arm_dir"],
        "provenance": readout.get("provenance"),
        "agent_model": readout["agent_model"], "judge_model": readout["judge_model"],
        "judge_kind": readout["judge_kind"],
        "seeds": arm_report["seeds"], "n_turns": arm_report["n_turns"],
        "n_branch_points": arm_report["n_branch_points"],
        "branch_points_by_n_violated": arm_report["branch_points_by_n_violated"],
        "response_cap": cap_record,
        "n_rows_unusable": readout.get("n_rows_unusable"),
        "targeting_check": targeting_check,
        "prefix_check": arm_report["prefix_check"],
        "dropped": drops,
        "estimand": {
            "primary": "recovery of constraints violated at t-1: targeted minus blanket",
            "secondary": ["retention of constraints held at t-1", "net y_graded", "cost per insertion"],
            "signed": "2026-09-11, before submission (screening section 10 item 13)",
            "rule": "no secondary quantity may be promoted to primary after the fact",
        },
        "primary": primary,
        "secondary_retention_of_held": secondary_held,
        "secondary_net_y": net_y_contrast(gates, rows, meta, base_rows, args.seed),
        "cost": cost_block(rows, meta, primary),
        "exploratory_by_n_violated": by_n_violated(gates, violated_units, held_units, args.seed),
        "in_loop_judge_discount": arm_report["targeting_agreement"],
        "verdict": ruling,
        "caveat": (
            f"{gold.get('n_constraints_judged', 0) - gold.get('n_constraints_in_calibration_set', 0)}"
            f"/{gold.get('n_constraints_judged', 0)} "
            f"({gold.get('share_outside_calibration_set', 0):.0%}) of the judged constraints lie "
            "outside the judge's calibration set; its accuracy there was never measured. That noise "
            "attenuates the contrast toward zero, so a small number here has two explanations this "
            "arm cannot separate (screening section 10 item 9)."
        ),
    }

    print(f"targeting check: {targeting_check['n_branch_rows_checked']} rows, "
          f"{targeting_check['n_mismatches']} mismatches; dropped {drops}")
    print(f"\nPRIMARY  recovery of constraints violated at t-1, targeted minus blanket")
    print(f"  {primary['reportable']}   bootstrap CI [{primary['ci'][0]:+.4f}, {primary['ci'][1]:+.4f}] "
          f"over {primary['n_items']} items, {primary['n_units']} constraint-observations")
    print(f"  levels: targeted {primary['levels']['targeted']:.4f}  blanket {primary['levels']['blanket']:.4f}")
    print(f"  MDE(this arm) {primary['mde_at_80pct_this_arm']:.4f}  "
          f"(pre-registered {PRE_REGISTERED_MDE})  "
          f"|effect|/MDE {primary['abs_effect_over_mde_this_arm']:.2f}")
    print(f"\nSECONDARY  retention of constraints already held at t-1, targeted minus blanket")
    print(f"  {secondary_held['reportable']}   CI [{secondary_held['ci'][0]:+.4f}, "
          f"{secondary_held['ci'][1]:+.4f}]   levels: targeted "
          f"{secondary_held['levels']['targeted']:.4f}  blanket {secondary_held['levels']['blanket']:.4f}")
    net = report["secondary_net_y"]
    for name in ("targeted_minus_blanket", "targeted_minus_zero", "blanket_minus_zero"):
        block = net[name]
        print(f"SECONDARY  net y {name}: {block['point']:+.4f}  CI [{block['ci'][0]:+.4f}, {block['ci'][1]:+.4f}]")
    print(f"\ncost: {report['cost']['inserted_tokens_median']}")
    print("\nEXPLORATORY by number broken at t-1 (design input for the S3 arm table, not a result):")
    for n, cell in sorted((k, v) for k, v in report["exploratory_by_n_violated"].items() if k.isdigit()):
        recovery, retention = cell["recovery_of_violated"], cell["retention_of_held"]
        held = (f"{retention['point']:+.4f} (n={retention['n_units']})" if retention["point"] is not None
                else "no unnamed constraints")
        print(f"  {n} broken: recovery {recovery['point']:+.4f} (n={recovery['n_units']})   "
              f"retention {held}")
    print(f"\nVERDICT {ruling['verdict']} -- {ruling['reason']}")
    print(f"  consequence: {ruling['consequence']}")
    print(f"\n{report['caveat']}")

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_path.write_text(json.dumps(report, indent=2))
    print(f"\nreport written to {args.out_path}")


if __name__ == "__main__":
    main()
