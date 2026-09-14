#!/usr/bin/env python3
"""G-S1 admission, recomputed on GPU-1's rows under a NAMED ITEM EXCLUSION.
Zero GPU: no model is loaded, no text is generated, no readout is re-run. The
3600 rows on disk are read and the signed criteria are applied again.

WHY THIS SCRIPT EXISTS. GPU-1 (15850523) came back `admitted: false` on one of
four admission blocks: cap pressure, computed PER ITEM against a 5% ceiling.
The arm truncated exactly ONE generation out of 3600 (`51-a2`, seed 1, step 1,
`half_step_down`, 206 tokens into a 206-token cap; pooled share 0.00028). That
single truncation lands on an item holding 18 rows (6 steps x 3 seeds), so the
finest non-zero rate the design can express is 1/18 = 0.0556 -- already above
the 5% line. ON THIS DESIGN THE THRESHOLD IS ZERO TOLERANCE, which is not what
it was written to mean: it was carried over from the `constraint` line's S1,
where an item carried an order of magnitude more rows.

THE REMEDY IS PRE-REGISTERED, NOT INVENTED HERE. Plan section 4.1 signed the
cap policy as "count the truncated rows and drop per item at the 5% line, per
the S1 precedent". The S1 precedent is job 15763577: exclusion applied AT THE
GATE STEP, the rows themselves left untouched and still scored, so the ruling
stays reversible. No criterion is modified by this script. What changed on
2026-09-14 is only that a user ruling named which item to drop -- required,
because the gate discipline is to report and hand back, never to route around
a gate on the analyzer's own authority.

WHAT THE CAP CHECK IS ACTUALLY GUARDING. `ell_t` is read off the GENERATED
text. A truncated paragraph is short and reads simpler, so its `ell_t` records
"it was cut off" and not "the model simplified it" -- contamination pointing
the same way as the effect we want to find. Pooling hides it: S1 measured 18.6%
overall while two of twelve items ate 75 of 87 truncations and one item was cut
100% of the time. Hence per item, never pooled.

THREE GUARDS, BECAUSE AN EXCLUSION FLAG IS EXACTLY THE KIND OF TOOL THAT GROWS.

  1. `--exclude-item` refuses any item the UNEXCLUDED cap check did not flag.
     The escape hatch reaches the items the gate itself named, and nothing
     else. Dropping an item for any other reason is selecting the sample by
     the result.
  2. Every admission block is recomputed on the retained rows -- not just the
     one that failed. Removing rows moves the row count, the action balance
     and the per-step range too, and a gate that only re-runs its own failure
     is not a gate.
  3. The report carries the before-exclusion verdict, the after-exclusion
     verdict, and the excluded item's own numbers IN THE SAME FILE as the
     numbers the exclusion changed. A reader who suspects the drop flattered
     the actuator can check it without loading the rows.

THE CRITERIA ARE IMPORTED FROM THE RUNNER, NOT RETYPED. `action_balance`,
`cap_pressure`, `per_step_range` and the constants come from
run_tsar_cefr_excitation_arm.py, so this re-score cannot drift from the
verdict the GPU job computed (the G-T1 rule: one place where criteria live).
The runner imports vLLM lazily inside its generation function, so importing it
here loads no CUDA.

WHAT IT DOES NOT DECIDE. D-2 (executor authority) and D-2.5 (the causal upper
bound) stay open. Gate discipline: they are not computed until admission
passes, and D-2.5 must be computed on the full retained N at once -- the same
in-sample bound read +0.0469 * on a 16-trajectory pilot and +0.0042 (crossing
zero) on 40 (kill_criterion.md section 3 (1)).
"""

from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import math
import pathlib
import random
import statistics
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
PDC_ROOT = REPO_ROOT / "persona_drift_control"
sys.path.insert(0, str(PDC_ROOT / "src"))

from persona_drift.run_provenance import provenance             # noqa: E402

DEFAULT_ARM_DIR = PDC_ROOT / "outputs" / "tsar_cefr_gpu1"
RUNNER = PDC_ROOT / "scripts" / "run_tsar_cefr_excitation_arm.py"

# The user ruling that named the item, recorded beside the number it changes.
POWER_Z = 2.80                      # two-sided alpha=0.05 at 80% power, as everywhere on this line
BOOTSTRAP_DRAWS = 10000             # as G-T1 on this line
BOOTSTRAP_SEED = 20260914           # fixed so the interval reproduces from the artifact
PAIR_UNIT = "source_id"             # plan section 4.4; the unit G-T1 already fixed for this check

EXCLUSION_RULING = "2026-09-14 user ruling (option 1 of three): named exclusion, zero GPU"
EXCLUSION_CLAUSE = "plan section 4.1 (cap policy, S1 precedent); S1 precedent = job 15763577"


def load_runner():
    """The criteria, from the runner itself. vLLM is imported inside the
    runner's generation function, so this loads no GPU stack."""
    spec = importlib.util.spec_from_file_location("tsar_arm", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


arm = load_runner()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm-dir", type=pathlib.Path, default=DEFAULT_ARM_DIR)
    p.add_argument("--gate", choices=["admission", "d2"], default="admission",
                   help="admission = G-S1; d2 = executor authority, which REFUSES to run "
                        "unless admission passes on the same rows")
    p.add_argument("--exclude-item", action="append", default=[],
                   help="text_id to drop from the gate. Refused unless the unexcluded "
                        "cap check flagged it. Repeatable.")
    p.add_argument("--out-path", type=pathlib.Path, default=None)
    return p.parse_args()


def read_jsonl(path: pathlib.Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# --------------------------------------------------------------------------
# guard 2: the grid, rebuilt from the rows instead of from the schedule
# --------------------------------------------------------------------------

def grid_completeness(rows: list[dict]) -> dict:
    """The runner checked `len(rows) == len(schedule) * n_steps`. The schedule
    is not on disk, so the count is rebuilt from the rows -- and made stricter
    than a count while we are here: every (item, seed, step) cell must appear
    exactly once. A count alone passes a run that dropped one cell and
    duplicated another."""
    items = sorted({row["text_id"] for row in rows})
    seeds = sorted({row["seed"] for row in rows})
    steps = sorted({row["step"] for row in rows})
    cells = collections.Counter((row["text_id"], row["seed"], row["step"]) for row in rows)
    expected = len(items) * len(seeds) * len(steps)
    missing = [c for c in ((i, s, t) for i in items for s in seeds for t in steps)
               if cells[c] == 0]
    duplicated = sorted(c for c, n in cells.items() if n > 1)
    return {"value": len(rows), "expected": expected,
            "n_items": len(items), "n_seeds": len(seeds), "n_steps": len(steps),
            "n_missing_cells": len(missing), "missing_cells": [list(c) for c in missing[:20]],
            "n_duplicated_cells": len(duplicated),
            "duplicated_cells": [list(c) for c in duplicated[:20]],
            "passed": len(rows) == expected and not missing and not duplicated}


def admission(rows: list[dict]) -> dict:
    """The four signed blocks, all of them, on whatever rows are handed in."""
    blocks = {"row_count": grid_completeness(rows),
              "action_balance": arm.action_balance(rows),
              "cap_pressure": arm.cap_pressure(rows),
              "per_step_range": arm.per_step_range(rows)}
    return {"blocks": blocks, "admitted": all(b["passed"] for b in blocks.values())}


# --------------------------------------------------------------------------
# guard 1: the escape hatch reaches only what the gate named
# --------------------------------------------------------------------------

def validate_exclusions(requested: list[str], unexcluded: dict, rows: list[dict]) -> None:
    flagged = set(unexcluded["blocks"]["cap_pressure"]["items_over_cap"])
    present = {row["text_id"] for row in rows}
    for item in requested:
        if item not in present:
            raise SystemExit(f"--exclude-item {item}: no such text_id in the rows")
        if item not in flagged:
            raise SystemExit(
                f"--exclude-item {item}: the unexcluded cap check did not flag it "
                f"(flagged: {sorted(flagged) or 'none'}). The exclusion flag reaches the "
                f"items the gate named and nothing else; dropping any other item selects "
                f"the sample by the result.")


# --------------------------------------------------------------------------
# guard 3: what leaving costs, reported beside what staying would have cost
# --------------------------------------------------------------------------

def direction_diagnostic(rows: list[dict], retained: list[dict], item: str) -> dict:
    """Did the drop flatter the actuator? Answered here, not left to the reader.

    THIS IS NOT D-2'S ANSWER AND MUST NEVER BE CITED AS EXECUTOR AUTHORITY.
    D-2 pairs by source text at the same state and bootstraps by source; this
    is a POOLED per-action mean, which mixes trajectory position with state
    (tsar_cefr_results.md says so under the D-2 heading). It is here for one
    narrower question: the exclusion removes rows, so does it remove rows that
    disagreed with the actuator's intended direction?
    """
    def gap(pool: list[dict]) -> float | None:
        by: dict[str, list[float]] = collections.defaultdict(list)
        for row in pool:
            by[row["action"]].append(row["level_expected"])
        if not by["copy"] or not by["step_down"]:
            return None
        return statistics.fmean(by["copy"]) - statistics.fmean(by["step_down"])

    per_item = {t: gap([r for r in rows if r["text_id"] == t])
                for t in {r["text_id"] for r in rows}}
    scored = {t: v for t, v in per_item.items() if v is not None}
    order = sorted(scored, key=scored.get)
    return {
        "quantity": "pooled mean(level_expected | copy) - mean(level_expected | step_down)",
        "sign_convention": "positive = step_down reads simpler than doing nothing",
        "all_rows": gap(rows),
        "retained_rows": gap(retained),
        "excluded_item": scored.get(item),
        "excluded_item_rank_from_most_backward": order.index(item) + 1 if item in scored else None,
        "n_items_scored": len(scored),
        "n_items_backward": sum(1 for v in scored.values() if v < 0),
        "not_the_d2_answer": "pooled per-action means mix trajectory position with state; "
                             "D-2 pairs by source at the same state and bootstraps by source",
    }


def excluded_item_profile(rows: list[dict], retained: list[dict], item: str) -> dict:
    """The selection-bias question, answerable from this file alone: was the
    dropped item the one where the actuator had to work hardest?"""
    mine = [row for row in rows if row["text_id"] == item]
    hits = [row for row in mine if row["hit_token_cap"]]

    def by_action(pool: list[dict]) -> dict:
        out: dict[str, list[float]] = collections.defaultdict(list)
        for row in pool:
            out[row["action"]].append(row["level_expected"])
        return {a: statistics.fmean(v) for a, v in sorted(out.items())}

    def tokens(pool: list[dict]) -> dict:
        vals = sorted(row["n_tokens_out"] for row in pool)
        return {"median": statistics.median(vals), "max": max(vals)}

    return {
        "text_id": item,
        "n_rows_removed": len(mine),
        "cap_hits": len(hits),
        "cap_hit_share": len(hits) / len(mine),
        "truncated_rows": [{"seed": r["seed"], "step": r["step"], "action": r["action"],
                            "n_tokens_out": r["n_tokens_out"]} for r in hits],
        "action_shares": {a: sum(1 for r in mine if r["action"] == a) / len(mine)
                          for a in sorted({r["action"] for r in mine})},
        "level_expected_mean": statistics.fmean(r["level_expected"] for r in mine),
        "level_expected_sd": statistics.stdev(r["level_expected"] for r in mine),
        "level_expected_by_action": by_action(mine),
        "source_level_expected": mine[0]["source_level_expected"],
        "n_tokens_out": tokens(mine),
        "retained_pool_for_contrast": {
            "level_expected_mean": statistics.fmean(r["level_expected"] for r in retained),
            "level_expected_by_action": by_action(retained),
            "source_level_expected_mean": statistics.fmean(
                r["source_level_expected"] for r in
                {r["text_id"]: r for r in retained}.values()),
            "n_tokens_out": tokens(retained),
        },
        "direction_diagnostic": direction_diagnostic(rows, retained, item),
        "note": "reported so the drop can be judged without reloading the rows; "
                "these numbers are diagnostics, not results",
    }


# --------------------------------------------------------------------------
# D-2: executor authority (plan section 4.4, K3's form)
# --------------------------------------------------------------------------

def with_input_level(rows: list[dict]) -> list[dict]:
    """Attach the level of each step's INPUT, so a row carries its own one-step
    change instead of a pooled mean.

    This is what "at the same state" buys: `level_expected` alone mixes where
    the trajectory had drifted to with what the action did to it, which is why
    the pooled per-action mean is not this gate's answer. Step 1's input is the
    source paragraph; step t's input is step t-1's output, byte-identical by
    construction of the harness.
    """
    by_traj: dict[tuple, dict[int, dict]] = collections.defaultdict(dict)
    for row in rows:
        by_traj[(row["text_id"], row["seed"])][row["step"]] = row
    out = []
    for steps in by_traj.values():
        for step, row in sorted(steps.items()):
            prev = steps.get(step - 1)
            level_in = prev["level_expected"] if prev else row["source_level_expected"]
            out.append({**row, "level_in": level_in,
                        "delta_level": row["level_expected"] - level_in})
    return out


def _bootstrap_ci(clusters: list[list[float]]) -> tuple[float, float, float]:
    """Resampled over clusters (sources), not rows: a source's steps and seeds
    are not independent draws."""
    rng = random.Random(BOOTSTRAP_SEED)
    n = len(clusters)
    draws = []
    for _ in range(BOOTSTRAP_DRAWS):
        picked = [clusters[rng.randrange(n)] for _ in range(n)]
        draws.append(statistics.fmean([v for c in picked for v in c]))
    draws.sort()
    return (draws[int(0.025 * BOOTSTRAP_DRAWS)], draws[int(0.975 * BOOTSTRAP_DRAWS)],
            statistics.stdev(draws))


def d2_contrast(rows: list[dict], steps: tuple[int, ...] | None = None) -> dict:
    """`step_down` minus `copy` on the NEXT-STEP change in `ell`, paired inside
    a source and bootstrapped over sources.

    Signed before the data was read (plan section 4.4, D-2 in
    kill_criterion.md section 2): RESOLVED needs the CI to exclude zero AND the
    effect to reach this arm's own MDE. A contrast below the MDE is a point
    estimate detected at less than 80% power -- the lesson K3 taught this line.
    """
    pool = [r for r in rows if steps is None or r["step"] in steps]
    by_source: dict[str, dict[str, list[float]]] = collections.defaultdict(
        lambda: collections.defaultdict(list))
    for row in pool:
        by_source[row[PAIR_UNIT]][row["action"]].append(row["delta_level"])

    paired: dict[str, float] = {}
    unpairable = []
    for source, acts in by_source.items():
        if acts["step_down"] and acts["copy"]:
            paired[source] = statistics.fmean(acts["step_down"]) - statistics.fmean(acts["copy"])
        else:
            unpairable.append(source)

    if len(paired) < 2:
        # The caller decides what this means: for the primary it is fatal, for
        # the confirmatory it is a fact about the excitation draw, recorded and
        # not allowed to take the primary down with it.
        return {"computable": False, "n_sources_paired": len(paired),
                "n_sources_unpairable": len(unpairable), "n_rows": len(pool),
                "steps_used": "all" if steps is None else list(steps),
                "reason": "fewer than two sources have both `step_down` and `copy` rows "
                          "in this window; the random excitation did not draw the pair"}

    clusters = [[v] for v in paired.values()]
    point = statistics.fmean(paired.values())
    lo, hi, boot_sd = _bootstrap_ci(clusters)
    sd_sources = statistics.stdev(paired.values())
    mde = POWER_Z * sd_sources / math.sqrt(len(paired))

    by_seed: dict[int, dict[str, list[float]]] = collections.defaultdict(
        lambda: collections.defaultdict(list))
    for row in pool:
        by_seed[row["seed"]][row["action"]].append(row["delta_level"])
    per_seed = {s: statistics.fmean(a["step_down"]) - statistics.fmean(a["copy"])
                for s, a in sorted(by_seed.items()) if a["step_down"] and a["copy"]}
    seed_vals = list(per_seed.values())

    return {
        "computable": True,
        "quantity": "mean(delta_level | step_down) - mean(delta_level | copy)",
        "sign_convention": "NEGATIVE is the working direction: step_down should drive "
                           "`ell` down further than doing nothing",
        "steps_used": "all" if steps is None else list(steps),
        "point": point, "ci95": [lo, hi], "bootstrap_sd": boot_sd,
        "paired_by": PAIR_UNIT, "bootstrap_over": PAIR_UNIT,
        "draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED,
        "n_sources_paired": len(paired), "n_sources_unpairable": len(unpairable),
        "sources_unpairable": sorted(unpairable)[:20],
        "n_rows": len(pool),
        "sd_between_sources": sd_sources,
        "mde_at_80pct_this_arm": mde,
        "abs_effect_over_mde": abs(point) / mde if mde else None,
        "ci_excludes_zero": bool(lo * hi > 0),
        "per_seed": {str(s): v for s, v in per_seed.items()},
        "reportable": (f"{statistics.fmean(seed_vals):+.4f} +/- "
                       f"{statistics.stdev(seed_vals):.4f} (n={len(seed_vals)} seeds)")
        if len(seed_vals) > 1 else None,
    }


def d2_verdict(primary: dict, confirmatory: dict) -> dict:
    """The pre-registered branch, and nothing else. D-2 fires when the CI
    contains 0 OR the effect is under this arm's MDE (kill_criterion.md
    section 2); firing closes the line, and NOT by swapping the judge."""
    reached = primary["ci_excludes_zero"] and primary["abs_effect_over_mde"] >= 1.0
    right_way = primary["point"] < 0
    if reached and right_way:
        verdict, fires = "RESOLVED", False
        reason = ("the CI excludes 0 and the effect is at or above this arm's own MDE, "
                  "in the working direction: the executor has authority over one step")
    elif reached and not right_way:
        verdict, fires = "RESOLVED_WRONG_DIRECTION", True
        reason = ("the CI excludes 0 and the effect reaches the MDE, but step_down drives "
                  "`ell` UP relative to doing nothing -- authority with the sign reversed "
                  "is not authority")
    else:
        verdict, fires = "UNDECIDABLE", True
        reason = (f"|{primary['point']:+.4f}| is {primary['abs_effect_over_mde']:.2f}x this "
                  f"arm's 80%-power MDE {primary['mde_at_80pct_this_arm']:.4f}"
                  + ("" if primary["ci_excludes_zero"] else "; the CI also crosses 0"))
    agree = ((primary["point"] < 0) == (confirmatory["point"] < 0)
             if confirmatory.get("computable") else None)
    return {"death_condition": "D-2", "verdict": verdict, "fires": fires, "reason": reason,
            "confirmatory_computable": bool(confirmatory.get("computable")),
            "confirmatory_agrees_in_sign": agree,
            "confirmatory_note": "step 1 only: every seed of an item starts from the SAME "
                                 "source paragraph, so the states being contrasted are "
                                 "identical rather than merely conditioned on",
            "consequence_if_fires": "close `tsar_cefr`; do NOT swap the readout "
                                    "(the one instrument swap is already spent)"}


def d2_report(rows: list[dict], excluded: list[str]) -> dict:
    """D-2 on the retained rows, and only if the gate ahead of it passed.

    The refusal is the point. A death condition computed on rows the admission
    gate rejected is a number with no standing, and the arm that produced it
    said so itself (`still_open` in arm_report.json). It is checked here rather
    than trusted to the operator's memory."""
    unexcluded = admission(rows)
    validate_exclusions(excluded, unexcluded, rows)
    retained = [row for row in rows if row["text_id"] not in set(excluded)]
    gate = admission(retained)
    if not gate["admitted"]:
        failing = [n for n, b in gate["blocks"].items() if not b["passed"]]
        raise SystemExit(
            f"refusing to judge D-2: G-S1 does not pass on these rows ({', '.join(failing)}). "
            f"A death condition computed on rejected rows has no standing.")

    dated = with_input_level(retained)
    primary = d2_contrast(dated)
    if not primary.get("computable"):
        raise SystemExit(f"D-2 has no primary contrast: {primary['reason']}")
    confirmatory = d2_contrast(dated, steps=(1,))
    return {
        "gate": "D-2 executor authority (plan section 4.4, K3's form)",
        "line": "tsar_cefr", "arm_job_id": "15850523",
        "admitted_by": "G-S1 on these rows (admission_report_excluded.json)",
        "exclusion": {"items": sorted(excluded), "n_rows_retained": len(retained),
                      "n_rows_total": len(rows)},
        "primary": primary,
        "confirmatory_step1_only": confirmatory,
        "verdict": d2_verdict(primary, confirmatory),
        "still_open": ["D-2.5 causal upper bound"],
        "provenance": provenance({"gate": "d2", "exclude_item": sorted(excluded),
                                  "criteria_source": str(RUNNER)}),
    }


def build_report(rows: list[dict], probe_rows: list[dict], excluded: list[str]) -> dict:
    unexcluded = admission(rows)
    validate_exclusions(excluded, unexcluded, rows)
    retained = [row for row in rows if row["text_id"] not in set(excluded)]
    after = admission(retained)
    return {
        "gate": "G-S1 (admission), recomputed under a named exclusion",
        "line": "tsar_cefr",
        "arm_job_id": "15850523",
        "admitted": after["admitted"],
        "admission": after["blocks"],
        "admission_before_exclusion": unexcluded["blocks"],
        "admitted_before_exclusion": unexcluded["admitted"],
        "exclusion": {
            "items": sorted(excluded),
            "unit": "text_id (a source paragraph paired with one target level); "
                    "the source's other target cell is retained",
            "ruling": EXCLUSION_RULING,
            "pre_registered_by": EXCLUSION_CLAUSE,
            "threshold": arm.MAX_ITEM_CAP_HIT_SHARE,
            "design_resolution": "an item holds n_seeds * n_steps rows; the finest "
                                 "non-zero cap-hit rate it can express is 1/18 = 0.0556",
            "n_rows_total": len(rows),
            "n_rows_retained": len(retained),
            "rows_are_untouched_on_disk": True,
            "profiles": [excluded_item_profile(rows, retained, item) for item in sorted(excluded)],
        },
        "saturation_probe": arm.saturation_probe_verdict(probe_rows),
        "unchanged_verbatim": arm.unchanged_verbatim(retained),
        "still_open": ["D-2 executor authority", "D-2.5 causal upper bound"],
        "caption_obligation": "every number downstream of this gate is computed on "
                              f"{len(retained)} of {len(rows)} rows; the excluded text_ids "
                              "and their own numbers are in this file",
        "provenance": provenance({"exclude_item": sorted(excluded),
                                  "arm_dir": str(DEFAULT_ARM_DIR),
                                  "criteria_source": str(RUNNER)}),
    }


def print_report(report: dict, out_path: pathlib.Path) -> None:
    before = "admitted" if report["admitted_before_exclusion"] else "NOT admitted"
    after = "ADMITTED" if report["admitted"] else "NOT ADMITTED"
    print(f"G-S1 on {report['exclusion']['n_rows_retained']} of "
          f"{report['exclusion']['n_rows_total']} rows "
          f"(excluded: {', '.join(report['exclusion']['items']) or 'none'})")
    print(f"  unexcluded verdict: {before}")
    for name, block in report["admission"].items():
        mark = "ok  " if block["passed"] else "FAIL"
        print(f"  [{mark}] {name}")
    print(f"  -> {after}")
    print(f"  -> {out_path}")


DEFAULT_NAME = {"admission": "admission_report_excluded.json", "d2": "d2_report.json"}


def print_d2(report: dict, out_path: pathlib.Path) -> None:
    v, p = report["verdict"], report["primary"]
    print(f"D-2 {v['verdict']}  (fires: {v['fires']})")
    print(f"  {p['quantity']}")
    print(f"  point {p['point']:+.4f}  CI95 [{p['ci95'][0]:+.4f}, {p['ci95'][1]:+.4f}]  "
          f"MDE {p['mde_at_80pct_this_arm']:.4f}  ({p['abs_effect_over_mde']:.2f}x)")
    print(f"  reportable: {p['reportable']}")
    c = report["confirmatory_step1_only"]
    if c.get("computable"):
        print(f"  step-1-only (identical states): {c['point']:+.4f} "
              f"[{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]  n={c['n_sources_paired']} sources"
              f"  sign agrees: {v['confirmatory_agrees_in_sign']}")
    else:
        print(f"  step-1-only: not computable ({c['reason']})")
    print(f"  -> {v['reason']}")
    print(f"  -> {out_path}")


def main() -> None:
    args = parse_args()
    out_path = args.out_path or (args.arm_dir / DEFAULT_NAME[args.gate])
    if out_path.exists():
        raise SystemExit(f"refusing to overwrite existing {out_path}")
    rows = read_jsonl(args.arm_dir / "trajectories.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.gate == "d2":
        report = d2_report(rows, args.exclude_item)
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print_d2(report, out_path)
        return
    probe_rows = read_jsonl(args.arm_dir / "saturation_probe.jsonl")
    report = build_report(rows, probe_rows, args.exclude_item)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print_report(report, out_path)


if __name__ == "__main__":
    main()
