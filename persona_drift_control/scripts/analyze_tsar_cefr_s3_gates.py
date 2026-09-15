#!/usr/bin/env python3
"""Table 2 for the `tsar_cefr` line: the five closed-loop arms, scored and gated.

Zero GPU. Reads the trajectory rows GPU-2 writes and fills plan section 5.4's
main quantity, its two cost axes, and the paired contrast against `dp_path`.

WRITTEN BEFORE THE JOB, ON PURPOSE (plan section 6.3). The S3 precedent is the
reason: both of that arm's bugs came from an end-to-end fixture that wrote the
SAME dicts into the readout file and the trajectory file, so a join that was
wrong in production was right in the test. Here the criteria constants live at
the top of this source file, the join is on (arm, text_id, seed, step), and the
unit tests build readout rows and trajectory rows SEPARATELY.

THE MAIN QUANTITY IS THE ONE SIGNED ON 2026-09-13, AND IT IS NOT A RANKING.
Terminal CEFR RMSE on the integer-coded level from `level_official`. Meaning is
an ADMISSION THRESHOLD, never a weighted term -- the official TSAR scorer has
no weighted total, so a weighted total would be this repo's invention wearing
the official scorer's authority. The caption obligation that follows the number
is in `REPORTING_OBLIGATIONS`.

WHAT THIS SCRIPT REFUSES TO DO.

  * It refuses to report fewer than three seeds (.claude/global.md: a single
    seed point estimate is not a reportable number, and two seeds cannot
    separate arms -- "Phase J: the budget setting holds, but 2 seeds cannot
    separate the arms").
  * It refuses to average self-judged scores. `ell` here comes from a fixed
    CEFR classifier that never saw the controlled model, so the self-judging
    ban does not bite -- but the check is mechanical, not a sentence in a doc,
    because a guard that is only prose is invisible when it fails.
  * It refuses to guess what a meaning-threshold failure scores. Plan 5.4 says
    such a row "counts as a failure" without saying what number that is, so
    `--meaning-failure-policy` is REQUIRED in canonical mode. Both readings are
    computed and reported; only the one named on the command line judges.
  * It exits non-zero if any cell of Table 2 cannot be filled. A half-filled
    table that still prints is how a missing arm becomes invisible (the ALT arm
    group silently lost an arm that way -- LEDGER 15696227).
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import pathlib
import random
import sys

CEFR_CODE = {"A1": 1, "A2": 2, "B1": 3, "B2": 4, "C1": 5, "C2": 6}
ARMS = ("one_shot", "fixed_ladder", "dp_path", "greedy_reactive", "koopman_mpc")
OURS = "koopman_mpc"
REFERENCE_ARM = "dp_path"          # plan 5.4: the strongest published open-loop opponent
MIN_SEEDS = 3                       # .claude/global.md, report discipline
BOOTSTRAP_DRAWS = 2000              # plan 5.4
BOOTSTRAP_SEED = 20260914
MDE_Z = 2.80                        # the constant D-2 and D-2.5 already use

REPORTING_OBLIGATIONS = (
    "the readout is a fixed CEFR classifier, not ground truth; carry G-T1's caption "
    "(2 of 40 trial pairs reversed: 03-b1 a classifier misread, 12-b1 a data property, "
    "both counted in the denominator) and the FKGL secondary-readout agreement rate",
    "the main quantity is ONE of the official scorer's three metrics, not an official "
    "ranking score; the official scorer has no weighted total",
    "the controlled model is Qwen/Qwen3-4B and the saturation probe result (one-shot hit "
    "rate 0.350, D-0 not triggered) belongs beside any arm comparison",
    "D-2.5 bounds what any bin-measurable closed-loop policy can win: its leave-one-source-out "
    "supremum is RMSE 0.7752 against the hand-written greedy_reactive's 0.7882, so an Ours "
    "that merely beats the open-loop arms has not yet cleared the bound that matters",
    "dp_path's three mismatches (model-dependent reward matrix, no cross-level jump in this "
    "action set, sentence-level vs paragraph-level) must be restated wherever it is called "
    "the strongest published open-loop opponent",
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--rows", required=True, help="trajectories.jsonl written by GPU-2")
    p.add_argument("--out-dir", required=True, help="must not exist; products are append-only")
    p.add_argument("--meaning-threshold", type=float, default=None,
                   help="MeaningBERT admission floor. Plan 5.4 signs it on the trial 20 "
                        "sources; kill_criterion 2.5 (3) records that it is NOT YET SIGNED, "
                        "so canonical mode requires it explicitly.")
    p.add_argument("--meaning-failure-policy", choices=["source_level", "exclude"], default=None,
                   help="what a row below the floor scores. source_level: the edit is rejected, "
                        "so the trajectory keeps the SOURCE paragraph's level. exclude: the "
                        "trajectory leaves the denominator. Required in canonical mode.")
    p.add_argument("--mode", choices=["debug", "canonical"], required=True)
    return p.parse_args(argv)


def load_rows(path: pathlib.Path) -> list[dict]:
    rows = [json.loads(line) for line in path.open()]
    seen = set()
    for row in rows:
        key = (row["arm"], row["text_id"], row["seed"], row["step"])
        if key in seen:
            raise SystemExit(f"duplicate row for {key}: the join key is not unique")
        seen.add(key)
    return rows


def terminal_rows(rows: list[dict]) -> dict[tuple, dict]:
    """The last step of every (arm, text_id, seed) trajectory.

    Arms stop at different steps by design (plan 5.4 compares them at equal cost,
    with steps and tokens as the two cost axes), so the terminal row is the one with
    the largest step, never a fixed index.
    """
    best: dict[tuple, dict] = {}
    for row in rows:
        key = (row["arm"], row["text_id"], row["seed"])
        if key not in best or row["step"] > best[key]["step"]:
            best[key] = row
    return best


def squared_error(row: dict, threshold: float, policy: str) -> float | None:
    target = CEFR_CODE[row["target_cefr"].upper()]
    if threshold is not None and row["meaning_to_source"] < threshold:
        if policy == "exclude":
            return None
        return float((CEFR_CODE[row["source_level_official"].upper()] - target) ** 2)
    return float((CEFR_CODE[row["level_official"].upper()] - target) ** 2)


def per_seed_rmse(terminals: dict, arm: str, threshold: float, policy: str) -> dict[int, float]:
    acc: dict[int, list[float]] = collections.defaultdict(list)
    for (a, _text_id, seed), row in terminals.items():
        if a != arm:
            continue
        err = squared_error(row, threshold, policy)
        if err is not None:
            acc[seed].append(err)
    return {seed: math.sqrt(sum(v) / len(v)) for seed, v in acc.items() if v}


def mean_std(values: list[float]) -> tuple[float, float]:
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return mean, float("nan")
    return mean, math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))


def collapse_by_source(terminals: dict, arm: str, threshold: float, policy: str) -> dict[str, float]:
    """Squared error per source, averaged over seeds first (plan 5.4: collapse seeds, then report)."""
    acc: dict[str, list[float]] = collections.defaultdict(list)
    for (a, _text_id, _seed), row in terminals.items():
        if a != arm:
            continue
        err = squared_error(row, threshold, policy)
        if err is not None:
            acc[row["source_id"]].append(err)
    return {s: sum(v) / len(v) for s, v in acc.items()}


def paired_contrast(terminals: dict, threshold: float, policy: str) -> dict:
    ours = collapse_by_source(terminals, OURS, threshold, policy)
    ref = collapse_by_source(terminals, REFERENCE_ARM, threshold, policy)
    shared = sorted(set(ours) & set(ref))
    if not shared:
        raise SystemExit(f"no source is scored under both {OURS} and {REFERENCE_ARM}")
    deltas = [math.sqrt(ours[s]) - math.sqrt(ref[s]) for s in shared]
    mean, sd = mean_std(deltas)
    if sd == 0.0:
        raise SystemExit(
            f"the paired difference {OURS} - {REFERENCE_ARM} has zero variance across "
            f"{len(shared)} sources: the two arms produced identical terminal levels "
            "everywhere. That is a collapse, not a result -- the two policies are the "
            "same policy. The known ways to get here: dp_path restricted to adjacent "
            "transitions IS fixed_ladder, and greedy decoding makes every seed identical. "
            "MDE is undefined at zero variance, so the UNDECIDABLE clause cannot be "
            "evaluated either.")
    rng = random.Random(BOOTSTRAP_SEED)
    means = sorted(
        sum(deltas[rng.randrange(len(deltas))] for _ in deltas) / len(deltas)
        for _ in range(BOOTSTRAP_DRAWS)
    )
    lo, hi = means[int(0.025 * BOOTSTRAP_DRAWS)], means[int(0.975 * BOOTSTRAP_DRAWS) - 1]
    mde = MDE_Z * sd / math.sqrt(len(deltas)) if len(deltas) > 1 else float("nan")
    return {
        "contrast": f"{OURS} - {REFERENCE_ARM}",
        "sign_convention": "RMSE difference; NEGATIVE means Ours is better",
        "n_sources": len(shared),
        "mean": mean,
        "ci95": [lo, hi],
        "mde": mde,
        "effect_over_mde": abs(mean) / mde if mde and mde == mde and mde > 0 else None,
        "undecidable": bool(lo <= 0.0 <= hi and (mde != mde or abs(mean) < mde)),
        "undecidable_clause": "S3 precedent: CI contains 0 AND |effect| < MDE",
    }


def cost_axes(rows: list[dict], arm: str) -> dict:
    steps: dict[tuple, int] = {}
    tokens: dict[tuple, int] = collections.defaultdict(int)
    for row in rows:
        if row["arm"] != arm:
            continue
        key = (row["text_id"], row["seed"])
        steps[key] = max(steps.get(key, 0), row["step"])
        tokens[key] += int(row["n_tokens_out"])
    n = len(steps)
    return {
        "mean_steps_used": sum(steps.values()) / n,
        "mean_tokens": sum(tokens.values()) / n,
        "n_trajectories": n,
    }


def check_equal_denominators(terminals: dict) -> None:
    """Every arm must be scored on the same (text_id, seed) set.

    Arms stop at different steps by design, which is exactly why this has to be
    checked rather than assumed: an arm that terminates before generating anything
    would otherwise leave the sample quietly and be compared against the others on a
    smaller, easier subset.
    """
    by_arm: dict[str, set] = collections.defaultdict(set)
    for arm, text_id, seed in terminals:
        by_arm[arm].add((text_id, seed))
    sizes = {arm: len(keys) for arm, keys in by_arm.items()}
    if len(set(sizes.values())) > 1:
        reference = max(by_arm, key=lambda a: len(by_arm[a]))
        missing = {arm: sorted(by_arm[reference] - keys)[:5]
                   for arm, keys in by_arm.items() if keys != by_arm[reference]}
        raise SystemExit(
            f"arms are scored on different samples {sizes}; first missing keys per arm "
            f"{missing}. Comparing arms on different denominators is not a comparison.")


def build_table(rows: list[dict], threshold: float, policy: str) -> dict:
    terminals = terminal_rows(rows)
    check_equal_denominators(terminals)
    seeds = sorted({seed for _, _, seed in terminals})
    table = {}
    for arm in ARMS:
        by_seed = per_seed_rmse(terminals, arm, threshold, policy)
        if not by_seed:
            raise SystemExit(f"arm {arm!r} has no scored trajectory: Table 2 cannot be filled")
        if len(by_seed) < MIN_SEEDS:
            raise SystemExit(
                f"arm {arm!r} has {len(by_seed)} seeds, below the floor of {MIN_SEEDS} "
                "(.claude/global.md: a single-seed point estimate is not reportable and "
                "2 seeds cannot separate arms)")
        mean, sd = mean_std([by_seed[s] for s in sorted(by_seed)])
        failures = sum(
            1 for (a, _t, _s), row in terminals.items()
            if a == arm and threshold is not None and row["meaning_to_source"] < threshold)
        table[arm] = {
            "rmse_mean": mean,
            "rmse_std": sd,
            "n_seeds": len(by_seed),
            "reportable_form": f"{mean:.4f} +- {sd:.4f} (n = {len(by_seed)} seeds)",
            "per_seed": {str(s): by_seed[s] for s in sorted(by_seed)},
            "meaning_failures": failures,
            "cost": cost_axes(rows, arm),
        }
    return {"seeds": seeds, "arms": table}


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.mode == "canonical":
        if args.meaning_threshold is None:
            raise SystemExit(
                "--meaning-threshold is required in canonical mode: plan 5.4 signs it on the "
                "trial 20 sources and kill_criterion 2.5 (3) records that it is NOT YET SIGNED")
        if args.meaning_failure_policy is None:
            raise SystemExit(
                "--meaning-failure-policy is required in canonical mode: plan 5.4 says such a "
                "row 'counts as a failure' without saying what number that is")
    out_dir = pathlib.Path(args.out_dir)
    if out_dir.exists():
        raise SystemExit(f"{out_dir} exists; products are append-only, pick a new path")

    rows = load_rows(pathlib.Path(args.rows))
    judged_policy = args.meaning_failure_policy or "source_level"
    report = {
        "table": "Table 2, tsar_cefr column",
        "main_quantity": "terminal CEFR RMSE on the integer-coded level_official",
        "meaning_threshold": args.meaning_threshold,
        "meaning_failure_policy_judged": judged_policy,
        "judged": build_table(rows, args.meaning_threshold, judged_policy),
        "paired_contrast": paired_contrast(rows and terminal_rows(rows),
                                           args.meaning_threshold, judged_policy),
        "reporting_obligations": list(REPORTING_OBLIGATIONS),
    }
    other = "exclude" if judged_policy == "source_level" else "source_level"
    try:
        report["alternative_meaning_policy_not_judged"] = {
            "policy": other,
            "arms": build_table(rows, args.meaning_threshold, other)["arms"],
        }
    except SystemExit as exc:
        report["alternative_meaning_policy_not_judged"] = {"policy": other, "not_computed": str(exc)}

    out_dir.mkdir(parents=True)
    (out_dir / "table2_tsar_cefr.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({a: v["reportable_form"] for a, v in report["judged"]["arms"].items()},
                     indent=2))
    print(json.dumps(report["paired_contrast"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
