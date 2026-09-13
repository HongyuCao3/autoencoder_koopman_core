#!/usr/bin/env python3
"""Table 2's `defense` column: six rows, aggregated the way the report caliber
requires (docs/article/MAIN_TABLE_DESIGN.md section 2).

This replaces nothing. `analyze_budget_arm_comparison.py` stays as the Phase J
record; it pools all 40 trajectories into one mean and bootstraps over
trajectories. Two things had to change for the main table, and both are
caliber, not taste:

1. `mean +/- std (n=seed)`. `.claude/global.md` bans single-seed and 2-seed
   point estimates outright, so every row has to be an aggregate over seeds
   with the seed count as `n`. That is why G1 reran the two endpoint arms to
   five seeds -- they were the only arms left at two. The std is a SAMPLE std
   (`ddof=1`), signed 2026-09-13; the repository previously carried both
   conventions (MAIN_TABLE_DESIGN.md section 6 item 5). The deciding argument
   was that this table mixes seed counts -- `defense` runs n=5, `constraint`
   n=3 -- and `ddof=0` shrinks a column by sqrt((n-1)/n), which is 18.4% at
   n=3 against 10.6% at n=5. The column with FEWER seeds would print the
   tighter error bar. `ddof=1` has no n-dependent factor.

2. The paired CI resamples ATTACKS, not trajectories. There are 8 attacks and
   5 seeds; resampling 40 trajectories independently treats five seeds of one
   attack as five independent observations and understates the interval. The
   shared `surrogate_eval.bootstrap_ci` resamples by group, which is also what
   every Table 1 cell and the `constraint` column use -- the point of the
   table is that its intervals are constructed the same way across columns.

The readout is self-judged (`.claude/global.md`'s single named exception for
this line). Self-judging is a one-directional miss that systematically
understates the effect, every quotation of these numbers carries that sentence
and `judge_kind=self`, and the exception does not travel to any other column.

CPU-only.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib

import numpy as np

from surrogate_eval import bootstrap_ci, seed_aggregate

SCORES_NAME = "trajectories.jsonl"
LATE_WINDOW = (3, 4, 5)
# Signed 2026-09-13. `seed_aggregate` has no default for this on purpose.
REPORT_DDOF = 1

# name -> (directory, Table 2 row role). The role is what the paper prints;
# the directory is where it came from. `randsched_p75` and the four losing
# fixed schedules are carried as diagnostics, not rows -- p75 drew 0.900
# reminders/trajectory rather than 1.000, so it is cost-matched to within 20%
# and not the equal-cost control (2026-09-07 audit correction,
# docs/experiments/adaptive_vs_fixed_claim_plan.md section 11.2 item 2).
ARMS = {
    "zero_control": ("outputs/koopman_defense_phaseE_zero_control_5seed", "row:zero_control"),
    "constant_remind": ("outputs/koopman_defense_phaseE_constant_remind_5seed", "row:full_dose"),
    "randsched_p100": ("outputs/koopman_defense_phaseJ_budget1_randsched_p100", "row:equal_cost_random"),
    "fixed_t1": ("outputs/koopman_defense_phaseJ_budget1_fixed_t1", "candidate:fixed_schedule"),
    "fixed_t2": ("outputs/koopman_defense_phaseJ_budget1_fixed_t2", "candidate:fixed_schedule"),
    "fixed_t3": ("outputs/koopman_defense_phaseJ_budget1_fixed_t3", "candidate:fixed_schedule"),
    "fixed_t4": ("outputs/koopman_defense_phaseJ_budget1_fixed_t4", "candidate:fixed_schedule"),
    "fixed_t5": ("outputs/koopman_defense_phaseJ_budget1_fixed_t5", "candidate:fixed_schedule"),
    "threshold": ("outputs/koopman_defense_phaseJ_budget1_threshold", "row:threshold_feedback"),
    "koopman_mpc": ("outputs/koopman_defense_phaseJ_budget1_koopman", "row:ours"),
    "randsched_p75": ("outputs/koopman_defense_phaseJ_budget1_randsched_p75", "diagnostic:not_cost_matched"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument(
        "--out-path",
        type=pathlib.Path,
        default=pathlib.Path("outputs/koopman_case_study/table2_defense_column.json"),
    )
    return parser.parse_args()


def load_arm(name: str, directory: pathlib.Path) -> dict:
    """One arm's per-trajectory metrics, plus the provenance fields the guards
    below compare across arms."""

    path = directory / SCORES_NAME
    if not path.exists():
        raise SystemExit(f"{name}: {path} missing")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    by_trajectory = collections.defaultdict(list)
    for row in rows:
        by_trajectory[(row["attack_id"], row["seed"])].append(row)

    metrics = {}
    for key, turns in by_trajectory.items():
        turns = sorted(turns, key=lambda row: row["turn"])
        ys = {row["turn"]: float(row["y_safety"]) for row in turns}
        late = [ys[turn] for turn in LATE_WINDOW if turn in ys]
        if not late:
            raise SystemExit(f"{name} {key}: no turn in the late window {LATE_WINDOW}")
        metrics[key] = {
            "terminal_y": ys[max(ys)],
            "late_y": float(np.mean(late)),
            "n_reminders": float(sum(int(row["u_remind"]) for row in turns)),
            "inserted_tokens": float(sum(int(row.get("inserted_tokens") or 0) for row in turns)),
        }

    return {
        "path": str(path),
        "design": sorted({row["excitation_design"] for row in rows}),
        "model": sorted({row["model"] for row in rows}),
        "judge_model": sorted({row["judge_model"] for row in rows}),
        "n_judge_parse_failures": sum(bool(row.get("judge_parse_failure")) for row in rows),
        "n_refusals": sum(bool(row.get("refusal_flag")) for row in rows),
        "per_trajectory": metrics,
    }


def guard(arms: dict[str, dict]) -> dict:
    """Raises before a single number is computed. Each check exists because
    its failure would be INVISIBLE in the finished table -- a mixed judge, a
    stale directory reused for two arms, or an arm missing a seed all produce
    a table that renders fine and means something else."""

    problems = []
    for name, arm in arms.items():
        for field in ("design", "model", "judge_model"):
            if len(arm[field]) != 1:
                problems.append(f"{name}: {len(arm[field])} distinct {field} in one directory: {arm[field]}")

    judges = {arm["judge_model"][0] for arm in arms.values() if len(arm["judge_model"]) == 1}
    if len(judges) > 1:
        problems.append(f"arms do not share a judge: {sorted(judges)} -- self-judged and independently judged rows cannot go in one column")
    models = {arm["model"][0] for arm in arms.values() if len(arm["model"]) == 1}
    if len(models) > 1:
        problems.append(f"arms do not share an agent model: {sorted(models)}")

    designs = collections.Counter(arm["design"][0] for arm in arms.values() if len(arm["design"]) == 1)
    for design, count in designs.items():
        if count > 1:
            problems.append(f"design {design!r} appears in {count} arms -- two arms are reading the same controller")

    keysets = {name: set(arm["per_trajectory"]) for name, arm in arms.items()}
    shared = set.intersection(*keysets.values())
    for name, keys in keysets.items():
        if keys != shared:
            problems.append(f"{name}: {len(keys)} trajectories vs {len(shared)} shared -- arms are not on the same (attack, seed) grid")

    seeds = sorted({key[1] for key in shared})
    if len(seeds) < 3:
        problems.append(f"{len(seeds)} seeds: report caliber requires >= 3")

    if problems:
        raise SystemExit("guard failed:\n  " + "\n  ".join(problems))

    return {
        "judge_model": sorted(judges)[0],
        "judge_kind": "self" if judges == models else "independent",
        "agent_model": sorted(models)[0],
        "seeds": seeds,
        "attacks": sorted({key[0] for key in shared}),
        "shared_trajectories": sorted(shared),
    }


def per_seed_means(arm: dict, metric: str, seeds: list[int], shared: list[tuple]) -> list[float]:
    """One number per seed -- the mean over attacks. `n` in `mean +/- std (n)`
    is the seed count, so the attack dimension has to be collapsed first."""

    return [
        float(np.mean([arm["per_trajectory"][key][metric] for key in shared if key[1] == seed]))
        for seed in seeds
    ]


def main() -> None:
    args = parse_args()
    arms = {name: load_arm(name, pathlib.Path(directory)) for name, (directory, _) in ARMS.items()}
    context = guard(arms)
    seeds, shared = context["seeds"], context["shared_trajectories"]

    levels = {}
    for name, arm in arms.items():
        levels[name] = {"role": ARMS[name][1], "design": arm["design"][0]}
        for metric in ("terminal_y", "late_y", "n_reminders", "inserted_tokens"):
            values = per_seed_means(arm, metric, seeds, shared)
            aggregate = seed_aggregate(values, ddof=REPORT_DDOF)
            levels[name][metric] = {
                "mean": aggregate["mean"],
                "std": aggregate["std"],
                "n_seeds": len(seeds),
                "per_seed": values,
                "reportable": f"{aggregate['mean']:.4f} +/- {aggregate['std']:.4f} (n={len(seeds)} seeds)",
            }

    # The best fixed schedule is chosen on the same data it is then compared
    # against, on `late_y`. That is the pre-registered choice
    # (docs/experiments/budget_constrained_defense_plan.md section 5) and it
    # is deliberately biased AGAINST the adaptive arm. It is also a SELECTION
    # signal, so the chosen arm's own level is reported as a row while the
    # selection itself is recorded here rather than quoted as a result.
    fixed = {name: arm for name, arm in levels.items() if ARMS[name][1] == "candidate:fixed_schedule"}
    best_fixed = max(fixed, key=lambda name: fixed[name]["late_y"]["mean"])
    best_fixed_terminal = max(fixed, key=lambda name: fixed[name]["terminal_y"]["mean"])

    contrasts = {}
    groups = [key[0] for key in shared]
    for name in arms:
        if name == best_fixed:
            continue
        for metric in ("terminal_y", "late_y"):
            diffs = np.array(
                [arms[name]["per_trajectory"][key][metric] - arms[best_fixed]["per_trajectory"][key][metric]
                 for key in shared]
            )
            boot = bootstrap_ci(diffs, groups, seed=args.bootstrap_seed, n_resamples=args.bootstrap)
            # The Phase J record bootstrapped trajectories, and the unit change
            # moves some CIs off zero. Carrying both is not indecision: a
            # reader has to be able to see that the switch was a construction
            # choice made for the whole table and not a choice made for this
            # contrast after seeing it.
            legacy = bootstrap_ci(
                diffs, list(range(len(diffs))), seed=args.bootstrap_seed, n_resamples=args.bootstrap
            )
            contrasts[f"{name}_minus_{best_fixed}::{metric}"] = {
                "point": boot["point"],
                "ci": [boot["ci_low"], boot["ci_high"]],
                "excludes_zero": boot["excludes_zero"],
                "n_attacks_resampled": boot["n_groups"],
                "n_paired_trajectories": boot["n_rows"],
                "trajectory_unit_ci": [legacy["ci_low"], legacy["ci_high"]],
                "trajectory_unit_excludes_zero": legacy["excludes_zero"],
                "unit_change_flips_significance": bool(boot["excludes_zero"] != legacy["excludes_zero"]),
                # How much the attack-clustered interval actually rests on.
                # With 8 clusters, an interval that excludes zero while most
                # clusters sit exactly at zero is carried by two or three
                # attacks, and the caption has to say so.
                "n_attacks_with_nonzero_mean_diff": int(
                    sum(abs(float(np.mean(diffs[[i for i, key in enumerate(shared) if key[0] == attack]]))) > 1e-12
                        for attack in context["attacks"])
                ),
                "n_trajectories_with_zero_diff": int((diffs == 0).sum()),
            }

    report = {
        "caliber": {
            "std_ddof": REPORT_DDOF,
            "std_kind": "sample sd, signed 2026-09-13 (MAIN_TABLE_DESIGN.md section 6 item 5)",
            "bootstrap_unit": "attack_id (paired by (attack_id, seed))",
            "bootstrap_unit_rationale": (
                "table-wide construction: Table 1's `defense` column groups by attack_id and its "
                "`constraint` column by item_id (eval_surrogate_rows_behavioral.py item_col). The "
                "Phase J record used the trajectory as the unit; both are reported per contrast."
            ),
            "n_resamples": args.bootstrap,
            "judge_kind": context["judge_kind"],
            "judge_model": context["judge_model"],
            "agent_model": context["agent_model"],
            "limitation_required_at_every_quotation": (
                "self-judged readout (.claude/global.md named exception, `defense` only): "
                "self-judging is a one-directional miss and systematically understates the effect"
            ),
            "n_seeds": len(seeds),
            "n_attacks": len(context["attacks"]),
        },
        "selection": {
            "best_fixed_schedule_by_late_y": best_fixed,
            "best_fixed_schedule_by_terminal_y": best_fixed_terminal,
            "note": "selection signal, chosen on the comparison data on purpose (biases against the adaptive arm); not a reportable result",
        },
        "levels": levels,
        "contrasts_vs_best_fixed": contrasts,
        "diagnostics": {
            name: {"judge_parse_failures": arm["n_judge_parse_failures"], "refusals": arm["n_refusals"]}
            for name, arm in arms.items()
        },
    }

    print(f"judge={context['judge_kind']} ({context['judge_model']})  "
          f"{len(context['attacks'])} attacks x {len(seeds)} seeds = {len(shared)} trajectories/arm  ddof={REPORT_DDOF}\n")
    print(f"{'arm':<18}{'role':<28}{'terminal_y':>22}{'late_y':>22}{'rem/traj':>10}{'tok/traj':>10}")
    for name, arm in levels.items():
        print(f"{name:<18}{arm['role']:<28}{arm['terminal_y']['reportable']:>22}{arm['late_y']['reportable']:>22}"
              f"{arm['n_reminders']['mean']:>10.3f}{arm['inserted_tokens']['mean']:>10.1f}")

    print(f"\nbest fixed schedule (selected on late_y): {best_fixed}")
    print(f"{'contrast vs ' + best_fixed:<34}{'point':>10}  {'95% CI (attack)':>22}  {'95% CI (trajectory)':>22}  flip")
    for key, value in contrasts.items():
        star = " *" if value["excludes_zero"] else "  "
        legacy_star = " *" if value["trajectory_unit_excludes_zero"] else "  "
        print(f"{key.replace('_minus_' + best_fixed, ''):<34}{value['point']:>+10.4f}  "
              f"[{value['ci'][0]:+.4f}, {value['ci'][1]:+.4f}]{star}  "
              f"[{value['trajectory_unit_ci'][0]:+.4f}, {value['trajectory_unit_ci'][1]:+.4f}]{legacy_star}  "
              f"{'YES' if value['unit_change_flips_significance'] else ''}")

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_path.write_text(json.dumps(report, indent=2))
    print(f"\nreport written to {args.out_path}")


if __name__ == "__main__":
    main()
