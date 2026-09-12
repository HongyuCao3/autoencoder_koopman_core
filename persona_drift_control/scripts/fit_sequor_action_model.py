#!/usr/bin/env python3
"""Fit the model S3's `koopman_mpc` arm plans with, and sign the arm table
(docs/experiments/constraint_retention_plan.md section 6). CPU only, no GPU.

This is the "S2 signs the arm table" step the plan requires before S3 may be
written, redone for the action space the targeted-reminder gate opened. It
estimates the per-constraint transition on every arm this line has already
run, prints the cell table so the thin cells are visible, and then reports what
the fitted model PREDICTS the four arms will do at the signed budget.

The prediction is a design input, not a result and not a gate. This line has
twice had a paper simulation under-call reality -- the operator's steady state
missed S1a's endpoint by 2.8x -- so the prediction is here to answer one
question only: at this budget, do the arms have room to differ at all? If the
model says every arm lands in the same place, the GPU time buys a zero that was
knowable for free; if it says they separate, the size it predicts is still not
the number the arm will report.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from persona_drift.sequor_bank import gold_constraints, load_sequor_bank, select_bank_items
from persona_drift.sequor_controllers import (  # noqa: E402
    NONE, best_fixed_schedule, broken_indices, equal_cost_random_schedule, fit_action_model,
    greedy_action, joint_transition, mpc_action, observations_from_branches,
    observations_from_rows, open_loop_value, plan_table)

SCHEDULE_ARMS = ("zero_control", "constant_remind", "bernoulli", "antithetic")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--s1-readout", type=pathlib.Path, required=True)
    p.add_argument("--branch-readout", type=pathlib.Path, required=True)
    p.add_argument("--branch-arm-dir", type=pathlib.Path, required=True)
    p.add_argument("--bank", type=pathlib.Path, required=True)
    p.add_argument("--gold", type=pathlib.Path, required=True)
    p.add_argument("--n-items", type=int, default=40)
    p.add_argument("--n-turns", type=int, default=20)
    p.add_argument("--late-from", type=int, default=15)
    p.add_argument("--budget-share", type=float, default=0.5,
                   help="Budget as a share of the full-dose blanket spend, per item. Signed at 0.5 "
                        "(2026-09-12), the same 'half the full dose' convention as G-S2-4.")
    p.add_argument("--out-path", type=pathlib.Path, required=True, help="Must not already exist.")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def word_tokens(text: str) -> int:
    """A tokenizer-free stand-in used ONLY for sizing the budget offline.

    The arm itself prices actions with the agent's own tokenizer; this is here
    so the plan and the budget can be computed without loading a model, and the
    two are cross-checked in the arm's dry run.
    """

    return len(text.split())


def main() -> None:
    args = parse_args()
    if args.out_path.exists():
        raise SystemExit(f"refusing to overwrite existing {args.out_path}")

    s1 = json.loads(args.s1_readout.read_text())
    branch = json.loads(args.branch_readout.read_text())
    for report, name in ((s1, args.s1_readout), (branch, args.branch_readout)):
        if report["judge_kind"] != "independent":
            raise SystemExit(f"{name}: judge_kind={report['judge_kind']!r}, expected 'independent'")

    meta = {}
    with (args.branch_arm_dir / "trajectories.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            meta[row["trajectory_id"]] = {"branch": row["branch"], "turn": row["turn"],
                                          "violated_indices": row["violated_indices"],
                                          "base_trajectory_id": row["base_trajectory_id"]}

    base_verdicts = {(row["trajectory_id"], row["turn"]): row["followed"] for row in s1["rows"]}
    observations = (
        observations_from_rows(s1["rows"], lambda row: range(3) if row["u_remind"] else ())
        + observations_from_branches(branch["rows"], meta, base_verdicts))
    model = fit_action_model(observations, k=3)

    items = select_bank_items(load_sequor_bank(args.bank, n_turns=args.n_turns),
                              gold_constraints(args.gold), args.n_items)

    print(f"{len(observations)} constraint-transitions over {len(model.counts)} cells")
    print(f"{'prev_followed':>14} {'named':>6} {'n_named':>8} {'p(followed next)':>18} {'n':>7}")
    for cell in sorted(model.counts):
        print(f"{str(cell[0]):>14} {str(cell[1]):>6} {cell[2]:>8} "
              f"{model.probabilities[cell]:>18.4f} {model.counts[cell]:>7}")

    rng = random.Random(args.seed)
    per_item = []
    for item in items:
        blanket = word_tokens(item.constraint_block)
        budget = int(round(args.budget_share * blanket * (args.n_turns - 1)))
        table = plan_table(item, model, budget, word_tokens, args.n_turns, args.late_from)
        fixed = best_fixed_schedule(item, model, budget, word_tokens, args.n_turns, args.late_from)
        random_schedule = equal_cost_random_schedule(item, budget, word_tokens, args.n_turns, rng)
        per_item.append({
            "item_id": item.conversation_id, "blanket_cost": blanket, "budget": budget,
            "max_blanket_reminders": budget // blanket,
            "mpc_expected": rollout_closed_loop(item, model, table, budget, args.n_turns,
                                                args.late_from, mode="mpc"),
            "greedy_expected": rollout_closed_loop(item, model, table, budget, args.n_turns,
                                                   args.late_from, mode="greedy"),
            "fixed_expected": open_loop_value(model, fixed, args.late_from),
            "random_expected": open_loop_value(model, random_schedule, args.late_from),
            "zero_expected": open_loop_value(model, [0] * args.n_turns, args.late_from),
            "fixed_schedule": fixed,
        })

    def mean(field: str) -> float:
        return sum(record[field] for record in per_item) / len(per_item)

    predicted = {
        "mpc_minus_fixed": mean("mpc_expected") - mean("fixed_expected"),
        "mpc_minus_greedy": mean("mpc_expected") - mean("greedy_expected"),
        "mpc_minus_random": mean("mpc_expected") - mean("random_expected"),
        "fixed_minus_zero": mean("fixed_expected") - mean("zero_expected"),
        "levels": {name: mean(f"{name}_expected")
                   for name in ("mpc", "greedy", "fixed", "random", "zero")},
    }

    report = {
        "n_observations": len(observations),
        "cells": {str(cell): {"p": model.probabilities[cell], "n": model.counts[cell]}
                  for cell in sorted(model.counts)},
        "budget_share": args.budget_share, "n_turns": args.n_turns, "late_from": args.late_from,
        "n_items": len(items),
        "budget_unit": "words of the inserted constraint block (offline stand-in; the arm prices "
                       "actions with the agent's own tokenizer and the dry run cross-checks them)",
        "per_item": per_item,
        "predicted_separation": predicted,
        "status": "DESIGN INPUT, not a result and not a gate. The model has under-called this "
                  "line's own measurements before (the operator's steady state missed S1a by 2.8x); "
                  "it is read here only to check the arms have room to differ at this budget.",
    }
    print(f"\npredicted at budget = {args.budget_share:.0%} of full dose "
          f"(median {sorted(r['max_blanket_reminders'] for r in per_item)[len(per_item)//2]} "
          f"blanket reminders of {args.n_turns - 1} turns):")
    for name, value in predicted["levels"].items():
        print(f"  {name:>7} late-window y  {value:.4f}")
    for name in ("mpc_minus_fixed", "mpc_minus_greedy", "mpc_minus_random", "fixed_minus_zero"):
        print(f"  {name:>18} {predicted[name]:+.4f}")

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_path.write_text(json.dumps(report, indent=2))
    print(f"\nreport written to {args.out_path}")


def rollout_closed_loop(item, model, table, budget: int, n_turns: int, late_from: int,
                        mode: str) -> float:
    """Expected late-window `y` of a closed-loop policy, propagated exactly.

    The belief is a distribution over the 8 joint states and over the remaining
    budget, which do not factorise once the action depends on the state -- so
    this carries (state, tokens_left) pairs forward rather than a scalar. Exact,
    not sampled, for the same reason `joint_transition` is enumerated.
    """

    belief = {((True,) * model.k, budget): 1.0}
    total, n = 0.0, 0
    for turn in range(1, n_turns + 1):
        nxt: dict[tuple[tuple[bool, ...], int], float] = {}
        for (state, tokens_left), weight in belief.items():
            if turn == 1:
                action = NONE
            elif mode == "mpc":
                action = mpc_action(table, turn, state, tokens_left)
            else:
                action = greedy_action(item, state, tokens_left, word_tokens)
            spent = table["costs"][state][action][0]
            for new_state, probability in joint_transition(model.transition(state, action)):
                if probability == 0.0:
                    continue
                key = (new_state, tokens_left - spent)
                nxt[key] = nxt.get(key, 0.0) + weight * probability
        belief = nxt
        if turn >= late_from:
            total += sum(weight * sum(state) / model.k for (state, _), weight in belief.items())
            n += 1
    return total / n if n else 0.0


if __name__ == "__main__":
    main()
