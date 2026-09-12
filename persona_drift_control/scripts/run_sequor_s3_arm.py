#!/usr/bin/env python3
"""S3: the closed-loop comparison (plan section 6, arm table rewritten
2026-09-12 after the targeted-reminder gate passed). Generation + in-loop
judging; the REPORTING readout is a separate job.

FOUR ARMS AT ONE TOKEN BUDGET. Cost is insertion tokens, the currency the
targeted action actually changes (21 against 41 median in the branch arm), and
every arm gets the same per-item budget:

    koopman_mpc          reads the state, plans against the fitted model
    greedy_targeted      reads the state, no model: name whatever is broken
    best_fixed_schedule  open-loop, the best schedule the model can pick
    equal_cost_random    open-loop, the same spend placed at random

`greedy_targeted` is not in the plan's original three. It is here because
without it a positive MPC result cannot say whether the fitted operator earned
anything or whether merely reacting to the state did, and that is the first
question a reader asks of an MPC table.

WHY THE OPEN-LOOP ARMS CANNOT PLAY THE TARGETED ACTION. "Name the broken
constraints" is undefined without reading the state, so a schedule fixed in
advance can only play `none` or `blanket`. That asymmetry IS the hypothesis
under test -- the targeted-branch gate measured that the targeted action is
worth +0.0748 on the constraints it names -- and it is stated here so the
restriction is never read as a baseline that was starved.

THE STATE COMES FROM THE AGENT'S OWN MODEL, and never reaches a reported
number. Plan section 8 signs the in-loop judge as the served model (self-judge
= selection signal); its known discounts are measured, not assumed: 5% of calls
do not parse at cap 1024 (job 15739196) and it agrees with the reporting judge
on which constraint to target only 48.6% of the time (job 15792569). A turn
whose in-loop readout does not parse plays `none` and is counted.

ZERO CONTROL IS NOT AN ARM HERE. `outputs/sequor_s1_arm/`'s zero_control rows
are the same items, the same harness and the same seeds with u=0 throughout;
regenerating them would spend a quarter of this job's GPU time reproducing rows
that already exist.

NOT SUBMITTED WITHOUT A USER RULING (.claude/experiments.md).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from persona_drift.run_provenance import provenance  # noqa: E402
from persona_drift.sequor_bank import (  # noqa: E402
    coverage_record, gold_constraints, load_sequor_bank, select_bank_items)
from persona_drift.sequor_closed_loop import budget_accounting, run_closed_loop_arm  # noqa: E402
from persona_drift.sequor_constraint_judge import (  # noqa: E402
    CONSTRAINT_JUDGE_PROMPT_TEMPLATE, extract_verdict)
from persona_drift.sequor_controllers import (  # noqa: E402
    POLICIES, ActionModel, action_cost, best_fixed_schedule, equal_cost_random_schedule,
    open_loop_value, plan_table)
from persona_drift.sequor_trajectory import (  # noqa: E402
    CAP_CRITERION, BranchArmConfig, arm_config_record, cap_accounting)

RESOURCES = pathlib.Path("resources/sequor")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--action-model", type=pathlib.Path, required=True,
                   help="fit_sequor_action_model.py report. The controller plans against this and "
                       "nothing else; its cell table is copied into run_config.json so the policy "
                       "can be replayed without refitting.")
    p.add_argument("--agent-model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--tuples-path", type=pathlib.Path, default=RESOURCES / "sequor_tuples3.jsonl")
    p.add_argument("--gold-path", type=pathlib.Path, default=RESOURCES / "sequor_gold_judge_calibration.jsonl")
    p.add_argument("--n-items", type=int, default=50)
    p.add_argument("--n-turns", type=int, default=20)
    p.add_argument("--late-from", type=int, default=15)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--arms", nargs="+", default=list(POLICIES))
    p.add_argument("--budget-share", type=float, required=True,
                   help="Per-item budget as a share of the full-dose blanket spend. Signed before "
                        "submission; the arms are only 'equal cost' relative to this number.")
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--inloop-max-new-tokens", type=int, default=1024,
                   help="1024, not 512: at 512 the 4B judge leaves 8.1%% of calls unparsed and the "
                        "controller goes blind on ~22%% of turns (job 15739196).")
    p.add_argument("--max-model-len", type=int, default=49152)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--constraints-in-system", action="store_true")
    p.add_argument("--decoding", choices=["greedy", "model_default"], default="model_default")
    p.add_argument("--mode", choices=["debug", "canonical", "analysis"], default="canonical")
    p.add_argument("--out-dir", type=pathlib.Path, required=True)
    p.add_argument("--dry-run", action="store_true",
                   help="Build the budgets, plans and schedules and report them; load no model and "
                        "generate nothing.")
    return p.parse_args()


def load_action_model(path: pathlib.Path, k: int) -> ActionModel:
    report = json.loads(path.read_text())
    probabilities, counts = {}, {}
    for key, cell in report["cells"].items():
        prev, named, n_named = eval(key)  # noqa: S307 - the key is this project's own tuple repr
        probabilities[(bool(prev), bool(named), int(n_named))] = cell["p"]
        counts[(bool(prev), bool(named), int(n_named))] = cell["n"]
    return ActionModel(probabilities=probabilities, counts=counts, k=k)


def main() -> None:
    args = parse_args()
    if args.out_dir.exists():
        raise SystemExit(f"refusing to write into existing {args.out_dir}")
    unknown = [arm for arm in args.arms if arm not in POLICIES]
    if unknown:
        raise SystemExit(f"unknown arm(s) {unknown}; expected from {POLICIES}")
    if len(args.seeds) < 3:
        raise SystemExit(f"{len(args.seeds)} seeds: two cannot separate arms (.claude/global.md)")

    prov = provenance(switches={
        "agent_model": args.agent_model, "n_items": args.n_items, "n_turns": args.n_turns,
        "seeds": args.seeds, "arms": args.arms, "budget_share": args.budget_share,
        "max_new_tokens": args.max_new_tokens, "inloop_max_new_tokens": args.inloop_max_new_tokens,
        "mode": args.mode, "constraints_in_system": args.constraints_in_system,
        "decoding": args.decoding, "action_model": str(args.action_model),
    })
    print(f"harness {prov['git_sha'][:12]}"
          + (f" (+{prov['n_dirty_paths']} uncommitted paths)" if prov["git_dirty"] else "")
          + (f"  job {prov['job_id']}" if prov["job_id"] else ""))

    gold = gold_constraints(args.gold_path)
    items = select_bank_items(load_sequor_bank(args.tuples_path, n_turns=args.n_turns), gold,
                              args.n_items)
    coverage = coverage_record(items, gold)
    k = len(items[0].constraints)
    model = load_action_model(args.action_model, k)
    n_gen = len(args.arms) * len(items) * args.n_turns * len(args.seeds)
    print(f"{len(args.arms)} arms x {len(items)} items x {args.n_turns} turns x {len(args.seeds)} "
          f"seeds -> {n_gen} generations + {n_gen * k} in-loop judge calls"
          + ("   [DEBUG: mode=debug, not evidence]" if args.mode == "debug" else ""))
    print(f"{coverage['share_outside_calibration_set']:.4f} of judged constraints lie outside the "
          f"judge's calibration set")

    if args.dry_run:
        count_tokens = lambda text: len(text.split())  # noqa: E731 - see the cross-check below
        print("DRY RUN: costs are word counts, not the agent's tokens; the real run prices actions "
              "with the model's tokenizer and the two are reported side by side in arm_report.json")
    else:
        from vllm import LLM, SamplingParams
        decoding = {"temperature": 0.0, "top_p": 1.0, "top_k": -1}
        if args.decoding == "model_default":
            from transformers import GenerationConfig
            defaults = GenerationConfig.from_pretrained(args.agent_model)
            if not defaults.do_sample:
                raise SystemExit(f"{args.agent_model} reports do_sample=false; 'model_default' is greedy")
            decoding = {"temperature": float(defaults.temperature), "top_p": float(defaults.top_p),
                        "top_k": int(defaults.top_k)}
        print(f"decoding={args.decoding} {decoding}")
        llm = LLM(model=args.agent_model, seed=args.seeds[0], dtype="bfloat16",
                  gpu_memory_utilization=args.gpu_memory_utilization, max_model_len=args.max_model_len)
        tokenizer = llm.get_tokenizer()
        count_tokens = lambda text: len(tokenizer.encode(f"\n\n{text}"))  # noqa: E731

    budgets, tables, fixed_schedules = {}, {}, {}
    t_plan = time.time()
    for item in items:
        blanket = count_tokens(item.constraint_block)
        budget = int(round(args.budget_share * blanket * (args.n_turns - 1)))
        budgets[item.conversation_id] = budget
        tables[item.conversation_id] = plan_table(item, model, budget, count_tokens,
                                                  args.n_turns, args.late_from)
        fixed_schedules[item.conversation_id] = best_fixed_schedule(
            item, model, budget, count_tokens, args.n_turns, args.late_from)
    print(f"plans built for {len(items)} items in {time.time() - t_plan:.1f}s; "
          f"budget median {sorted(budgets.values())[len(budgets)//2]} tokens "
          f"(= {sorted(budgets[i.conversation_id] // count_tokens(i.constraint_block) for i in items)[len(items)//2]} "
          f"blanket reminders of {args.n_turns - 1} turns)")
    print(f"best fixed schedule reminders per item: min "
          f"{min(sum(s) for s in fixed_schedules.values())}, "
          f"max {max(sum(s) for s in fixed_schedules.values())}; "
          f"distinct schedules {len({tuple(s) for s in fixed_schedules.values()})}/{len(items)}")

    if args.dry_run:
        print("\ndry run complete; nothing generated, nothing written")
        return

    rows, configs, batches = [], {}, 0
    t0 = time.time()
    for seed in args.seeds:
        config = BranchArmConfig(
            model_id=args.agent_model, n_turns=args.n_turns, seed=seed,
            max_new_tokens=args.max_new_tokens, constraints_in_system=args.constraints_in_system,
            **decoding)
        params = SamplingParams(temperature=config.temperature, top_p=config.top_p,
                                top_k=config.top_k, max_tokens=config.max_new_tokens, seed=seed)
        judge_params = SamplingParams(temperature=0.0, max_tokens=args.inloop_max_new_tokens, seed=seed)
        random_schedules = {
            item.conversation_id: equal_cost_random_schedule(
                item, budgets[item.conversation_id], count_tokens, args.n_turns,
                random.Random(f"{seed}:{item.conversation_id}"))
            for item in items}

        for arm in args.arms:
            clock = {"t0": time.time(), "batches": 0}

            def generate_batch(conversations, _p=params, _c=clock, _a=arm, _s=seed):
                outputs = llm.chat(conversations, _p, chat_template_kwargs={"enable_thinking": False})
                _c["batches"] += 1
                print(f"  seed {_s} {_a:20s} turn {_c['batches']:3d}  {len(conversations)} seqs  "
                      f"{time.time() - _c['t0']:.0f}s", flush=True)
                return [{"text": o.outputs[0].text, "finish_reason": o.outputs[0].finish_reason,
                         "n_output_tokens": len(o.outputs[0].token_ids)} for o in outputs]

            def judge_batch(pairs, _p=judge_params):
                prompts = [[{"role": "user", "content": CONSTRAINT_JUDGE_PROMPT_TEMPLATE.format(
                    answer=answer, constraint=constraint)}] for constraint, answer in pairs]
                outputs = llm.chat(prompts, _p, chat_template_kwargs={"enable_thinking": False})
                return [extract_verdict(o.outputs[0].text) for o in outputs]

            arm_rows = run_closed_loop_arm(
                items, generate_batch, judge_batch, config, run_id=args.out_dir.name, policy=arm,
                budgets=budgets, tables=tables,
                schedules=(fixed_schedules if arm == "best_fixed_schedule"
                           else random_schedules if arm == "equal_cost_random" else {}),
                count_tokens=count_tokens)
            rows.extend(arm_rows)
            batches += clock["batches"]
            configs[f"seed{seed}:{arm}"] = arm_config_record(config, items)
            print(f"  seed {seed} {arm}: {len(arm_rows)} rows, "
                  f"{(time.time()-clock['t0'])/60:.1f} min", flush=True)
    elapsed = time.time() - t0

    # Rows first, summary second (job 15761810 lost a report, not its rows).
    args.out_dir.mkdir(parents=True)
    with (args.out_dir / "trajectories.jsonl").open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{len(rows)} rows written to {args.out_dir / 'trajectories.jsonl'}", flush=True)

    spend = budget_accounting(rows, args.arms)
    caps = cap_accounting(rows, CAP_CRITERION)
    (args.out_dir / "run_config.json").write_text(json.dumps({
        "mode": args.mode, "arm": "S3 closed loop (4 policies at one token budget)",
        "provenance": prov, "seeds": args.seeds, "arms": args.arms,
        "budget_share": args.budget_share, "budgets": budgets,
        "late_from": args.late_from,
        "action_model_cells": {str(cell): {"p": model.probabilities[cell], "n": model.counts[cell]}
                               for cell in sorted(model.counts)},
        "action_model_path": str(args.action_model),
        "best_fixed_schedules": fixed_schedules,
        "decoding_mode": args.decoding, "decoding": decoding,
        "constraints_in_system": args.constraints_in_system,
        "inloop_judge": {"model": args.agent_model, "kind": "self",
                         "max_new_tokens": args.inloop_max_new_tokens,
                         "note": "selection signal only; never reported"},
        "gold_coverage": coverage,
        "per_arm_config": configs,
        **arm_config_record(BranchArmConfig(
            model_id=args.agent_model, n_turns=args.n_turns, seed=args.seeds[0],
            max_new_tokens=args.max_new_tokens,
            constraints_in_system=args.constraints_in_system, **decoding), items),
    }, indent=2, ensure_ascii=False))
    (args.out_dir / "arm_report.json").write_text(json.dumps({
        "mode": args.mode, "n_rows": len(rows), "n_items": len(items), "arms": args.arms,
        "seeds": args.seeds, "n_turns": args.n_turns, "late_from": args.late_from,
        "budget_share": args.budget_share,
        "max_new_tokens": args.max_new_tokens,
        "budget_accounting": spend,
        "predicted_by_the_model": {
            arm: (open_loop_value(model, fixed_schedules[items[0].conversation_id], args.late_from)
                  if arm == "best_fixed_schedule" else None)
            for arm in args.arms},
        "gold_coverage": coverage, **caps,
        "elapsed_s": elapsed, "seconds_per_generation": elapsed / len(rows), "batches": batches,
        "not_a_result": "Generation and in-loop judging only. Every reportable number comes from "
                        "the independent judge in the scoring job; y_inloop is a selection signal.",
    }, indent=2, ensure_ascii=False))

    print(f"\n{len(rows)} rows, {elapsed/60:.1f} min ({elapsed/len(rows):.2f} s/generation)")
    for arm, stats in spend.items():
        print(f"  {arm:20s} spent {stats['share_of_budget_spent']:.2%} of budget  "
              f"actions {stats['n_actions']:4d} (targeted {stats['n_targeted_actions']:4d} / "
              f"blanket {stats['n_blanket_actions']:4d})  blind {stats['blind_turn_share']:.1%}")
    print(f"\nwritten to {args.out_dir}")
    print("Scoring is a separate job: the independent judge is the only source of a reported number.")


if __name__ == "__main__":
    main()
