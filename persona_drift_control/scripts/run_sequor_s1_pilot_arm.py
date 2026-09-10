#!/usr/bin/env python3
"""S1 sizing pilot for the `constraint` line (2026-09-09).

THE QUESTION: what is S1a's MDE, actually?

Plan section 4 sizes S1a on an ASSUMPTION -- "per-item difference sd about
0.30, so MDE@80% about 0.13 at n=40" -- and says in the same breath to
recompute it from a smoke run before submitting. That recomputation has never
been possible, because nothing in this line has ever run a SUSTAINED arm. The
S0-0 branch arm measures a one-step gain from a byte-identical prefix; S1a
compares the ENDPOINT of a trajectory that was reminded every turn against one
that never was. Those are different quantities with different variances, and
the ERGO line already paid for treating one MDE as if it belonged to the
other (an endpoint MDE of 0.49 used to excuse one-step gates running at
se 0.013).

So this pilot runs the S1 design at pilot scale and measures the thing:

  zero_control      u=0 everywhere               ]  S1a: the dose contrast,
  constant_remind   u=1 from turn 2              ]  paired by item

  bernoulli         u_t ~ Bern(0.5) from turn 2  ]  S1b: antithetic pairing,
  antithetic        the exact complement         ]  one reminded side per cell

on the SAME 12 items S0-0 used, under the fidelity harness, 3 seeds.

`--item-selection` (added 2026-09-10, after the pilot answered the sizing
question) is how the same four arms run as S1 itself: `screening` is the
pilot's fully gold-covered set and stays the default, `bank` is the whole bank
ordered by conversation_id, which S1 needs because only 16 dialogues are fully
covered. The ruling to use `bank` for S1 is user-signed (2026-09-10); its cost
-- a non-zero share of judged constraints outside the judge's calibration set
-- is recorded in `gold_coverage` and must be reported wherever these items
produce a number.

WHAT IT BUYS, all of it needed before S1 can be sized or submitted:
  - the per-item endpoint difference sd, hence S1a's real MDE at n=40;
  - whether `y` still has range at turn 20 under a SUSTAINED reminder, which
    the branch arm cannot answer (it never walks a reminded trajectory);
  - the u-balance and cell coverage G-S1 checks, on real rows;
  - throughput for the S1 runtime estimate at this batch width.

WHAT THE PILOT IS NOT: with `--item-selection screening` and n=12 this is a
sizing run, not S1, and no K gate reads it. Its endpoint numbers are a variance estimate, not
a result, and the pre-registered consequence of a small pilot effect is to
re-size S1 -- never to declare authority absent.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from persona_drift.run_provenance import provenance  # noqa: E402
from persona_drift.sequor_bank import (  # noqa: E402
    coverage_record,
    gold_constraints,
    load_sequor_bank,
    select_bank_items,
    select_screening_items,
)
from persona_drift.sequor_trajectory import (  # noqa: E402
    CAP_CRITERION,
    SCHEDULE_ARMS,
    BranchArmConfig,
    arm_config_record,
    assert_schedules_are_well_formed,
    build_schedules,
    cap_accounting,
    run_schedule_arm,
)

RESOURCES = pathlib.Path("resources/sequor")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--agent-model", required=True)
    p.add_argument("--out-dir", type=pathlib.Path, required=True, help="Must not already exist.")
    p.add_argument("--tuples-path", type=pathlib.Path, default=RESOURCES / "sequor_tuples3.jsonl")
    p.add_argument("--gold-path", type=pathlib.Path, default=RESOURCES / "sequor_gold_judge_calibration.jsonl")
    p.add_argument("--n-items", type=int, default=12,
                   help="12 = the S0-0 item set (the pilot). S1's N comes from the sizing "
                        "report, not from this default.")
    p.add_argument("--item-selection", choices=["screening", "bank"], default="screening",
                   help="screening = fully gold-covered dialogues only, the pilot's set and the "
                        "default. bank = the whole bank ordered by conversation_id, which S1 must "
                        "use because only 16 dialogues are fully covered (screening section 9.2). "
                        "Under `bank`, the share of judged constraints outside the judge's "
                        "calibration set is non-zero and travels in the artifact.")
    p.add_argument("--n-turns", type=int, default=20)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                   help="Sampled decoding needs >= 3 (screening section 10 item 5).")
    p.add_argument("--arms", nargs="+", default=list(SCHEDULE_ARMS), choices=list(SCHEDULE_ARMS))
    p.add_argument("--constraints-in-system", action="store_true")
    p.add_argument("--decoding", choices=["greedy", "model_default"], default="greedy")
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--system-prompt", default=None)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--max-model-len", type=int, default=49152)
    p.add_argument("--mode", choices=["debug", "canonical"], required=True)
    return p.parse_args()


def endpoint_summary(rows: list[dict], late_from: int) -> dict:
    """Response-length and coverage summary per arm. The variance estimate S1a
    needs is computed after judging, by the analyzer -- `y` does not exist yet
    at this point in the pipeline, and inventing a proxy for it here is how a
    generation script quietly becomes a second readout."""

    by_arm: dict[str, list[dict]] = {}
    for row in rows:
        by_arm.setdefault(row["branch"], []).append(row)
    return {
        arm: {
            "n_rows": len(arm_rows),
            "n_items": len({r["item_id"] for r in arm_rows}),
            "n_seeds": len({r["seed"] for r in arm_rows}),
            "u_mean_from_turn2": statistics.fmean(
                [r["u_remind"] for r in arm_rows if r["turn"] >= 2]),
            "n_reminders": sum(r["u_remind"] for r in arm_rows),
            "inserted_tokens_total": sum(r["inserted_tokens"] or 0 for r in arm_rows),
            "output_tokens_median": statistics.median(r["n_output_tokens"] for r in arm_rows),
            "output_tokens_median_late": statistics.median(
                [r["n_output_tokens"] for r in arm_rows if r["turn"] >= late_from]),
            "echo_jaccard_median": statistics.median(
                [r["echo_jaccard_prev"] for r in arm_rows if r["echo_jaccard_prev"] is not None]),
            "verbatim_repeat_share": statistics.fmean(
                [float(bool(r["echo_verbatim_prev"])) for r in arm_rows
                 if r["echo_verbatim_prev"] is not None]),
        }
        for arm, arm_rows in sorted(by_arm.items())
    }


def cell_coverage(rows: list[dict]) -> dict:
    """G-S1's cell check on the rows that exist: in the antithetic design each
    (item, turn, seed) cell must have exactly one reminded side across the two
    excitation arms. Checked on rows rather than on the schedule, because the
    schedule being right does not prove the runner used it."""

    cells: dict[tuple, list[int]] = {}
    for row in rows:
        if row["branch"] in ("bernoulli", "antithetic") and row["turn"] >= 2:
            cells.setdefault((row["item_id"], row["turn"], row["seed"]), []).append(row["u_remind"])
    complete = [k for k, v in cells.items() if len(v) == 2]
    balanced = [k for k in complete if sum(cells[k]) == 1]
    return {
        "n_cells": len(cells),
        "n_cells_with_both_arms": len(complete),
        "n_cells_with_exactly_one_reminder": len(balanced),
        "cells_are_matched": len(complete) == len(cells) and len(balanced) == len(complete),
    }


def main() -> None:
    args = parse_args()
    if args.out_dir.exists():
        raise SystemExit(f"refusing to write into existing {args.out_dir}")

    prov = provenance(switches={
        "agent_model": args.agent_model, "n_items": args.n_items, "n_turns": args.n_turns,
        "seeds": args.seeds, "arms": args.arms, "max_new_tokens": args.max_new_tokens,
        "mode": args.mode, "system_prompt": args.system_prompt,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "constraints_in_system": args.constraints_in_system, "decoding": args.decoding,
    })
    print(f"harness {prov['git_sha'][:12]}"
          + (f" (+{prov['n_dirty_paths']} uncommitted paths)" if prov["git_dirty"] else "")
          + (f"  job {prov['job_id']}" if prov["job_id"] else ""))

    gold = gold_constraints(args.gold_path)
    bank = load_sequor_bank(args.tuples_path, n_turns=args.n_turns)
    select = select_screening_items if args.item_selection == "screening" else select_bank_items
    items = select(bank, gold, args.n_items)
    coverage = coverage_record(items, gold)
    print(f"item selection {args.item_selection}: {len(items)} dialogues, "
          f"{coverage['share_outside_calibration_set']:.4f} of judged constraints outside the "
          f"judge's calibration set")
    n_gen = len(items) * args.n_turns * len(args.seeds) * len(args.arms)
    print(f"{len(args.arms)} arms x {len(items)} items x {args.n_turns} turns x "
          f"{len(args.seeds)} seeds -> {n_gen} generations"
          + ("   [DEBUG: mode=debug, not evidence]" if args.mode == "debug" else ""))

    # The schedules, and their structural check, BEFORE the model is loaded: a
    # malformed design should cost zero GPU seconds to discover.
    schedules_by_seed = {seed: build_schedules(items, args.n_turns, seed) for seed in args.seeds}
    schedule_checks = {
        seed: assert_schedules_are_well_formed(sched, args.n_turns)
        for seed, sched in schedules_by_seed.items()
    }
    for seed, check in schedule_checks.items():
        print(f"  seed {seed}: bernoulli u mean {check['bernoulli_u_mean']:.4f}, "
              f"turn 1 action-free, antithetic exact")

    from vllm import LLM, SamplingParams

    decoding = {"temperature": 0.0, "top_p": 1.0, "top_k": -1}
    if args.decoding == "model_default":
        from transformers import GenerationConfig
        model_defaults = GenerationConfig.from_pretrained(args.agent_model)
        if not model_defaults.do_sample:
            raise SystemExit(f"{args.agent_model} reports do_sample=false; 'model_default' is greedy")
        decoding = {"temperature": float(model_defaults.temperature),
                    "top_p": float(model_defaults.top_p), "top_k": int(model_defaults.top_k)}
    print(f"decoding={args.decoding} {decoding}  constraints_in_system={args.constraints_in_system}")

    llm = LLM(model=args.agent_model, seed=args.seeds[0], dtype="bfloat16",
              gpu_memory_utilization=args.gpu_memory_utilization, max_model_len=args.max_model_len)
    tokenizer = llm.get_tokenizer()
    block_tokens = {
        item.conversation_id: len(tokenizer.encode(f"\n\n{item.constraint_block}")) for item in items
    }

    rows: list[dict] = []
    configs: dict[str, dict] = {}
    batches = 0
    t0 = time.time()
    for seed in args.seeds:
        config = BranchArmConfig(
            model_id=args.agent_model, n_turns=args.n_turns, seed=seed,
            max_new_tokens=args.max_new_tokens, system_prompt=args.system_prompt,
            constraints_in_system=args.constraints_in_system, **decoding,
        )
        params = SamplingParams(temperature=config.temperature, top_p=config.top_p,
                                top_k=config.top_k, max_tokens=config.max_new_tokens, seed=seed)
        for arm in args.arms:
            clock = {"t0": time.time(), "batches": 0}

            def generate_batch(conversations: list[list[dict]], _p=params, _c=clock, _a=arm,
                               _s=seed) -> list[dict]:
                outputs = llm.chat(conversations, _p, chat_template_kwargs={"enable_thinking": False})
                _c["batches"] += 1
                print(f"  seed {_s} {_a:16s} batch {_c['batches']:3d}  {len(conversations)} seqs  "
                      f"{time.time() - _c['t0']:.0f}s", flush=True)
                return [{
                    "text": o.outputs[0].text, "finish_reason": o.outputs[0].finish_reason,
                    "n_output_tokens": len(o.outputs[0].token_ids), "n_inserted_tokens": None,
                } for o in outputs]

            arm_rows = run_schedule_arm(items, generate_batch, config, run_id=args.out_dir.name,
                                        arm=arm, schedule=schedules_by_seed[seed][arm])
            for row in arm_rows:
                row["inserted_tokens"] = block_tokens[row["item_id"]] if row["u_remind"] else 0
            rows.extend(arm_rows)
            batches += clock["batches"]
            configs[f"seed{seed}:{arm}"] = arm_config_record(config, items)
            print(f"  seed {seed} {arm}: {len(arm_rows)} rows, "
                  f"{(time.time()-clock['t0'])/60:.1f} min", flush=True)
    elapsed = time.time() - t0

    # Rows first, summary second (job 15761810).
    args.out_dir.mkdir(parents=True)
    with (args.out_dir / "trajectories.jsonl").open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{len(rows)} rows written to {args.out_dir / 'trajectories.jsonl'}", flush=True)

    caps = cap_accounting(rows, CAP_CRITERION)
    cells = cell_coverage(rows)
    late_from = max(2, args.n_turns - 5)
    per_arm = endpoint_summary(rows, late_from)
    config = BranchArmConfig(
        model_id=args.agent_model, n_turns=args.n_turns, seed=args.seeds[0],
        max_new_tokens=args.max_new_tokens, system_prompt=args.system_prompt,
        constraints_in_system=args.constraints_in_system, **decoding,
    )

    (args.out_dir / "run_config.json").write_text(json.dumps({
        "mode": args.mode, "arm": "S1 sizing pilot (4 schedule arms)", "provenance": prov,
        "seeds": args.seeds, "arms": args.arms, "decoding_mode": args.decoding,
        "decoding": decoding, "constraints_in_system": args.constraints_in_system,
        "gold_coverage": coverage,
        "schedules": {str(seed): sched for seed, sched in schedules_by_seed.items()},
        "schedule_checks": {str(seed): check for seed, check in schedule_checks.items()},
        "per_arm_config": configs,
        **arm_config_record(config, items),
    }, indent=2, ensure_ascii=False))
    (args.out_dir / "arm_report.json").write_text(json.dumps({
        "mode": args.mode, "n_rows": len(rows), "n_items": len(items),
        "arms": args.arms, "seeds": args.seeds, "n_turns": args.n_turns,
        "decoding_mode": args.decoding, "decoding": decoding,
        "constraints_in_system": args.constraints_in_system,
        "max_new_tokens": args.max_new_tokens,
        "late_turns_from": late_from,
        "per_arm": per_arm,
        "cell_coverage": cells,
        "schedule_checks": {str(seed): check for seed, check in schedule_checks.items()},
        "gold_coverage": coverage,
        **caps,
        "elapsed_s": elapsed, "seconds_per_generation": elapsed / len(rows),
        "batches": batches,
        "not_a_result": "Sizing pilot at n=12. It measures the variance S1a must be sized "
                        "against; it does not test authority, and no K gate reads it.",
    }, indent=2, ensure_ascii=False))

    print(f"\n{len(rows)} rows, {elapsed/60:.1f} min ({elapsed/len(rows):.2f} s/generation)")
    for arm, stats in per_arm.items():
        print(f"  {arm:16s} {stats['n_rows']:4d} rows  u_mean(t>=2) {stats['u_mean_from_turn2']:.3f}  "
              f"reminders {stats['n_reminders']:4d}  cost {stats['inserted_tokens_total']:6d} tok  "
              f"median resp {stats['output_tokens_median']:5.0f} (late {stats['output_tokens_median_late']:5.0f})  "
              f"echo {stats['echo_jaccard_median']:.3f}")
    print(f"\ncells with exactly one reminded side: "
          f"{cells['n_cells_with_exactly_one_reminder']}/{cells['n_cells']} "
          f"(matched: {cells['cells_are_matched']})")
    over = caps["items_over_cap_criterion"]
    print(f"cap: {caps['n_hit_token_cap']}/{len(rows)} ({caps['token_cap_share']*100:.1f}%) overall; "
          f"items over the {CAP_CRITERION*100:.0f}% criterion: {over or 'none'}")
    print(f"\nwritten to {args.out_dir}")
    print("Sizing pilot: it measures a variance. Judging is a separate phase, and the MDE it "
          "implies is for re-sizing S1, never for a verdict on authority.")


if __name__ == "__main__":
    main()
