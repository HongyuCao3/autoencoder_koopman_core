#!/usr/bin/env python3
"""S0-0 counterfactual-branch arm for the `constraint` line
(docs/experiments/constraint_signal_screening.md section 2).

N dialogues x T turns, every turn from the second on generated TWICE from a
byte-identical prefix -- once continuing under u=0, once with the constraint
block restated (u=1). The one-step gain of a reminder is measured per
(item, turn), not regressed for.

Runs in the `constraint` line's own vLLM environment. Generation only: the
graded readout `y_t` is applied afterwards by
scripts/score_sequor_trajectories.py, so both judges (in-loop self and the
independent reporting judge) score the SAME responses and a judge swap never
means re-generating trajectories.

Nothing here decides anything about gates. The gates (K1 readout range, K2
state beyond turn index, K3 executor authority) are computed by
scripts/analyze_sequor_s0_0_gates.py from the scored rows.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from persona_drift.run_provenance import provenance  # noqa: E402
from persona_drift.sequor_bank import (  # noqa: E402
    gold_constraints,
    load_sequor_bank,
    select_screening_items,
)
from persona_drift.sequor_trajectory import (  # noqa: E402
    BranchArmConfig,
    arm_config_record,
    assert_pairs_share_prefix,
    run_branch_arm,
)

RESOURCES = pathlib.Path("resources/sequor")

# Pre-registered 2026-09-09 (screening doc section 10 item 4, and the ruling
# after job 15756689): a response cap that BINDS makes the readout partly a
# truncation measurement. The share is checked PER ITEM, not globally, because
# response length is a function of the constraint set -- 15756689 sat at 18.6%
# overall while 2 of its 12 items accounted for 75 of the 87 truncations
# ("Write a longer dialog", "Write creatively as a story") and the other ten
# were under 13%. A global average hides exactly the failure that matters.
CAP_CRITERION = 0.05


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--agent-model", required=True,
                   help="The system under study. Also the in-loop judge's model (plan section 3).")
    p.add_argument("--out-dir", type=pathlib.Path, required=True, help="Must not already exist.")
    p.add_argument("--tuples-path", type=pathlib.Path, default=RESOURCES / "sequor_tuples3.jsonl")
    p.add_argument("--gold-path", type=pathlib.Path, default=RESOURCES / "sequor_gold_judge_calibration.jsonl")
    p.add_argument("--n-items", type=int, default=12,
                   help="Pre-registered N for the screening arm. Items are the dialogues whose "
                        "k=3 constraints are ALL in the judge-calibration gold set.")
    p.add_argument("--n-turns", type=int, default=20)
    p.add_argument("--seeds", type=int, nargs="+", default=[0],
                   help="One arm per seed, all in this run directory, rows tagged with theirs. "
                        "Greedy decoding needs one; sampled decoding needs >= 3, because the "
                        "counterfactual delta then carries sampling noise on top of the action.")
    p.add_argument("--constraints-in-system", action="store_true",
                   help="Put the constraint block in a system message instead of user turn 1. "
                        "The fidelity arm (job 15761176) showed this only matters TOGETHER with "
                        "sampled decoding: alone it moved retention -0.033 (CI spanning 0), and "
                        "with sampling -0.329, three times the additive prediction.")
    p.add_argument("--decoding", choices=["greedy", "model_default"], default="greedy",
                   help="`model_default` reads temperature/top_p/top_k from the model's own "
                        "generation_config at runtime -- what upstream reported using and never "
                        "printed. Recorded in the run config.")
    p.add_argument("--max-new-tokens", type=int, default=1024,
                   help="Agent response cap. Upstream did not report its generation settings; the "
                        "gold responses run to a median of 429 words (p90 679), so a cap that bites "
                        "would make the readout partly a truncation measurement. The report records "
                        "the share of responses that hit it.")
    p.add_argument("--system-prompt", default=None,
                   help="Default absent: these dialogues carry their constraints in turn 1, and the "
                        "ERGO line showed that adding or removing a system message changes the "
                        "answer to the closed-loop question (LEDGER section 3, prompt_profile).")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--max-model-len", type=int, default=32768,
                   help="A T=20 dialogue reaches ~20 x (question + capped response) tokens; too "
                        "small a value silently drops the early turns, which is where the "
                        "constraints were stated.")
    p.add_argument("--mode", choices=["debug", "canonical"], required=True,
                   help="debug artifacts are never evidence (plan section 8). A smoke run over 2 "
                        "items is what turns the runtime estimate into a measurement.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.out_dir.exists():
        raise SystemExit(f"refusing to write into existing {args.out_dir}")

    prov = provenance(switches={
        "agent_model": args.agent_model, "n_items": args.n_items, "n_turns": args.n_turns,
        "seeds": args.seeds, "max_new_tokens": args.max_new_tokens, "mode": args.mode,
        "system_prompt": args.system_prompt, "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "constraints_in_system": args.constraints_in_system, "decoding": args.decoding,
    })
    print(f"harness {prov['git_sha'][:12]}"
          + (f" (+{prov['n_dirty_paths']} uncommitted paths)" if prov["git_dirty"] else "")
          + (f"  job {prov['job_id']}" if prov["job_id"] else ""))

    items = select_screening_items(
        load_sequor_bank(args.tuples_path, n_turns=args.n_turns),
        gold_constraints(args.gold_path),
        args.n_items,
    )
    print(f"{len(items)} items x {args.n_turns} turns x {len(args.seeds)} seed(s) -> "
          f"{len(items) * args.n_turns * len(args.seeds)} base rows + "
          f"{len(items) * (args.n_turns - 1) * len(args.seeds)} counterfactual rows"
          + ("   [DEBUG: mode=debug, not evidence]" if args.mode == "debug" else ""))

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
    configs: dict[int, dict] = {}
    t0 = time.time()
    for seed in args.seeds:
        config = BranchArmConfig(
            model_id=args.agent_model, n_turns=args.n_turns, seed=seed,
            max_new_tokens=args.max_new_tokens, system_prompt=args.system_prompt,
            constraints_in_system=args.constraints_in_system, **decoding,
        )
        params = SamplingParams(temperature=config.temperature, top_p=config.top_p,
                                top_k=config.top_k, max_tokens=config.max_new_tokens, seed=seed)
        clock = {"t0": time.time(), "batches": 0}

        def generate_batch(conversations: list[list[dict]], _p=params, _c=clock) -> list[dict]:
            outputs = llm.chat(conversations, _p, chat_template_kwargs={"enable_thinking": False})
            _c["batches"] += 1
            print(f"  seed {seed} batch {_c['batches']:3d}  {len(conversations)} seqs  "
                  f"{time.time() - _c['t0']:.0f}s", flush=True)
            return [{
                "text": o.outputs[0].text, "finish_reason": o.outputs[0].finish_reason,
                "n_output_tokens": len(o.outputs[0].token_ids), "n_inserted_tokens": None,
            } for o in outputs]

        seed_rows = run_branch_arm(items, generate_batch, config, run_id=args.out_dir.name)
        for row in seed_rows:
            row["inserted_tokens"] = block_tokens[row["item_id"]] if row["u_remind"] else 0
        rows.extend(seed_rows)
        configs[seed] = arm_config_record(config, items)
        print(f"  seed {seed} done: {len(seed_rows)} rows, {(time.time()-clock['t0'])/60:.1f} min",
              flush=True)
    elapsed = time.time() - t0
    config = BranchArmConfig(
        model_id=args.agent_model, n_turns=args.n_turns, seed=args.seeds[0],
        max_new_tokens=args.max_new_tokens, system_prompt=args.system_prompt,
        constraints_in_system=args.constraints_in_system, **decoding,
    )

    n_pairs = assert_pairs_share_prefix(rows)
    capped = [r for r in rows if r["hit_token_cap"]]
    tokens = sorted(r["n_output_tokens"] for r in rows)

    by_item: dict[str, list[dict]] = {}
    for row in rows:
        by_item.setdefault(row["item_id"], []).append(row)
    cap_by_item = {
        item: {
            "n_rows": len(item_rows),
            "n_hit_token_cap": sum(1 for r in item_rows if r["hit_token_cap"]),
            "token_cap_share": sum(1 for r in item_rows if r["hit_token_cap"]) / len(item_rows),
            "output_tokens_median": sorted(r["n_output_tokens"] for r in item_rows)[len(item_rows) // 2],
            "constraints": item_rows[0]["constraints"],
        }
        for item, item_rows in sorted(by_item.items())
    }
    items_over_criterion = [i for i, v in cap_by_item.items() if v["token_cap_share"] > CAP_CRITERION]

    args.out_dir.mkdir(parents=True)
    with (args.out_dir / "trajectories.jsonl").open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.out_dir / "run_config.json").write_text(json.dumps({
        "mode": args.mode, "arm": "S0-0 counterfactual branch", "provenance": prov,
        "seeds": args.seeds, "decoding_mode": args.decoding, "decoding": decoding,
        "constraints_in_system": args.constraints_in_system,
        "per_seed_config": configs,
        **arm_config_record(config, items),
    }, indent=2, ensure_ascii=False))
    (args.out_dir / "arm_report.json").write_text(json.dumps({
        "mode": args.mode, "n_rows": len(rows), "n_pairs": n_pairs,
        "prefix_check": "every pair shares a byte-identical prefix (assert_pairs_share_prefix)",
        "seeds": args.seeds, "decoding_mode": args.decoding, "decoding": decoding,
        "constraints_in_system": args.constraints_in_system,
        "max_new_tokens": args.max_new_tokens,
        "n_hit_token_cap": len(capped), "token_cap_share": len(capped) / len(rows),
        "cap_criterion": CAP_CRITERION, "token_cap_by_item": cap_by_item,
        "items_over_cap_criterion": items_over_criterion,
        "cap_criterion_pass": not items_over_criterion,
        "output_tokens_median": tokens[len(tokens) // 2], "output_tokens_max": tokens[-1],
        "elapsed_s": elapsed, "seconds_per_generation": elapsed / len(rows),
        "batches": turn_clock["batches"],
    }, indent=2, ensure_ascii=False))

    print(f"\n{len(rows)} rows, {n_pairs} counterfactual pairs, {elapsed/60:.1f} min "
          f"({elapsed/len(rows):.2f} s/generation)")
    print(f"response tokens: median {tokens[len(tokens)//2]}, max {tokens[-1]}")
    print(f"hit the {args.max_new_tokens}-token cap: {len(capped)}/{len(rows)} "
          f"({len(capped)/len(rows)*100:.1f}%) overall")
    for item, stats in cap_by_item.items():
        flag = "  <-- OVER" if stats["token_cap_share"] > CAP_CRITERION else ""
        print(f"    {item:16s} {stats['n_hit_token_cap']:3d}/{stats['n_rows']:3d} "
              f"({stats['token_cap_share']*100:5.1f}%)  median {stats['output_tokens_median']:5d}{flag}")
    if items_over_criterion:
        print(f"\n  CAP CRITERION FAILED for {len(items_over_criterion)} item(s): "
              f"{items_over_criterion}\n  Their readout is partly a truncation measurement. The gate "
              f"script REFUSES this arm until each is either fixed or explicitly excluded "
              f"(--exclude-item), which is recorded in the gate report. Report it; do not raise the "
              f"cap or drop items unasked.")
    else:
        print(f"\n  cap criterion PASSED: every item at or under {CAP_CRITERION*100:.0f}%")
    print(f"written to {args.out_dir}")


if __name__ == "__main__":
    main()
