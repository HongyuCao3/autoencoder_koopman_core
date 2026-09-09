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
    p.add_argument("--seed", type=int, default=0)
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
        "seed": args.seed, "max_new_tokens": args.max_new_tokens, "mode": args.mode,
        "system_prompt": args.system_prompt, "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
    })
    print(f"harness {prov['git_sha'][:12]}"
          + (f" (+{prov['n_dirty_paths']} uncommitted paths)" if prov["git_dirty"] else "")
          + (f"  job {prov['job_id']}" if prov["job_id"] else ""))

    items = select_screening_items(
        load_sequor_bank(args.tuples_path, n_turns=args.n_turns),
        gold_constraints(args.gold_path),
        args.n_items,
    )
    config = BranchArmConfig(
        model_id=args.agent_model, n_turns=args.n_turns, seed=args.seed,
        max_new_tokens=args.max_new_tokens, temperature=0.0, system_prompt=args.system_prompt,
    )
    print(f"{len(items)} items x {args.n_turns} turns -> "
          f"{len(items) * args.n_turns} base rows + {len(items) * (args.n_turns - 1)} counterfactual rows"
          + ("   [DEBUG: mode=debug, not evidence]" if args.mode == "debug" else ""))

    from vllm import LLM, SamplingParams

    llm = LLM(model=args.agent_model, seed=args.seed, dtype="bfloat16",
              gpu_memory_utilization=args.gpu_memory_utilization, max_model_len=args.max_model_len)
    params = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens, seed=args.seed)
    tokenizer = llm.get_tokenizer()
    turn_clock = {"t0": time.time(), "batches": 0}

    def generate_batch(conversations: list[list[dict]]) -> list[dict]:
        outputs = llm.chat(conversations, params, chat_template_kwargs={"enable_thinking": False})
        turn_clock["batches"] += 1
        print(f"  batch {turn_clock['batches']:3d}  {len(conversations)} seqs  "
              f"{time.time() - turn_clock['t0']:.0f}s elapsed", flush=True)
        return [{
            "text": o.outputs[0].text,
            "finish_reason": o.outputs[0].finish_reason,
            "n_output_tokens": len(o.outputs[0].token_ids),
            "n_inserted_tokens": None,
        } for o in outputs]

    t0 = time.time()
    rows = run_branch_arm(items, generate_batch, config, run_id=args.out_dir.name)
    elapsed = time.time() - t0

    # inserted cost, from the agent's own tokenizer (reminded rows only)
    block_tokens = {
        item.conversation_id: len(tokenizer.encode(f"\n\n{item.constraint_block}")) for item in items
    }
    for row in rows:
        row["inserted_tokens"] = block_tokens[row["item_id"]] if row["u_remind"] else 0

    n_pairs = assert_pairs_share_prefix(rows)
    capped = [r for r in rows if r["hit_token_cap"]]
    tokens = sorted(r["n_output_tokens"] for r in rows)

    args.out_dir.mkdir(parents=True)
    with (args.out_dir / "trajectories.jsonl").open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.out_dir / "run_config.json").write_text(json.dumps({
        "mode": args.mode, "arm": "S0-0 counterfactual branch", "provenance": prov,
        **arm_config_record(config, items),
    }, indent=2, ensure_ascii=False))
    (args.out_dir / "arm_report.json").write_text(json.dumps({
        "mode": args.mode, "n_rows": len(rows), "n_pairs": n_pairs,
        "prefix_check": "every pair shares a byte-identical prefix (assert_pairs_share_prefix)",
        "n_hit_token_cap": len(capped), "token_cap_share": len(capped) / len(rows),
        "output_tokens_median": tokens[len(tokens) // 2], "output_tokens_max": tokens[-1],
        "elapsed_s": elapsed, "seconds_per_generation": elapsed / len(rows),
        "batches": turn_clock["batches"],
    }, indent=2))

    print(f"\n{len(rows)} rows, {n_pairs} counterfactual pairs, {elapsed/60:.1f} min "
          f"({elapsed/len(rows):.2f} s/generation)")
    print(f"response tokens: median {tokens[len(tokens)//2]}, max {tokens[-1]}")
    print(f"hit the {args.max_new_tokens}-token cap: {len(capped)}/{len(rows)} "
          f"({len(capped)/len(rows)*100:.1f}%)"
          + ("   <-- a readout partly measuring truncation; raise the cap before canonical"
             if len(capped) / len(rows) > 0.05 else ""))
    print(f"written to {args.out_dir}")


if __name__ == "__main__":
    main()
