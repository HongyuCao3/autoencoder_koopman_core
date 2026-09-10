#!/usr/bin/env python3
"""Bank-wide response-length probe for the `constraint` line (2026-09-09).

THE QUESTION: what response cap does S1 need, and which items will bind it?

The per-item cap criterion has now fired twice, and both times the cause was
the constraint text, not the model or the harness: job 15756689 sat at 18.6%
truncation overall with `tuple_294_1` ("Write a longer dialog") at 100%, and
under the fidelity harness at cap 2048 the same item alone hit 21.4% while the
other eleven were at 0.0%. S0-0 ran on 12 items and could absorb naming one of
them. S1 wants N=40 drawn from the whole 200-dialogue bank, where the item set
is no longer small enough to inspect by hand.

A regex over the constraint text was tried first and is not good enough to
size an arm on: it flags both observed long items but also flags
`tuple_245_1`, whose median is 1294 tokens and which never hit the cap --
sensitive, not specific. So this measures instead of guessing.

WHAT IT DOES: zero-control, 3 turns, 1 seed, every dialogue in the bank, under
the fidelity harness (constraints in system + the model's own sampling
defaults), at the cap S1 would use. Output is the per-item token distribution
and the per-item cap share.

WHY 3 TURNS EXTRAPOLATES: response length here is a function of the constraint
set, not of context depth -- job 15756689's truncations were flat across turns
(~4 of 24 at every turn, no drift as the context grew) while two items carried
75 of 87. That is the assumption this probe rests on, and it is the reason a
3-turn probe can size a 20-turn arm. It is an assumption, not a theorem: an
item near the boundary at turn 3 can cross it by turn 20, so the probe sizes
the CAP and flags the items, it does not certify an arm.

NOT EVIDENCE FOR ANY GATE. This is infrastructure: it decides a generation
setting and an item list. No K gate reads it, and no number from it goes in
the paper.
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
    coverage_record,
    gold_constraints,
    load_sequor_bank,
    select_bank_items,
)
from persona_drift.sequor_trajectory import (  # noqa: E402
    CAP_CRITERION,
    BranchArmConfig,
    arm_config_record,
    cap_accounting,
    run_branch_arm,
)

RESOURCES = pathlib.Path("resources/sequor")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--agent-model", required=True)
    p.add_argument("--out-dir", type=pathlib.Path, required=True, help="Must not already exist.")
    p.add_argument("--tuples-path", type=pathlib.Path, default=RESOURCES / "sequor_tuples3.jsonl")
    p.add_argument("--gold-path", type=pathlib.Path, default=RESOURCES / "sequor_gold_judge_calibration.jsonl")
    p.add_argument("--n-items", type=int, default=None,
                   help="Default: the whole bank. Whatever N=40 rule is signed later, the "
                        "measurement is already there.")
    p.add_argument("--n-turns", type=int, default=3,
                   help="Enough to read the length distribution; see the module docstring on why "
                        "this extrapolates to T=20 and where that assumption stops.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=2048,
                   help="The cap S1 would run at. Items over the criterion HERE are the ones that "
                        "would bind it there.")
    p.add_argument("--constraints-in-system", action="store_true",
                   help="The fidelity harness. Length is measured under the harness S1 will use, "
                        "because the system placement plus sampling is what grew tuple_294_1's "
                        "median from 1653 to 1858 tokens.")
    p.add_argument("--decoding", choices=["greedy", "model_default"], default="model_default")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--max-model-len", type=int, default=12288,
                   help="3 turns x (question + a 2048-token response) with headroom. Not the "
                        "49152 an S1 arm needs -- a shorter context is what lets 200 dialogues "
                        "run concurrently.")
    p.add_argument("--mode", choices=["debug", "canonical"], required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.out_dir.exists():
        raise SystemExit(f"refusing to write into existing {args.out_dir}")

    prov = provenance(switches={
        "agent_model": args.agent_model, "n_items": args.n_items, "n_turns": args.n_turns,
        "seed": args.seed, "max_new_tokens": args.max_new_tokens, "mode": args.mode,
        "constraints_in_system": args.constraints_in_system, "decoding": args.decoding,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
    })
    print(f"harness {prov['git_sha'][:12]}"
          + (f" (+{prov['n_dirty_paths']} uncommitted paths)" if prov["git_dirty"] else "")
          + (f"  job {prov['job_id']}" if prov["job_id"] else ""))

    bank = load_sequor_bank(args.tuples_path, n_turns=args.n_turns)
    items = select_bank_items(bank, gold_constraints(args.gold_path), args.n_items)
    coverage = coverage_record(items, gold_constraints(args.gold_path))
    print(f"{len(items)} items x {args.n_turns} turns x 1 seed -> "
          f"{len(items) * args.n_turns} zero-control rows"
          + ("   [DEBUG: mode=debug, not evidence]" if args.mode == "debug" else ""))
    print(f"gold coverage: {coverage['n_items_fully_covered']}/{len(items)} items fully covered, "
          f"{coverage['share_outside_calibration_set']:.1%} of judged constraints outside the "
          f"calibration set")

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

    llm = LLM(model=args.agent_model, seed=args.seed, dtype="bfloat16",
              gpu_memory_utilization=args.gpu_memory_utilization, max_model_len=args.max_model_len)

    config = BranchArmConfig(
        model_id=args.agent_model, n_turns=args.n_turns, seed=args.seed,
        max_new_tokens=args.max_new_tokens, constraints_in_system=args.constraints_in_system,
        **decoding,
    )
    params = SamplingParams(temperature=config.temperature, top_p=config.top_p,
                            top_k=config.top_k, max_tokens=config.max_new_tokens, seed=args.seed)
    clock = {"t0": time.time(), "batches": 0}

    def generate_batch(conversations: list[list[dict]]) -> list[dict]:
        outputs = llm.chat(conversations, params, chat_template_kwargs={"enable_thinking": False})
        clock["batches"] += 1
        print(f"  batch {clock['batches']:3d}  {len(conversations)} seqs  "
              f"{time.time() - clock['t0']:.0f}s", flush=True)
        return [{
            "text": o.outputs[0].text, "finish_reason": o.outputs[0].finish_reason,
            "n_output_tokens": len(o.outputs[0].token_ids), "n_inserted_tokens": None,
        } for o in outputs]

    t0 = time.time()
    rows = run_branch_arm(items, generate_batch, config, run_id=args.out_dir.name, branch=False)
    for row in rows:
        row["inserted_tokens"] = 0
    elapsed = time.time() - t0

    # Rows first, summary second (job 15761810: a NameError in the cheap code
    # after generation must not be able to destroy the expensive artifact).
    args.out_dir.mkdir(parents=True)
    with (args.out_dir / "trajectories.jsonl").open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{len(rows)} rows written to {args.out_dir / 'trajectories.jsonl'}", flush=True)

    caps = cap_accounting(rows, CAP_CRITERION)
    (args.out_dir / "run_config.json").write_text(json.dumps({
        "mode": args.mode, "arm": "bank-wide response-length probe (zero-control)",
        "provenance": prov, "seed": args.seed, "decoding_mode": args.decoding,
        "decoding": decoding, "constraints_in_system": args.constraints_in_system,
        "gold_coverage": coverage,
        **arm_config_record(config, items),
    }, indent=2, ensure_ascii=False))
    (args.out_dir / "probe_report.json").write_text(json.dumps({
        "mode": args.mode, "n_rows": len(rows), "n_items": len(items),
        "n_turns": args.n_turns, "seed": args.seed,
        "decoding_mode": args.decoding, "decoding": decoding,
        "constraints_in_system": args.constraints_in_system,
        "max_new_tokens": args.max_new_tokens,
        "gold_coverage": coverage,
        **caps,
        "elapsed_s": elapsed, "seconds_per_generation": elapsed / len(rows),
        "batches": clock["batches"],
        "not_evidence": "Infrastructure probe: sizes a cap and flags items. No gate reads it.",
    }, indent=2, ensure_ascii=False))

    over = caps["items_over_cap_criterion"]
    print(f"\n{len(rows)} rows, {elapsed/60:.1f} min ({elapsed/len(rows):.2f} s/generation)")
    print(f"response tokens: median {caps['output_tokens_median']}, max {caps['output_tokens_max']}")
    print(f"hit the {args.max_new_tokens}-token cap: {caps['n_hit_token_cap']}/{len(rows)} "
          f"({caps['token_cap_share']*100:.1f}%) overall")
    print(f"\nitems over the {CAP_CRITERION*100:.0f}% per-item criterion: {len(over)}/{len(items)}")
    for item in over:
        stats = caps["token_cap_by_item"][item]
        print(f"    {item:16s} {stats['n_hit_token_cap']:2d}/{stats['n_rows']:2d} "
              f"({stats['token_cap_share']*100:5.1f}%)  median {stats['output_tokens_median']:5d}")
        for c in stats["constraints"]:
            print(f"        - {c[:100]}")
    ranked = sorted(caps["token_cap_by_item"].items(),
                    key=lambda kv: -kv[1]["output_tokens_median"])[:10]
    print("\nlongest 10 items by median response tokens:")
    for item, stats in ranked:
        print(f"    {item:16s} median {stats['output_tokens_median']:5d}  "
              f"cap {stats['token_cap_share']*100:5.1f}%")
    print(f"\nwritten to {args.out_dir}")
    print("This probe sizes a cap and names items. It decides no gate and reports no result.")


if __name__ == "__main__":
    main()
