#!/usr/bin/env python3
"""Upstream-fidelity arm for the `constraint` line (2026-09-09).

THE QUESTION: our harness holds the model 1.7x above the benchmark it came
from -- upstream-comparable retention (all k=3 constraints held) is 0.833 at
turn 1 against upstream's ~0.50 for this same model id, and falls only to
0.583 by turn 20 where upstream reaches <0.12 by turn 50. That gap is why 56%
of the S0-0 arm's pairs had no headroom for a reminder to recover (delta_y
+0.0026 at y_prev=1.00, against +0.125 to +0.20 wherever headroom existed),
and therefore why K1 (readout range) failed.

Is the gap caused by the two harness knobs we chose ourselves?

  placement:  the constraint block in USER TURN 1 (what S0-0 did -- inferred
              from the vendored field, never checked against upstream's own
              conversation builder) vs in a SYSTEM message.
  decoding:   GREEDY (what S0-0 did) vs the model's OWN DEFAULT, which is what
              upstream reported using. The defaults are read from the model's
              generation_config at runtime rather than hardcoded, so "default"
              means the model's default and the report records the values.

Zero-control only: no counterfactual branching, because the question here is
the shape of the retention curve under each harness, not the gain from an
action. Four variants share ONE model load in one job -- which is also what
keeps this inside the instrument quota (docs/LEDGER.md: 5 of the last 10 jobs
are already instruments, and two more would trip the stop rule).

PRE-REGISTERED before this runs, in the ledger and screening doc section 10:
a variant "closes the gap" if its mean upstream-comparable retention over
t1..t20 is at least 0.10 BELOW the greedy/turn-1 baseline with an item-paired
bootstrap CI excluding 0, AND its turn-1 value is <= 0.60. Pass -> re-run the
S0-0 arm under that variant and recompute K1/K2/K3. No variant passes -> the
readout has no range on this task with this model, and closing the line
stands. Verdicts are computed by scripts/analyze_sequor_fidelity.py, not here.
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
    run_branch_arm,
)

RESOURCES = pathlib.Path("resources/sequor")

# (variant name, constraints_in_system, use the model's own sampling defaults)
VARIANTS = [
    ("turn1_greedy", False, False),   # exactly what the S0-0 arm did -- the baseline
    ("turn1_default", False, True),
    ("system_greedy", True, False),
    ("system_default", True, True),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--agent-model", required=True)
    p.add_argument("--out-dir", type=pathlib.Path, required=True, help="Must not already exist.")
    p.add_argument("--tuples-path", type=pathlib.Path, default=RESOURCES / "sequor_tuples3.jsonl")
    p.add_argument("--gold-path", type=pathlib.Path, default=RESOURCES / "sequor_gold_judge_calibration.jsonl")
    p.add_argument("--n-items", type=int, default=12,
                   help="Same 12 items as the S0-0 arm, so every variant is paired with it "
                        "item-for-item and only the harness differs.")
    p.add_argument("--n-turns", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=2048,
                   help="2048 as the S0-0 arm settled on: 0/468 truncated there, and a cap that "
                        "binds would confound a retention curve with our own token budget.")
    p.add_argument("--max-model-len", type=int, default=49152)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--mode", choices=["debug", "canonical"], required=True)
    p.add_argument("--variant", action="append", default=None, choices=[v[0] for v in VARIANTS],
                   help="Repeatable; default is all four.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.out_dir.exists():
        raise SystemExit(f"refusing to write into existing {args.out_dir}")
    wanted = args.variant or [v[0] for v in VARIANTS]
    variants = [v for v in VARIANTS if v[0] in wanted]

    prov = provenance(switches={
        "agent_model": args.agent_model, "n_items": args.n_items, "n_turns": args.n_turns,
        "seed": args.seed, "max_new_tokens": args.max_new_tokens, "mode": args.mode,
        "variants": [v[0] for v in variants], "max_model_len": args.max_model_len,
    })
    print(f"harness {prov['git_sha'][:12]}"
          + (f" (+{prov['n_dirty_paths']} uncommitted paths)" if prov["git_dirty"] else "")
          + (f"  job {prov['job_id']}" if prov["job_id"] else ""))

    items = select_screening_items(
        load_sequor_bank(args.tuples_path, n_turns=args.n_turns),
        gold_constraints(args.gold_path),
        args.n_items,
    )
    print(f"{len(items)} items x {args.n_turns} turns x {len(variants)} variants = "
          f"{len(items) * args.n_turns * len(variants)} generations"
          + ("   [DEBUG: mode=debug, not evidence]" if args.mode == "debug" else ""))

    from transformers import GenerationConfig
    from vllm import LLM, SamplingParams

    # "default" means the model's own default, read from its config, not a
    # number typed here: upstream reported using defaults and did not print them.
    model_defaults = GenerationConfig.from_pretrained(args.agent_model)
    defaults = {
        "temperature": float(model_defaults.temperature), "top_p": float(model_defaults.top_p),
        "top_k": int(model_defaults.top_k), "do_sample": bool(model_defaults.do_sample),
    }
    print(f"model's own generation defaults: {defaults}")
    if not defaults["do_sample"]:
        raise SystemExit(
            f"{args.agent_model} reports do_sample=false, so 'greedy' and 'default' are the same "
            f"harness and this arm cannot separate them. Check the model id."
        )

    llm = LLM(model=args.agent_model, seed=args.seed, dtype="bfloat16",
              gpu_memory_utilization=args.gpu_memory_utilization, max_model_len=args.max_model_len)

    all_rows: list[dict] = []
    variant_records: dict[str, dict] = {}
    for name, in_system, use_defaults in variants:
        config = BranchArmConfig(
            model_id=args.agent_model, n_turns=args.n_turns, seed=args.seed,
            max_new_tokens=args.max_new_tokens,
            temperature=defaults["temperature"] if use_defaults else 0.0,
            top_p=defaults["top_p"] if use_defaults else 1.0,
            top_k=defaults["top_k"] if use_defaults else -1,
            constraints_in_system=in_system, variant=name,
        )
        params = SamplingParams(
            temperature=config.temperature, top_p=config.top_p, top_k=config.top_k,
            max_tokens=config.max_new_tokens, seed=args.seed,
        )
        clock = {"t0": time.time(), "batches": 0}

        def generate_batch(conversations: list[list[dict]], _params=params, _clock=clock) -> list[dict]:
            outputs = llm.chat(conversations, _params, chat_template_kwargs={"enable_thinking": False})
            _clock["batches"] += 1
            return [{
                "text": o.outputs[0].text, "finish_reason": o.outputs[0].finish_reason,
                "n_output_tokens": len(o.outputs[0].token_ids), "n_inserted_tokens": 0,
            } for o in outputs]

        t0 = time.time()
        rows = run_branch_arm(items, generate_batch, config, run_id=f"{args.out_dir.name}__{name}",
                              branch=False)
        elapsed = time.time() - t0
        for row in rows:
            row["inserted_tokens"] = 0
        all_rows.extend(rows)

        capped = sum(1 for r in rows if r["hit_token_cap"])
        tokens = sorted(r["n_output_tokens"] for r in rows)
        variant_records[name] = {
            **arm_config_record(config, items),
            "constraints_in_system": in_system, "uses_model_defaults": use_defaults,
            "sampling": {"temperature": config.temperature, "top_p": config.top_p, "top_k": config.top_k},
            "n_rows": len(rows), "n_hit_token_cap": capped, "token_cap_share": capped / len(rows),
            "output_tokens_median": tokens[len(tokens) // 2], "output_tokens_max": tokens[-1],
            "elapsed_s": elapsed, "seconds_per_generation": elapsed / len(rows),
        }
        print(f"  {name:16s} {len(rows):4d} rows  {elapsed/60:5.1f} min  "
              f"({elapsed/len(rows):.2f} s/gen)  median {tokens[len(tokens)//2]:5d} tokens  "
              f"truncated {capped}/{len(rows)}", flush=True)

    args.out_dir.mkdir(parents=True)
    with (args.out_dir / "trajectories.jsonl").open("w") as fh:
        for row in all_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.out_dir / "run_config.json").write_text(json.dumps({
        "mode": args.mode, "arm": "upstream-fidelity 2x2 (placement x decoding), zero-control",
        "model_id": args.agent_model, "provenance": prov,
        "model_generation_defaults": defaults,
        "baseline_variant": "turn1_greedy",
        "variants": variant_records,
        "item_ids": [i.conversation_id for i in items],
    }, indent=2, ensure_ascii=False))
    print(f"\n{len(all_rows)} rows over {len(variants)} variants -> {args.out_dir}")
    print("next: score with the reporting judge, then scripts/analyze_sequor_fidelity.py")


if __name__ == "__main__":
    main()
