#!/usr/bin/env python3
"""Targeted-reminder pilot for the `constraint` line: does restating ONLY the
constraint that broke beat restating all three?

This is the one empirical question left after options (a) and (b) both came
back null on paper (docs/experiments/constraint_results.md). Both nulls have
the same mechanism: with a single on/off action, the state can only scale what
a reminder is worth, never move the best turn to spend it on. A targeted
reminder cannot be scheduled in advance -- which rule is broken at turn 12 is
not knowable at turn 1 -- so this is the cheapest action change that could give
the closed loop something a fixed plan cannot imitate.

WHAT IT COSTS AND WHY IT IS CHEAP. No trajectory is regenerated. The branches
hang off the stored `zero_control` rows of an existing arm, whose turn-t
response already IS the no-reminder counterfactual from the identical prefix.
Each branch point costs two generations, and every rebuilt prefix is hashed
against the digest the stored row recorded (`prefix_sha256`), so "same prefix"
is checked, not assumed.

PRE-REGISTERED, before any GPU (screening section 10 item 13):

    PRIMARY   recovery rate of a constraint that was broken at t-1: the share
              judged satisfied at t, targeted minus blanket, paired per
              (item, seed, turn, constraint) and bootstrapped over ITEMS.
    MDE       0.0525 at 40 items, from the item-level sd 0.1185 measured on the
              S1b arms' own u=1-vs-u=0 recovery contrast. The blanket reminder
              buys +0.0589 on a broken constraint, so this design resolves a
              targeted reminder that roughly DOUBLES the actuator's effect and
              nothing smaller. Anything below the MDE is UNDECIDABLE, never
              "targeting does not work".
    SECONDARY (must be reported with the primary, not instead of it) retention
              of the constraints that were ALREADY KEPT at t-1. A targeted
              reminder mentions them nowhere, and a model that reads the
              omission as permission would recover the broken rule while
              dropping the others -- which is why the net `y` contrast is
              reported beside the per-constraint one.
    COST      `inserted_tokens` per branch: the targeted block is shorter, so
              an equal-token comparison is available if the recovery rates tie.
    CONSEQUENCE  above the MDE -> the closed loop's value is in WHICH rule to
              restate, and plan section 6's arm table is rewritten around a
              targeted actuator. Below it -> the action dimension is degenerate
              too, and `koopman_mpc` is dropped rather than re-specified a
              fourth time.

ONE ASYMMETRY: the two fresh branches share a sampling seed with each other,
but the stored u=0 row was drawn at a different time under sampled decoding and
is NOT RNG-paired with them. Hence the primary is targeted-minus-blanket, both
fresh and both under common random numbers; the u=0 comparison is secondary.

The readout is applied afterwards by scripts/score_sequor_trajectories.py, as
everywhere on this line: the same responses are scored by the independent
reporting judge, and a judge swap never means regenerating anything.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from persona_drift.run_provenance import provenance  # noqa: E402
from persona_drift.sequor_bank import load_sequor_bank  # noqa: E402
from persona_drift.sequor_targeted_branch import (  # noqa: E402
    BLANKET,
    TARGETED,
    base_trajectories,
    branch_points,
    run_targeted_branches,
    targeting_agreement,
)
from persona_drift.sequor_trajectory import (  # noqa: E402
    CAP_CRITERION,
    BranchArmConfig,
    arm_config_record,
    cap_accounting,
)

RESOURCES = pathlib.Path("resources/sequor")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-arm-dir", type=pathlib.Path, required=True,
                   help="Arm whose zero_control rows the branches hang off.")
    p.add_argument("--target-readout", type=pathlib.Path, required=True,
                   help="Readout whose verdict at t-1 picks the target. Pre-registered as the "
                        "INDEPENDENT one: the question is whether a targeted reminder is more "
                        "effective, not whether a 4B can tell what broke. What the cheaper in-loop "
                        "judge would have targeted is reported beside it, at zero GPU.")
    p.add_argument("--agreement-readout", type=pathlib.Path, default=None,
                   help="The other readout, for that comparison. Optional.")
    p.add_argument("--out-dir", type=pathlib.Path, required=True, help="Must not already exist.")
    p.add_argument("--tuples-path", type=pathlib.Path, default=RESOURCES / "sequor_tuples3.jsonl")
    p.add_argument("--agent-model", required=True)
    p.add_argument("--max-new-tokens", type=int, default=2048,
                   help="Must match the base arm's cap or the branches are not comparable to the "
                        "stored u=0 row; the runner refuses a mismatch.")
    p.add_argument("--max-model-len", type=int, default=49152)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--batch-size", type=int, default=128,
                   help="Generations per llm.chat call. Throughput is set by how many sequences "
                        "the KV cache holds at this context length (~9 at 49k on an 80GB card), "
                        "not by batch width, so this only buys progress lines and bounds how much "
                        "a single failed call costs.")
    p.add_argument("--mode", choices=["debug", "canonical"], required=True)
    p.add_argument("--limit-points", type=int, default=None,
                   help="Debug only: branch on the first N points. Refused in canonical mode.")
    p.add_argument("--dry-run", action="store_true",
                   help="Build every prompt, check every prefix digest, print the accounting -- "
                        "and load no model. This is the pre-submission check, not a smoke test.")
    return p.parse_args()


def load_readout(path: pathlib.Path) -> tuple[dict, dict]:
    report = json.loads(path.read_text())
    verdicts = {(row["trajectory_id"], row["turn"]): row.get("followed") for row in report["rows"]}
    return report, verdicts


def main() -> None:
    args = parse_args()
    if args.out_dir.exists():
        raise SystemExit(f"refusing to write into existing {args.out_dir}")
    if args.mode == "canonical" and args.limit_points is not None:
        raise SystemExit("--limit-points is a debug switch; a canonical arm branches on every point")

    base_rows = [json.loads(line) for line in (args.base_arm_dir / "trajectories.jsonl").open()]
    base_config = json.loads((args.base_arm_dir / "run_config.json").read_text())
    target_report, target_verdicts = load_readout(args.target_readout)
    if target_report["judge_kind"] != "independent":
        raise SystemExit(f"--target-readout is judge_kind={target_report['judge_kind']!r}; the "
                         f"pre-registered target signal is the independent readout")

    base = base_trajectories(base_rows)
    sample = base_rows[0]
    decoding = sample["decoding_config"]
    if decoding["max_new_tokens"] != args.max_new_tokens:
        raise SystemExit(f"base arm generated at cap {decoding['max_new_tokens']}, this run asks for "
                         f"{args.max_new_tokens}: the branches would not be comparable to the stored u=0 row")
    config = BranchArmConfig(
        model_id=args.agent_model, n_turns=sample["n_turns"], seed=sample["seed"],
        max_new_tokens=args.max_new_tokens, temperature=decoding["temperature"],
        top_p=decoding["top_p"], top_k=decoding["top_k"],
        system_prompt=sample["system_prompt"], constraints_in_system=sample["constraints_in_system"],
    )
    if sample["model"] != args.agent_model:
        raise SystemExit(f"base arm was generated by {sample['model']!r}, this run would use "
                         f"{args.agent_model!r}: the branch and its u=0 row must come from one model")

    items = {item.conversation_id: item for item in
             load_sequor_bank(args.tuples_path, n_turns=sample["n_turns"])}
    points = branch_points(base, target_verdicts, sample["n_turns"])
    if args.limit_points:
        points = points[:args.limit_points]

    prov = provenance(switches={
        "base_arm_dir": str(args.base_arm_dir), "target_readout": str(args.target_readout),
        "agent_model": args.agent_model, "max_new_tokens": args.max_new_tokens,
        "mode": args.mode, "limit_points": args.limit_points, "dry_run": args.dry_run,
    })
    print(f"harness {prov['git_sha'][:12]}"
          + (f" (+{prov['n_dirty_paths']} uncommitted paths)" if prov["git_dirty"] else "")
          + (f"  job {prov['job_id']}" if prov["job_id"] else ""))

    trajectories = len(base)
    turns_available = sum(len(v) - 1 for v in base.values())
    by_n_violated: dict[int, int] = {}
    for point in points:
        by_n_violated[point["n_violated_at_t_minus_1"]] = by_n_violated.get(
            point["n_violated_at_t_minus_1"], 0) + 1
    print(f"{trajectories} base trajectories ({len({i for i, _ in base}) } items x "
          f"{len({s for _, s in base})} seeds), {turns_available} branchable turns")
    print(f"{len(points)} branch points ({len(points)/turns_available:.1%} of them) -> "
          f"{2*len(points)} generations")
    print(f"  broken constraints at t-1: " + ", ".join(
        f"{k} -> {v}" for k, v in sorted(by_n_violated.items())))
    if args.agreement_readout:
        _, other = load_readout(args.agreement_readout)
        agreement = targeting_agreement(target_verdicts, other, points)
        print(f"  a controller on {args.agreement_readout.name} would target identically on "
              f"{agreement['agreement_share']:.1%} of them, differently on "
              f"{agreement['different_target']}, and would not act on "
              f"{agreement['would_have_skipped_unparsed']} unparsed + "
              f"{agreement['would_have_seen_nothing_broken']} it thinks are clean")
    else:
        agreement = None

    if args.dry_run:
        calls = {"n": 0}

        def generate_batch(conversations: list[list[dict]], seeds: list[int]) -> list[dict]:
            calls["n"] += 1
            assert len(seeds) == len(conversations)
            return [{"text": "", "finish_reason": "stop", "n_output_tokens": 0,
                     "n_inserted_tokens": None} for _ in conversations]

        rows = run_targeted_branches(items, base, points, generate_batch, config, "dry-run")
        print(f"\nDRY RUN: {len(rows)} rows would be written in {calls['n']} batched calls; every "
              f"prefix digest matched the stored row and every user message matched the bank")
        for row in rows[:2]:
            print(f"\n--- {row['branch']} {row['item_id']} turn {row['turn']} "
                  f"(broken: {row['violated_indices']}) ---\n{row['user_message'][-400:]}")
        return

    from vllm import LLM, SamplingParams

    llm = LLM(model=args.agent_model, seed=config.seed, dtype="bfloat16",
              gpu_memory_utilization=args.gpu_memory_utilization, max_model_len=args.max_model_len)
    tokenizer = llm.get_tokenizer()
    def sampling_params(seed: int) -> "SamplingParams":
        return SamplingParams(temperature=config.temperature, top_p=config.top_p,
                              top_k=config.top_k, max_tokens=config.max_new_tokens, seed=seed)

    clock = {"t0": time.time(), "batches": 0}

    def generate_batch(conversations: list[list[dict]], seeds: list[int], _c=clock) -> list[dict]:
        generated = []
        for start in range(0, len(conversations), args.batch_size):
            chunk = conversations[start:start + args.batch_size]
            per_request = [sampling_params(s) for s in seeds[start:start + args.batch_size]]
            outputs = llm.chat(chunk, per_request, chat_template_kwargs={"enable_thinking": False})
            _c["batches"] += 1
            done = start + len(chunk)
            print(f"  batch {_c['batches']:3d}  {done}/{len(conversations)} seqs  "
                  f"{time.time()-_c['t0']:.0f}s", flush=True)
            generated.extend({"text": o.outputs[0].text, "finish_reason": o.outputs[0].finish_reason,
                              "n_output_tokens": len(o.outputs[0].token_ids),
                              "n_inserted_tokens": None} for o in outputs)
        return generated

    t0 = time.time()
    rows = run_targeted_branches(items, base, points, generate_batch, config, args.out_dir.name)
    elapsed = time.time() - t0
    for row in rows:
        block = row["user_message"].split("\n\n")[-1]
        row["inserted_tokens"] = len(tokenizer.encode(f"\n\n{block}"))

    # Rows first, summary second: job 15761810 lost 52 GPU-minutes of generation
    # to a NameError in the report built after it.
    args.out_dir.mkdir(parents=True)
    with (args.out_dir / "trajectories.jsonl").open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{len(rows)} rows written to {args.out_dir / 'trajectories.jsonl'}", flush=True)

    caps = cap_accounting(rows, CAP_CRITERION)
    tokens = {branch: sorted(r["inserted_tokens"] for r in rows if r["branch"] == branch)
              for branch in (BLANKET, TARGETED)}
    (args.out_dir / "run_config.json").write_text(json.dumps({
        "mode": args.mode, "arm": "targeted-reminder counterfactual branches", "provenance": prov,
        "base_arm_dir": str(args.base_arm_dir), "target_readout": str(args.target_readout),
        "base_arm_provenance": base_config.get("provenance"),
        "decoding": decoding, "seeds": sorted({s for _, s in base}),
        **arm_config_record(config, list(items.values())),
    }, indent=2, ensure_ascii=False))
    (args.out_dir / "arm_report.json").write_text(json.dumps({
        "mode": args.mode, "n_rows": len(rows), "n_branch_points": len(points),
        "n_turns": config.n_turns, "seeds": sorted({s for _, s in base}),
        "n_items": len({i for i, _ in base}),
        "branch_points_by_n_violated": by_n_violated,
        "prefix_check": "every rebuilt prefix was hashed against the stored row's prefix_sha256",
        "targeting_agreement": agreement,
        "inserted_tokens_median": {b: (v[len(v)//2] if v else None) for b, v in tokens.items()},
        **caps,
        "elapsed_s": elapsed, "seconds_per_generation": elapsed / max(len(rows), 1),
        "batches": clock["batches"],
    }, indent=2, ensure_ascii=False))
    print(f"\n{len(rows)} rows, {len(points)} branch points, {elapsed/60:.1f} min "
          f"({elapsed/max(len(rows),1):.2f} s/generation)")
    print(f"inserted tokens (median): blanket {tokens[BLANKET][len(tokens[BLANKET])//2]}, "
          f"targeted {tokens[TARGETED][len(tokens[TARGETED])//2]}")
    print(f"hit the {args.max_new_tokens}-token cap: {caps['n_hit_token_cap']}/{len(rows)} "
          f"({caps['token_cap_share']*100:.1f}%)")
    print(f"written to {args.out_dir}")


if __name__ == "__main__":
    main()
