#!/usr/bin/env python3
"""GPU-2a for the `tsar_cefr` line: the counterfactual BRANCH TREE that makes
D-2.5 answerable. Plan section 5 / kill_criterion.md section 2 (D-2.5).

WHY THIS ARM EXISTS. D-2.5 was pre-registered to be computed on GPU-1's rows
with no GPU at all. That assumption does not survive contact with GPU-1's
design, for a reason specific to how the two lines excite:

  `defense`  every `fixed_t{k}` arm is byte-identical to `zero_control` for
             turns 1..k-1, so ANY policy could be replayed off disk exactly.
  `tsar_cefr` GPU-1 draws six actions independently per trajectory. Each item
             got 3 of 4^6 = 4096 paths; 0 of 199 items saw all four actions
             from the identical step-1 state; the IPW effective sample for one
             6-step policy is 597/4096 = 0.146 trajectories.

The estimator was never the problem -- uniform random actions make IPW
unbiased. The POWER is the problem, and power is bought with counterfactuals,
not with N: doubling the trajectories still leaves 0.29 of one trajectory per
candidate policy. So this arm buys the counterfactuals directly. From every
item's source it enumerates EVERY action sequence of length 4, which makes the
horizon-4 bound exact: no operator is fitted, no importance weight is taken,
no policy is extrapolated to a state that was never visited.

    depth 1     4 nodes     each item's source, each of the four actions
    depth 2    16 nodes
    depth 3    64 nodes
    depth 4   256 nodes
              ---------
              340 nodes per item  x 199 items = 67,660 generations

HORIZON 4 IS THE PLAN'S OWN HORIZON, not a budget compromise: the MPC in plan
section 5.4 optimises over H=4. Depth 6 would be 4^6 per item -- 1.1M
generations -- and is not on the table; depth 3 (16,716 generations) is a
strictly weaker bound, so a depth-3 zero would already close the line.

WHAT IS DELIBERATELY NOT SEEDED. Decoding is greedy, as in GPU-1. GPU-1's
three seeds drew ACTION SEQUENCES, and this arm enumerates the action space
exhaustively -- that axis is saturated, not dropped. The remaining uncertainty
is over which source paragraphs the bank happens to contain, and it is carried
by the bootstrap over `source_id`. THE REPORTING CONSEQUENCE IS A USER
DECISION, NOT THIS SCRIPT'S: .claude/global.md asks for `mean +/- std (n)`
over >= 3 seeds, and a deterministic enumeration has no seed axis to average.

THE ITEM `51-a2` IS EXCLUDED, carrying the 2026-09-14 ruling forward so this
arm and the gate that admitted GPU-1 stand on the same 199 items.

CAPS ARE GPU-1'S CAPS: max_new_tokens = ceil(1.5 * source tokens), fixed for
the whole tree, taken from the SOURCE and not from the parent node -- the same
rule GPU-1 ran under, so the two arms' rows are comparable. Truncation is
counted per item and reported; it is not silently repaired.

NOT SUBMITTED WITHOUT A USER RULING (.claude/experiments.md item 1), and the
D-2.5 criteria must be signed BEFORE this runs -- an upper bound whose policy
class is chosen after the tree is on disk is not an upper bound.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys
import time

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "persona_drift_control" / "src"))

from persona_drift import tsar_cefr_actions as actions          # noqa: E402
from persona_drift import tsar_cefr_bank as bank                # noqa: E402
from persona_drift import tsar_cefr_readout as readout          # noqa: E402
from persona_drift.run_provenance import provenance             # noqa: E402

SIGNED_TEMPERATURE = 6.5            # G-T1's readout temperature; never re-fitted here
CAP_LENGTH_MULTIPLIER = 1.5         # GPU-1's rule, unchanged
MIN_MAX_NEW_TOKENS = 128
EXCLUDED_ITEMS = ("51-a2",)         # 2026-09-14 user ruling, carried forward
SIGNED_DEPTH = 4                    # plan section 5.4's MPC horizon


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--agent-model", required=True)
    p.add_argument("--out-dir", type=pathlib.Path, required=True, help="Must not already exist.")
    p.add_argument("--split", default="test", choices=sorted(bank.SPLITS))
    p.add_argument("--depth", type=int, default=SIGNED_DEPTH,
                   help="tree depth; 4 is the signed value (the MPC horizon). A SMALLER "
                        "depth yields a strictly weaker bound, which is sound; a larger "
                        "one is refused because the cost is 4x per level.")
    p.add_argument("--n-sources", type=int, default=None,
                   help="debug only: truncate the bank to the first N source paragraphs")
    p.add_argument("--temperature", type=float, default=SIGNED_TEMPERATURE,
                   help="control-readout temperature; G-T1 signed 6.5")
    p.add_argument("--readout-device", type=int, default=0)
    p.add_argument("--readout-batch-size", type=int, default=32)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--chunk", type=int, default=8192,
                   help="prompts per llm.chat call; progress and host memory only, "
                        "it does not change a single generated token")
    p.add_argument("--dry-run", action="store_true",
                   help="build and census the tree, load no model, generate nothing")
    p.add_argument("--mode", choices=["debug", "canonical"], required=True)
    return p.parse_args()


def token_cap(n_source_tokens: int) -> int:
    return max(MIN_MAX_NEW_TOKENS, int(-(-n_source_tokens * CAP_LENGTH_MULTIPLIER // 1)))


def tree_census(n_items: int, depth: int) -> dict:
    per_item = sum(len(actions.ACTION_NAMES) ** d for d in range(1, depth + 1))
    return {"n_items": n_items, "depth": depth,
            "nodes_per_item": per_item,
            "nodes_by_depth": {d: n_items * len(actions.ACTION_NAMES) ** d
                               for d in range(1, depth + 1)},
            "n_generations": n_items * per_item,
            "n_readout_rows": n_items * per_item + n_items}


def select_items(items, n_sources, mode):
    if n_sources is not None:
        if mode == "canonical":
            raise SystemExit("--n-sources truncates the bank; it is a debug switch")
        keep = set(bank.source_ids(items)[:n_sources])
        items = [item for item in items if item.source_id in keep]
    kept = [item for item in items if item.text_id not in set(EXCLUDED_ITEMS)]
    dropped = sorted({item.text_id for item in items} & set(EXCLUDED_ITEMS))
    if mode == "canonical" and set(dropped) != set(EXCLUDED_ITEMS):
        raise SystemExit(
            f"expected to exclude {EXCLUDED_ITEMS} but the split contains {dropped}: "
            "this arm must stand on the same 199 items as the gate that admitted GPU-1")
    return kept, dropped


def build_roots(items, cap_of) -> list[dict]:
    """Depth-0 nodes: the source paragraphs themselves.

    Split out of main() and exercised by --dry-run and by a unit test, because
    the first submission (15896362) died here after 2m21s of GPU time on
    `item.source_text` -- TsarItem's field is `original`. A dry run that
    returns before it touches the bank cannot catch an attribute error, which
    makes it a check that passes exactly when it is not needed.
    """
    return [{"text_id": item.text_id, "source_id": item.source_id,
             "target_cefr": item.target_cefr, "depth": 0, "path": (),
             "node_id": f"{item.text_id}|", "parent_id": None,
             "text_out": item.original,
             "cap": cap_of(item.original)}
            for item in items]


def expand(llm, sampling_cls, frontier: list[dict], chunk: int) -> list[dict]:
    """One depth of the tree: every frontier node x every action, in order.

    The node id is the action path (`07-a2|step_down.copy`), so a row names the
    counterfactual it belongs to without a join, and the parent is a prefix of
    the child by construction rather than by a stored pointer that could rot.
    """
    jobs = []
    for node in frontier:
        for action in actions.ACTION_NAMES:
            jobs.append({
                "text_id": node["text_id"], "source_id": node["source_id"],
                "target_cefr": node["target_cefr"],
                "depth": node["depth"] + 1,
                "path": node["path"] + (action,),
                "node_id": f"{node['text_id']}|{'.'.join(node['path'] + (action,))}",
                "parent_id": node["node_id"], "action": action,
                "text_in": node["text_out"], "cap": node["cap"],
            })
    rows = []
    for start in range(0, len(jobs), chunk):
        part = jobs[start:start + chunk]
        conversations = [
            [{"role": "system", "content": actions.SYSTEM_PROMPT},
             {"role": "user", "content": actions.user_message(
                 job["text_in"], job["action"], job["target_cefr"])}]
            for job in part
        ]
        params = [sampling_cls(temperature=0.0, max_tokens=job["cap"], seed=0) for job in part]
        started = time.time()
        outputs = llm.chat(conversations, params, chat_template_kwargs={"enable_thinking": False})
        for job, output in zip(part, outputs):
            completion = output.outputs[0]
            text_out = completion.text.strip()
            rows.append({**job, "text_out": text_out,
                         "n_tokens_out": len(completion.token_ids),
                         "hit_token_cap": completion.finish_reason == "length"})
        print(f"  depth {part[0]['depth']}: {start + len(part)}/{len(jobs)} "
              f"in {(time.time() - started) / 60:.1f} min", flush=True)
    return rows


def cap_pressure_by_item(rows: list[dict]) -> dict:
    by_item: dict[str, list[bool]] = collections.defaultdict(list)
    for row in rows:
        by_item[row["text_id"]].append(bool(row["hit_token_cap"]))
    shares = {t: sum(v) / len(v) for t, v in by_item.items()}
    return {"pooled_share": sum(row["hit_token_cap"] for row in rows) / len(rows),
            "worst_item": max(shares, key=shares.get) if shares else None,
            "worst_share": max(shares.values()) if shares else 0.0,
            "note": "reported, not repaired; the D-2.5 analyzer applies the signed rule"}


def _write_rows(path: pathlib.Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    if args.depth > SIGNED_DEPTH:
        raise SystemExit(f"--depth {args.depth} exceeds the signed depth {SIGNED_DEPTH} "
                         f"({len(actions.ACTION_NAMES)}x the cost per extra level)")
    if args.out_dir.exists():
        raise SystemExit(f"refusing to write into existing {args.out_dir}")

    items = bank.load_tsar_bank(bank.fetch(args.split))
    kept, dropped = select_items(items, args.n_sources, args.mode)
    census = tree_census(len(kept), args.depth)
    print(json.dumps({"kept_items": len(kept), "excluded": dropped, **census},
                     ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        # A word-count proxy for the cap: the real cap needs the model
        # tokenizer, but every OTHER failure on this path -- missing item
        # fields, the prompt template, the node-id scheme -- is reachable
        # without a GPU and is what this dry run exists to reach.
        roots = build_roots(kept, lambda text: token_cap(len(text.split())))
        sample = roots[0]
        print(f"roots built: {len(roots)}; first node_id {sample['node_id']!r}, "
              f"proxy cap {sample['cap']}")
        print("--- one rendered user message, first action ---")
        print(actions.user_message(sample["text_out"], actions.ACTION_NAMES[0],
                                   sample["target_cefr"])[:400])
        print("dry run: no model loaded, nothing generated")
        return

    from vllm import LLM, SamplingParams
    llm = LLM(model=args.agent_model, gpu_memory_utilization=args.gpu_memory_utilization,
              max_model_len=args.max_model_len, enforce_eager=False)
    tokenizer = llm.get_tokenizer()

    roots = build_roots(kept, lambda text: token_cap(len(tokenizer.encode(text))))

    rows, frontier = [], roots
    for _ in range(args.depth):
        produced = expand(llm, SamplingParams, frontier, args.chunk)
        rows.extend(produced)
        frontier = [{**r, "depth": r["depth"]} for r in produced]

    args.out_dir.mkdir(parents=True)
    _write_rows(args.out_dir / "tree_raw.jsonl", rows)
    print(f"{len(rows)} nodes written before readout", flush=True)

    probes = readout.CefrProbes(device=args.readout_device, batch_size=args.readout_batch_size)
    source_of = {item.text_id: item.original for item in kept}
    levels = probes.levels([r["text_out"] for r in rows], temperature=args.temperature)
    meanings = probes.meaning([source_of[r["text_id"]] for r in rows],
                              [r["text_out"] for r in rows])
    for row, level, meaning in zip(rows, levels, meanings):
        row["level_expected"] = level.level_expected
        row["level_official"] = level.level_official
        row["meaning_to_source"] = meaning
        row["fkgl"] = readout.fkgl(row["text_out"])

    source_levels = probes.levels([item.original for item in kept], temperature=args.temperature)
    base = {item.text_id: lvl for item, lvl in zip(kept, source_levels)}
    for row in rows:
        row["source_level_expected"] = base[row["text_id"]].level_expected
        row["source_fkgl"] = readout.fkgl(source_of[row["text_id"]])

    _write_rows(args.out_dir / "tree.jsonl", rows)
    (args.out_dir / "run_config.json").write_text(json.dumps({
        "mode": args.mode, "arm": "tsar_cefr GPU-2a counterfactual branch tree",
        "census": census, "excluded_items": dropped,
        "cap_pressure": cap_pressure_by_item(rows),
        "decoding": "greedy (temperature 0, seed 0); the action axis is enumerated, not sampled",
        "admitted_by": "G-S1 on GPU-1's rows (admission_report_excluded.json)",
        "provenance": provenance({
            "agent_model": args.agent_model, "split": args.split, "depth": args.depth,
            "readout_temperature": args.temperature, "excluded_items": list(EXCLUDED_ITEMS),
            "actions": list(actions.ACTION_NAMES)}),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"written to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
