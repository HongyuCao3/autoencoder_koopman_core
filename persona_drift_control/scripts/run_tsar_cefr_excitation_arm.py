#!/usr/bin/env python3
"""GPU-1 for the `tsar_cefr` line: the open-loop random-excitation arm.
Plan section 4 (docs/operational_plan_tsar_cefr_line_2026-09-13.md).

WHAT IT BUYS. Identification data, and with it FOUR of the seven death
conditions at once (kill_criterion.md section 2), which is why it is a method
job and not an instrument one:

  D-0    saturation: does the model land on the target level in ONE move?
  D-1 2  post-run range: does `ell_t` still take >= 3 values at every step?
  D-2    executor authority: does `step_down` beat `copy` on the next step?
  D-2.5  the causal upper bound on everything a closed loop could add,
         computed on THESE rows before any operator is fitted.

ADMITTED BY G-T1, WHICH PASSED ON 2026-09-14 UNDER A REVISED CRITERION 1b.
That revision loosened the gate after the data was seen; the four-point record
of why is in docs/experiments/tsar_cefr_results.md, and every number this arm
produces inherits the caption obligation written there (2 of 40 trial pairs
ordered wrong: `03-b1` a classifier misread, `12-b1` a data property).

THREE THINGS ABOUT THE DESIGN THAT LOOK LIKE MISTAKES AND ARE NOT.

1. Decoding is greedy and the three seeds still differ. The seed does not
   perturb generation -- it draws the ACTION SEQUENCE. Excitation is the
   variation this arm exists to create; sampling noise on top of it would only
   inflate the variance of every gain estimate. Two seeds that happen to draw
   the same six actions for the same item produce byte-identical trajectories,
   which is correct and is counted (`duplicate_action_sequences`).

2. `copy` is generated, not spliced. It is the zero-input control that
   identifies `A`, and what it must measure is what this harness does when
   told to do nothing -- see tsar_cefr_actions. `unchanged_verbatim` reports
   how often the model actually returned its input.

3. The readout runs AFTER generation, in a second pass, not inside the loop.
   It can, because the actions are independent of the state: this is the
   open-loop arm, so nothing in the loop needs to know `ell_t`. That keeps the
   CEFR classifiers and vLLM off the same GPU at the same time.

THE TEMPERATURE OF THE CONTROL READOUT IS THE ONE G-T1 SIGNED (6.5, fitted by
NLL on the 40 trial references). It is not re-fitted here: re-fitting it on
this arm's own rows would make the readout a function of the data it reads.

NOT SUBMITTED WITHOUT A USER RULING (.claude/experiments.md item 1).
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import pathlib
import statistics
import sys
import time

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "persona_drift_control" / "src"))

from persona_drift import tsar_cefr_actions as actions          # noqa: E402
from persona_drift import tsar_cefr_bank as bank                # noqa: E402
from persona_drift import tsar_cefr_readout as readout          # noqa: E402
from persona_drift.run_provenance import provenance             # noqa: E402

SIGNED_TEMPERATURE = 6.5            # G-T1, fitted by NLL on the trial references
ACTION_SHARE_BAND = (0.20, 0.30)    # plan section 4.3
MIN_STEP_DISTINCT_LEVELS = 3        # D-1 (2)
MAX_ITEM_CAP_HIT_SHARE = 0.05       # plan section 4.3, the S1 precedent
SATURATION_HIT_THRESHOLD = 0.80     # D-0
CAP_LENGTH_MULTIPLIER = 1.5         # plan section 4.1
MIN_MAX_NEW_TOKENS = 128


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--agent-model", required=True)
    p.add_argument("--out-dir", type=pathlib.Path, required=True, help="Must not already exist.")
    p.add_argument("--split", default="test", choices=sorted(bank.SPLITS))
    p.add_argument("--probe-split", default="trial", choices=sorted(bank.SPLITS),
                   help="D-0 saturation probe runs on the split G-T1 used, not on the arm's own items")
    p.add_argument("--n-steps", type=int, default=6)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--n-sources", type=int, default=None,
                   help="debug only: truncate the bank to the first N source paragraphs")
    p.add_argument("--temperature", type=float, default=SIGNED_TEMPERATURE,
                   help="control-readout temperature; the G-T1 value, not re-fitted here")
    p.add_argument("--readout-device", type=int, default=0)
    p.add_argument("--readout-batch-size", type=int, default=16)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--skip-saturation-probe", action="store_true",
                   help="debug only; D-0 cannot be judged without it")
    p.add_argument("--dry-run", action="store_true",
                   help="build every prompt and the whole schedule, load no model, write nothing")
    p.add_argument("--mode", choices=["debug", "canonical"], required=True)
    return p.parse_args()


# --------------------------------------------------------------------------
# the schedule
# --------------------------------------------------------------------------

def build_schedule(items, seeds, n_steps) -> list[dict]:
    """One entry per trajectory: an item, a seed, and the six actions it draws."""
    return [{"text_id": item.text_id, "source_id": item.source_id,
             "target_cefr": item.target_cefr, "seed": seed,
             "source_text": item.original,
             "actions": actions.draw_actions(seed, item.text_id, n_steps)}
            for item in items for seed in seeds]


def duplicate_action_sequences(schedule: list[dict]) -> int:
    """Trajectories that two seeds drew identically. Under greedy decoding these
    are byte-identical rows -- expected at 4^n_steps, reported so a reader does
    not read them as a bug."""
    seen = collections.Counter((entry["text_id"], tuple(entry["actions"])) for entry in schedule)
    return sum(count - 1 for count in seen.values() if count > 1)


def token_cap(n_source_tokens: int) -> int:
    return max(MIN_MAX_NEW_TOKENS, int(math.ceil(n_source_tokens * CAP_LENGTH_MULTIPLIER)))


# --------------------------------------------------------------------------
# admission (plan section 4.3) and the two death conditions this job decides
# --------------------------------------------------------------------------

def action_balance(rows: list[dict]) -> dict:
    counts = collections.Counter(row["action"] for row in rows)
    shares = {name: counts.get(name, 0) / len(rows) for name in actions.ACTION_NAMES}
    lo, hi = ACTION_SHARE_BAND
    return {"shares": shares, "band": [lo, hi], "n_rows": len(rows),
            "passed": all(lo <= share <= hi for share in shares.values())}


def per_step_range(rows: list[dict]) -> dict:
    """D-1 (2): the post-run version of the G-T1 range question, on real rows."""
    by_step: dict[int, list[float]] = collections.defaultdict(list)
    for row in rows:
        by_step[row["step"]].append(row["level_expected"])
    steps = {}
    for step, levels in sorted(by_step.items()):
        distinct = len({round(v / 0.05) for v in levels})
        sd = statistics.stdev(levels) if len(levels) > 1 else 0.0
        steps[step] = {"distinct": distinct, "sd": sd, "n": len(levels),
                       "passed": distinct >= MIN_STEP_DISTINCT_LEVELS and sd > 0}
    return {"per_step": steps, "threshold_distinct": MIN_STEP_DISTINCT_LEVELS,
            "passed": all(s["passed"] for s in steps.values()),
            "death_condition": "D-1 (2)"}


def cap_pressure(rows: list[dict]) -> dict:
    """Per ITEM, not pooled: the S1 lesson is that a cap bites on the items whose
    text is long, and pooling hides exactly those."""
    by_item: dict[str, list[bool]] = collections.defaultdict(list)
    for row in rows:
        by_item[row["text_id"]].append(bool(row["hit_token_cap"]))
    shares = {text_id: sum(hits) / len(hits) for text_id, hits in by_item.items()}
    over = sorted(t for t, s in shares.items() if s > MAX_ITEM_CAP_HIT_SHARE)
    return {"threshold": MAX_ITEM_CAP_HIT_SHARE, "items_over_cap": over,
            "n_items_over_cap": len(over),
            "worst": max(shares.values()) if shares else 0.0,
            "pooled_share": sum(row["hit_token_cap"] for row in rows) / len(rows),
            "passed": not over}


def saturation_probe_verdict(probe_rows: list[dict]) -> dict:
    """D-0: one shot at the target level. Above 80% there is nothing to regulate."""
    hits = sum(1 for row in probe_rows if row["level_official"] == row["target_cefr"])
    share = hits / len(probe_rows) if probe_rows else float("nan")
    return {"n": len(probe_rows), "hits": hits, "one_shot_hit_share": share,
            "threshold": SATURATION_HIT_THRESHOLD,
            "fires": bool(probe_rows) and share > SATURATION_HIT_THRESHOLD,
            "death_condition": "D-0",
            "note": "official readout (top-1 confidence classifier's argmax), not the control readout"}


def unchanged_verbatim(rows: list[dict]) -> dict:
    zero = [row for row in rows if row["action"] == actions.ZERO_INPUT_ACTION]
    same = sum(1 for row in zero if row["text_in"].strip() == row["text_out"].strip())
    return {"n_copy_steps": len(zero),
            "share": same / len(zero) if zero else float("nan"),
            "note": "a property of the harness's null action, not a defect"}


def arm_report(rows: list[dict], probe_rows: list[dict], schedule: list[dict]) -> dict:
    admission = {"row_count": {"value": len(rows),
                               "expected": len(schedule) * max(r["step"] for r in rows),
                               "passed": len(rows) == len(schedule) * max(r["step"] for r in rows)},
                 "action_balance": action_balance(rows),
                 "cap_pressure": cap_pressure(rows),
                 "per_step_range": per_step_range(rows)}
    return {"gate": "G-S1 (admission) + D-0 / D-1 (2)", "line": "tsar_cefr",
            "admitted": all(block["passed"] for block in admission.values()),
            "admission": admission,
            "saturation_probe": saturation_probe_verdict(probe_rows),
            "unchanged_verbatim": unchanged_verbatim(rows),
            "duplicate_action_sequences": duplicate_action_sequences(schedule),
            "still_open": ["D-2 executor authority", "D-2.5 causal upper bound"],
            "decided_by": "scripts/analyze_tsar_cefr_gpu1.py, zero GPU, on these rows"}


# --------------------------------------------------------------------------

def generate_trajectories(llm, sampling_cls, tokenizer, schedule, n_steps) -> list[dict]:
    """Six batched sweeps. Step t's input is step t-1's output; the action for
    step t was drawn before the arm started and does not look at either."""
    current = [entry["source_text"] for entry in schedule]
    caps = [token_cap(len(tokenizer.encode(text))) for text in current]
    rows: list[dict] = []
    for step in range(1, n_steps + 1):
        conversations = [
            [{"role": "system", "content": actions.SYSTEM_PROMPT},
             {"role": "user", "content": actions.user_message(
                 text, entry["actions"][step - 1], entry["target_cefr"])}]
            for entry, text in zip(schedule, current)
        ]
        params = [sampling_cls(temperature=0.0, max_tokens=cap, seed=0) for cap in caps]
        started = time.time()
        # `chat` + enable_thinking=False is this repo's path to Qwen3 (S1/S3
        # runners). A reasoning preamble here would land in `text_out` and become
        # the next step's INPUT, so the harness would be rewriting its own notes.
        outputs = llm.chat(conversations, params, chat_template_kwargs={"enable_thinking": False})
        elapsed = time.time() - started
        nxt = []
        for entry, text_in, output in zip(schedule, current, outputs):
            completion = output.outputs[0]
            text_out = completion.text.strip()
            rows.append({
                "text_id": entry["text_id"], "source_id": entry["source_id"],
                "target_cefr": entry["target_cefr"], "seed": entry["seed"],
                "step": step, "action": entry["actions"][step - 1],
                "text_in": text_in, "text_out": text_out,
                "n_tokens_out": len(completion.token_ids),
                "hit_token_cap": completion.finish_reason == "length",
            })
            nxt.append(text_out if text_out else text_in)
        current = nxt
        print(f"step {step}/{n_steps}: {len(outputs)} generations in {elapsed/60:.1f} min", flush=True)
    return rows


def read_out(rows: list[dict], probe_rows: list[dict], schedule: list[dict], args) -> None:
    """One pass over every generated text. Mutates rows in place.

    The arm's source text comes from the SCHEDULE, keyed by text_id; the probe's
    comes from its own `text_in`. They are never merged into one map: trial and
    test text_ids differ only in the case of their suffix (`01-a2` / `01-A2`),
    so a shared map would be one `.upper()` away from silently pairing an item
    with the wrong source.
    """
    probes = readout.CefrProbes(device=args.readout_device, batch_size=args.readout_batch_size)
    source_of = {entry["text_id"]: entry["source_text"] for entry in schedule}

    for batch, sources_for in ((rows, lambda r: source_of[r["text_id"]]),
                               (probe_rows, lambda r: r["text_in"])):
        if not batch:
            continue
        texts = [row["text_out"] for row in batch]
        levels = probes.levels(texts, temperature=args.temperature)
        meanings = probes.meaning([sources_for(row) for row in batch], texts)
        for row, level, meaning in zip(batch, levels, meanings):
            row["level_expected"] = level.level_expected
            row["level_official"] = level.level_official
            row["meaning_to_source"] = meaning
            row["fkgl"] = readout.fkgl(row["text_out"])

    # Step 0 is the source itself: the operator needs the state it started from.
    sources = sorted({(entry["text_id"], entry["source_text"]) for entry in schedule})
    levels = probes.levels([text for _, text in sources], temperature=args.temperature)
    baseline = {text_id: level for (text_id, _), level in zip(sources, levels)}
    for row in rows:
        row["source_level_expected"] = baseline[row["text_id"]].level_expected
        row["source_fkgl"] = readout.fkgl(source_of[row["text_id"]])


def _write_rows(path: pathlib.Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    if args.out_dir.exists():
        raise SystemExit(f"refusing to write into existing {args.out_dir}")

    path = bank.fetch(args.split)
    items = bank.load_tsar_bank(path)
    if args.n_sources is not None:
        if args.mode == "canonical":
            raise SystemExit("--n-sources truncates the bank; it is a debug switch")
        keep = set(bank.source_ids(items)[:args.n_sources])
        items = [item for item in items if item.source_id in keep]
    schedule = build_schedule(items, args.seeds, args.n_steps)

    probe_items = []
    if not args.skip_saturation_probe:
        probe_items = bank.load_tsar_bank(bank.fetch(args.probe_split))

    print(f"{len(schedule)} trajectories x {args.n_steps} steps = "
          f"{len(schedule) * args.n_steps} generations, "
          f"+ {len(probe_items)} saturation probes", flush=True)
    print(f"duplicate action sequences across seeds: {duplicate_action_sequences(schedule)}", flush=True)

    if args.dry_run:
        sample = schedule[0]
        print("\n--- first step prompt of the first trajectory ---")
        print(actions.user_message(sample["source_text"], sample["actions"][0], sample["target_cefr"])[:600])
        if probe_items:
            print("\n--- first saturation probe prompt ---")
            print(actions.saturation_message(probe_items[0].original, probe_items[0].target_cefr)[:600])
        print(f"\ndry run: no model loaded, nothing written to {args.out_dir}")
        return

    prov = provenance(switches={
        "agent_model": args.agent_model, "split": args.split,
        "n_steps": args.n_steps, "seeds": args.seeds,
        "readout_temperature": args.temperature,
        "cefr_models": list(readout.CEFR_MODELS), "meaning_model": readout.MEANING_MODEL,
        "actions": list(actions.ACTION_NAMES), "decoding": "greedy",
        "data_revision": bank.SPLITS[args.split]["revision"],
        "data_content_sha256": bank.content_sha256(path),
    })

    from vllm import LLM, SamplingParams
    llm = LLM(model=args.agent_model, dtype="bfloat16",
              gpu_memory_utilization=args.gpu_memory_utilization,
              max_model_len=args.max_model_len)
    tokenizer = llm.get_tokenizer()

    rows = generate_trajectories(llm, SamplingParams, tokenizer, schedule, args.n_steps)

    probe_rows = []
    if probe_items:
        conversations = [
            [{"role": "system", "content": actions.SYSTEM_PROMPT},
             {"role": "user", "content": actions.saturation_message(item.original, item.target_cefr)}]
            for item in probe_items]
        caps = [token_cap(len(tokenizer.encode(item.original))) for item in probe_items]
        outputs = llm.chat(conversations,
                           [SamplingParams(temperature=0.0, max_tokens=c, seed=0) for c in caps],
                           chat_template_kwargs={"enable_thinking": False})
        probe_rows = [{"text_id": item.text_id, "source_id": item.source_id,
                       "target_cefr": item.target_cefr, "step": 1, "action": "one_shot",
                       "text_in": item.original, "text_out": output.outputs[0].text.strip(),
                       "n_tokens_out": len(output.outputs[0].token_ids),
                       "hit_token_cap": output.outputs[0].finish_reason == "length"}
                      for item, output in zip(probe_items, outputs)]

    # ROWS BEFORE READOUT. vLLM does not reliably hand its KV cache back inside
    # the same process, so the classifier pass is the most likely place this job
    # dies -- and it is the half that costs nothing to redo. The S3 precedent:
    # write rows before any summary, so a failure loses the tail, not the run.
    args.out_dir.mkdir(parents=True)
    _write_rows(args.out_dir / "trajectories_raw.jsonl", rows)
    if probe_rows:
        _write_rows(args.out_dir / "saturation_probe_raw.jsonl", probe_rows)
    print(f"{len(rows)} generated rows written before readout", flush=True)

    del llm
    import gc
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except ImportError:
        pass

    read_out(rows, probe_rows, schedule, args)

    _write_rows(args.out_dir / "trajectories.jsonl", rows)
    if probe_rows:
        _write_rows(args.out_dir / "saturation_probe.jsonl", probe_rows)

    report = arm_report(rows, probe_rows, schedule)
    (args.out_dir / "arm_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.out_dir / "run_config.json").write_text(json.dumps({
        "mode": args.mode, "arm": "tsar_cefr GPU-1 open-loop random excitation",
        "provenance": prov, "n_trajectories": len(schedule),
        "n_generations": len(rows) + len(probe_rows),
        "data_path": str(path),
        "admitted_by": "G-T1 PASS 2026-09-14 (criterion 1b revised; see tsar_cefr_results.md)",
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nadmitted: {report['admitted']}")
    for name, block in report["admission"].items():
        print(f"  [{'ok ' if block['passed'] else 'FAIL'}] {name}")
    probe = report["saturation_probe"]
    print(f"  D-0 one-shot hit share {probe['one_shot_hit_share']:.3f} "
          f"(fires above {probe['threshold']}): {probe['fires']}")
    print(f"written to {args.out_dir}")


if __name__ == "__main__":
    main()
