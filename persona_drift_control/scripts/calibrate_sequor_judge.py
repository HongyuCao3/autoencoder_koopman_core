#!/usr/bin/env python3
"""S0 judge calibration for the `constraint` line
(docs/experiments/constraint_retention_plan.md section 3, gates G-S0-1/G-S0-2;
docs/experiments/constraint_signal_screening.md section 9.1, gate G-S0-0).

Runs one candidate judge over the vendored SEQUOR gold set (2000 rows, exactly
label-balanced) and reports whether it can tell a constraint-following answer
from a constraint-violating one -- using upstream's own prompt and verdict
parser, so the numbers are comparable to upstream's judge calibration.

G-S0-0 exists because the gold set has a LENGTH CONFOUND: `violate` responses
are systematically longer (median 478 vs 378 words), so a rule that reads no
constraint at all scores 64.8% paired / 59.7% unpaired. Any accuracy reported
without those floors alongside it does not establish that the judge is reading
the constraint. This is the same discipline ERGO failed on `turn`: the null
must be given every cheap exogenous signal the model gets.

One GPU, one model, no sampling (temperature 0, do_sample off).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from persona_drift.chat_model import ChatModel, GenerationConfig  # noqa: E402
from persona_drift.sequor_calibration_metrics import (  # noqa: E402
    cohens_kappa,
    length_floors,
    stratified_slice,
)
from persona_drift.run_provenance import provenance  # noqa: E402
from persona_drift.sequor_constraint_judge import (  # noqa: E402
    extract_verdict,
    judge_constraint_followed,
)

# pipeline/best_judge, upstream's own best judge, for context in the report.
UPSTREAM_REFERENCE = {"judge": "GPT-oss-120B", "gold_yes_accuracy": 0.8830, "gold_no_accuracy": 0.9880}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--judge-model", required=True)
    p.add_argument("--gold-path", type=pathlib.Path,
                   default=pathlib.Path("resources/sequor/sequor_gold_judge_calibration.jsonl"))
    p.add_argument("--out-path", type=pathlib.Path, required=True, help="JSON report; must not already exist.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--enable-thinking", action="store_true")
    p.add_argument("--limit", type=int, default=None,
                   help="Score only N rows, sampled EVENLY across the four (source_model, label) "
                        "cells -- a head slice would be all gpt-5.2/follow, since the gold file is "
                        "ordered by cell. SMOKE TEST ONLY: a limited run writes mode='debug' into "
                        "the report and its numbers are not evidence.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.out_path.exists():
        raise SystemExit(f"refusing to overwrite existing {args.out_path}")
    rows = [json.loads(l) for l in args.gold_path.open() if l.strip()]
    if args.limit:
        rows = stratified_slice(rows, args.limit)
    print(f"{len(rows)} gold rows from {args.gold_path}"
          + ("  [DEBUG: --limit set, not evidence]" if args.limit else ""))

    prov = provenance(switches={
        "backend": "hf", "judge_model": args.judge_model, "device": args.device,
        "max_new_tokens": args.max_new_tokens, "enable_thinking": args.enable_thinking,
        "seed": args.seed, "limit": args.limit, "gold_path": str(args.gold_path),
    })
    print(f"harness {prov['git_sha'][:12]}"
          + (f" (+{prov['n_dirty_paths']} uncommitted paths)" if prov["git_dirty"] else "")
          + (f"  job {prov['job_id']}" if prov["job_id"] else ""))

    judge = ChatModel(args.judge_model, device=args.device, enable_thinking=args.enable_thinking)
    config = GenerationConfig(max_new_tokens=args.max_new_tokens, temperature=0.0, do_sample=False)

    scored, t0 = [], time.time()
    for i, r in enumerate(rows):
        followed, raw = judge_constraint_followed(
            judge, r["constraint"], r["response"], seed=args.seed,
            config=config, enable_thinking=args.enable_thinking)
        scored.append({
            "gold_id": r["gold_id"], "label": r["label"], "source_model": r["source_model"],
            "constraint": r["constraint"], "predicted_followed": followed,
            "n_output_chars": len(raw), "hit_token_cap": len(raw) > 0 and extract_verdict(raw) is None,
        })
        if (i + 1) % 100 == 0:
            rate = (time.time() - t0) / (i + 1)
            print(f"  {i+1}/{len(rows)}  {rate:.2f} s/row  eta {(len(rows)-i-1)*rate/60:.1f} min", flush=True)

    parsed = [s for s in scored if s["predicted_followed"] is not None]
    n_failed = len(scored) - len(parsed)
    pred = [s["predicted_followed"] for s in parsed]
    gold = [s["label"] == "follow" for s in parsed]

    def acc(label: str) -> float | None:
        sub = [(p, g) for p, g, s in zip(pred, gold, parsed) if s["label"] == label]
        return sum(1 for p, g in sub if p == g) / len(sub) if sub else None

    gold_yes, gold_no = acc("follow"), acc("violate")
    overall = sum(1 for p, g in zip(pred, gold) if p == g) / len(pred) if pred else 0.0
    balanced = (gold_yes + gold_no) / 2 if (gold_yes is not None and gold_no is not None) else None
    kappa = cohens_kappa(pred, gold) if pred else 0.0
    floors = length_floors(rows)

    g_s0_1 = bool(overall >= 0.80 and kappa >= 0.60)
    g_s0_2 = bool(balanced is not None and balanced >= 0.75 and kappa >= 0.40)

    fmt = lambda v: "  n/a " if v is None else f"{v:.4f}"
    print(f"\n=== {args.judge_model} on {len(rows)} gold rows "
          f"(parse failures {n_failed}, {n_failed/len(scored)*100:.1f}%) ===")
    print(f"  overall agreement   {fmt(overall)}")
    print(f"  Gold-Yes  (follow)  {fmt(gold_yes)}   [upstream {UPSTREAM_REFERENCE['judge']}: "
          f"{UPSTREAM_REFERENCE['gold_yes_accuracy']:.4f}]")
    print(f"  Gold-No (violate)   {fmt(gold_no)}   [upstream: {UPSTREAM_REFERENCE['gold_no_accuracy']:.4f}]")
    print(f"  balanced accuracy   {fmt(balanced)}")
    print(f"  Cohen's kappa       {fmt(kappa)}")
    print(f"  seconds per row     {(time.time()-t0)/max(len(scored),1):.2f}"
          f"   -> full 2000-row run would take {(time.time()-t0)/max(len(scored),1)*2000/3600:.2f} h")
    print(f"\n=== G-S0-0 length floors (what a judge reading NO constraint scores) ===")
    print(f"  paired 'shorter one follows'    {fmt(floors['paired_accuracy'])}  ({floors['paired_cells']} cells)")
    print(f"  unpaired 'shorter than median'  {fmt(floors['unpaired_accuracy'])}")
    if floors["paired_accuracy"] is not None:
        print(f"  -> this judge is {(overall - floors['paired_accuracy'])*100:+.1f} points over the paired floor")
    print(f"\n  G-S0-1 (agreement >= 0.80 AND kappa >= 0.60): {'PASS' if g_s0_1 else 'FAIL'}")
    print(f"  G-S0-2 (balanced >= 0.75 AND kappa >= 0.40):   {'PASS' if g_s0_2 else 'FAIL'}")

    report = {
        "mode": "debug" if args.limit else "canonical", "backend": "hf",
        "provenance": prov,
        "judge_model": args.judge_model, "gold_path": str(args.gold_path), "n_rows": len(rows),
        "seed": args.seed, "max_new_tokens": args.max_new_tokens, "enable_thinking": args.enable_thinking,
        "n_parse_failures": n_failed, "parse_failure_rate": n_failed / len(scored),
        "overall_agreement": overall, "gold_yes_accuracy": gold_yes, "gold_no_accuracy": gold_no,
        "balanced_accuracy": balanced, "cohens_kappa": kappa,
        "g_s0_0_length_floors": floors, "upstream_reference": UPSTREAM_REFERENCE,
        "g_s0_1_pass": g_s0_1, "g_s0_2_pass": g_s0_2,
        "elapsed_s": time.time() - t0, "seconds_per_row": (time.time() - t0) / max(len(scored), 1),
        "rows": scored,
    }
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_path.write_text(json.dumps(report, indent=2))
    print(f"\nreport written to {args.out_path}")


if __name__ == "__main__":
    main()
