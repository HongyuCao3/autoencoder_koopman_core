#!/usr/bin/env python3
"""S0 judge calibration on vLLM -- the same measurement as
scripts/calibrate_sequor_judge.py, on a different serving stack.

Why both exist: the HF-transformers runner measured 7.42 s/row (job 15719117),
which makes the 2000-row calibration 4.12 h per candidate and S1's 9600 judge
calls ~20 h. vLLM batches them. But swapping the serving stack under a
measurement instrument is exactly the move that cost the ERGO line 18 jobs
re-answering one question, so this script's FIRST job is not to calibrate
anything -- it is `--compare-to`, which replays a previous HF report's rows and
reports verdict-level agreement. Calibrating on an unverified stack is not
allowed to be the cheap path.

The prompt and parser come from persona_drift.sequor_constraint_judge and the
metrics from persona_drift.sequor_calibration_metrics -- the same objects the
HF runner uses, not copies, so a disagreement between the two reports can only
come from the model or the sampler.

Greedy decoding, and `enable_thinking=False` passed through the chat template
exactly as ChatModel does it.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import time


def load_shared_module(name: str):
    """Load src/persona_drift/<name>.py by path, without importing the package.

    `persona_drift/__init__.py` imports .analysis, which needs pandas, and this
    script's environment (plan section 8) has vLLM and nothing else -- a dry run
    on 2026-09-09 died on exactly that ModuleNotFoundError, before any GPU was
    requested. Loading the file directly keeps the property the separate env was
    built to preserve: this runner and the HF runner read the same prompt,
    parser and metric SOURCE, not copies of them.
    """

    path = pathlib.Path(__file__).resolve().parents[1] / "src" / "persona_drift" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_sequor_vllm_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_metrics = load_shared_module("sequor_calibration_metrics")
_judge = load_shared_module("sequor_constraint_judge")
_prov = load_shared_module("run_provenance")

provenance = _prov.provenance

cohens_kappa = _metrics.cohens_kappa
length_floors = _metrics.length_floors
stratified_slice = _metrics.stratified_slice
CONSTRAINT_JUDGE_PROMPT_TEMPLATE = _judge.CONSTRAINT_JUDGE_PROMPT_TEMPLATE
extract_verdict = _judge.extract_verdict

UPSTREAM_REFERENCE = {"judge": "GPT-oss-120B", "gold_yes_accuracy": 0.8830, "gold_no_accuracy": 0.9880}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--judge-model", required=True)
    p.add_argument("--gold-path", type=pathlib.Path,
                   default=pathlib.Path("resources/sequor/sequor_gold_judge_calibration.jsonl"))
    p.add_argument("--out-path", type=pathlib.Path, required=True, help="JSON report; must not already exist.")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--enable-thinking", action="store_true")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--limit", type=int, default=None,
                   help="Stratified over the four (source_model, label) cells. Writes mode='debug'.")
    p.add_argument("--compare-to", type=pathlib.Path, default=None,
                   help="A previous report (either backend). Rows shared with it are compared "
                        "verdict-by-verdict and the agreement is written into this report. "
                        "Run this BEFORE trusting any calibration produced on a new stack.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.out_path.exists():
        raise SystemExit(f"refusing to overwrite existing {args.out_path}")
    rows = [json.loads(l) for l in args.gold_path.open() if l.strip()]
    if args.limit:
        rows = stratified_slice(rows, args.limit)
    print(f"{len(rows)} gold rows" + ("  [DEBUG: --limit set, not evidence]" if args.limit else ""))

    prov = provenance(switches={
        "backend": "vllm", "judge_model": args.judge_model,
        "max_new_tokens": args.max_new_tokens, "enable_thinking": args.enable_thinking,
        "seed": args.seed, "limit": args.limit, "gold_path": str(args.gold_path),
        "gpu_memory_utilization": args.gpu_memory_utilization, "max_model_len": args.max_model_len,
        "compare_to": str(args.compare_to) if args.compare_to else None,
    })
    print(f"harness {prov['git_sha'][:12]}"
          + (f" (+{prov['n_dirty_paths']} uncommitted paths)" if prov["git_dirty"] else "")
          + (f"  job {prov['job_id']}" if prov["job_id"] else ""))

    from vllm import LLM, SamplingParams

    llm = LLM(model=args.judge_model, seed=args.seed, dtype="bfloat16",
              gpu_memory_utilization=args.gpu_memory_utilization, max_model_len=args.max_model_len)
    params = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens, seed=args.seed)

    conversations = [[{"role": "user", "content": CONSTRAINT_JUDGE_PROMPT_TEMPLATE.format(
        answer=r["response"], constraint=r["constraint"])}] for r in rows]

    t0 = time.time()
    outputs = llm.chat(conversations, params,
                       chat_template_kwargs={"enable_thinking": args.enable_thinking})
    elapsed = time.time() - t0

    scored = []
    for r, out in zip(rows, outputs):
        raw = out.outputs[0].text
        scored.append({
            "gold_id": r["gold_id"], "label": r["label"], "source_model": r["source_model"],
            "constraint": r["constraint"], "predicted_followed": extract_verdict(raw),
            "n_output_chars": len(raw), "n_output_tokens": len(out.outputs[0].token_ids),
            "finish_reason": out.outputs[0].finish_reason,
        })

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

    comparison = None
    if args.compare_to:
        prev = json.loads(args.compare_to.read_text())
        prev_by = {r["gold_id"]: r["predicted_followed"] for r in prev["rows"]}
        shared = [s for s in scored if s["gold_id"] in prev_by]
        agree = [s for s in shared if s["predicted_followed"] == prev_by[s["gold_id"]]]
        disagree = [{"gold_id": s["gold_id"], "label": s["label"],
                     "this_backend": s["predicted_followed"], "other_backend": prev_by[s["gold_id"]]}
                    for s in shared if s["predicted_followed"] != prev_by[s["gold_id"]]]
        comparison = {
            "compared_to": str(args.compare_to), "other_backend_model": prev.get("judge_model"),
            "other_backend": prev.get("backend", "hf"), "n_shared_rows": len(shared),
            "n_agree": len(agree), "agreement": len(agree) / len(shared) if shared else None,
            "disagreements": disagree,
        }
        print(f"\n=== backend consistency vs {args.compare_to} ===")
        print(f"  {len(agree)}/{len(shared)} verdicts identical"
              f"  ({comparison['agreement']*100:.1f}%)" if shared else "  no shared rows")
        for d in disagree:
            print(f"    DISAGREE {d['gold_id']} (gold {d['label']}): vllm={d['this_backend']} "
                  f"hf={d['other_backend']}")

    fmt = lambda v: "  n/a " if v is None else f"{v:.4f}"
    per_row = elapsed / max(len(scored), 1)
    print(f"\n=== {args.judge_model} on {len(rows)} gold rows via vLLM "
          f"(parse failures {n_failed}, {n_failed/len(scored)*100:.1f}%) ===")
    print(f"  overall agreement   {fmt(overall)}")
    print(f"  Gold-Yes  (follow)  {fmt(gold_yes)}   [upstream {UPSTREAM_REFERENCE['judge']}: "
          f"{UPSTREAM_REFERENCE['gold_yes_accuracy']:.4f}]")
    print(f"  Gold-No (violate)   {fmt(gold_no)}   [upstream: {UPSTREAM_REFERENCE['gold_no_accuracy']:.4f}]")
    print(f"  balanced accuracy   {fmt(balanced)}")
    print(f"  Cohen's kappa       {fmt(kappa)}")
    print(f"  batch wall time     {elapsed/60:.2f} min   ({per_row:.3f} s/row amortized)")
    print(f"  truncated at cap    {sum(1 for s in scored if s['finish_reason'] == 'length')}")
    print(f"\n=== G-S0-0 length floors ===")
    print(f"  paired 'shorter one follows'    {fmt(floors['paired_accuracy'])}  ({floors['paired_cells']} cells)")
    print(f"  unpaired 'shorter than median'  {fmt(floors['unpaired_accuracy'])}")
    g_s0_1 = bool(overall >= 0.80 and kappa >= 0.60)
    g_s0_2 = bool(balanced is not None and balanced >= 0.75 and kappa >= 0.40)
    print(f"\n  G-S0-1 (agreement >= 0.80 AND kappa >= 0.60): {'PASS' if g_s0_1 else 'FAIL'}")
    print(f"  G-S0-2 (balanced >= 0.75 AND kappa >= 0.40):   {'PASS' if g_s0_2 else 'FAIL'}")

    report = {
        "mode": "debug" if args.limit else "canonical", "backend": "vllm",
        "provenance": prov,
        "judge_model": args.judge_model, "gold_path": str(args.gold_path), "n_rows": len(rows),
        "seed": args.seed, "max_new_tokens": args.max_new_tokens, "enable_thinking": args.enable_thinking,
        "n_parse_failures": n_failed, "parse_failure_rate": n_failed / len(scored),
        "overall_agreement": overall, "gold_yes_accuracy": gold_yes, "gold_no_accuracy": gold_no,
        "balanced_accuracy": balanced, "cohens_kappa": kappa,
        "g_s0_0_length_floors": floors, "upstream_reference": UPSTREAM_REFERENCE,
        "g_s0_1_pass": g_s0_1, "g_s0_2_pass": g_s0_2,
        "backend_consistency": comparison,
        "elapsed_s": elapsed, "seconds_per_row": per_row, "rows": scored,
    }
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_path.write_text(json.dumps(report, indent=2))
    print(f"\nreport written to {args.out_path}")


if __name__ == "__main__":
    main()
