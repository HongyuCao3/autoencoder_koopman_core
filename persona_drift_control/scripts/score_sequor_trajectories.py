#!/usr/bin/env python3
"""Apply the graded readout `y_t` to a `constraint`-line arm's trajectories.

`y_t = (satisfied active constraints) / k`, k=3, from upstream SEQUOR's own
per-constraint judge prompt and verdict parser (vendored verbatim in
persona_drift.sequor_constraint_judge -- this project changes ONLY the
aggregation: upstream collapses the k decisions with `all()` into a binary
turn success, and both readouts are written here).

Scoring is a separate job from generation on purpose. The same responses get
scored by the in-loop (self) judge and by the independent reporting judge, so:

- the two readouts can be compared on the SAME rows, which is the only way to
  see the self-judging bias that hid a 4x effect size on the `defense` line;
- `--judge-kind` is recorded in the artifact, so a report-side guard can tell
  a selection signal from a reportable number (.claude/global.md -> 报告口径);
- swapping the judge never means re-generating trajectories, the coupling that
  made an instrument fix cost the ERGO line 18 jobs (LEDGER section 2).

Runs in the `constraint` line's vLLM environment.
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
from persona_drift.sequor_constraint_judge import (  # noqa: E402
    CONSTRAINT_JUDGE_PROMPT_TEMPLATE,
    binary_turn_success,
    extract_verdict,
    graded_readout,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm-dir", type=pathlib.Path, required=True,
                   help="A run directory containing trajectories.jsonl.")
    p.add_argument("--judge-model", required=True)
    p.add_argument("--judge-kind", choices=["self", "independent"], required=True,
                   help="`self` = the agent's own model (a SELECTION signal, never reported); "
                        "`independent` = the reporting judge chosen by S0. Required, not inferred: "
                        "inferring it from a model id is exactly how a self-judged number gets "
                        "reported by accident.")
    p.add_argument("--out-path", type=pathlib.Path, required=True, help="JSON report; must not exist.")
    p.add_argument("--max-new-tokens", type=int, default=512,
                   help="512 measured sufficient on the S0 smoke: judge output median ~185 tokens, "
                        "max ~340, 0/24 parse failures, 0/24 truncated.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--max-model-len", type=int, default=8192)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.out_path.exists():
        raise SystemExit(f"refusing to overwrite existing {args.out_path}")
    rows = [json.loads(l) for l in (args.arm_dir / "trajectories.jsonl").open() if l.strip()]
    arm_config = json.loads((args.arm_dir / "run_config.json").read_text())
    if arm_config.get("model_id") == args.judge_model and args.judge_kind == "independent":
        raise SystemExit(
            f"--judge-kind independent but the judge IS the agent's model "
            f"({args.judge_model}): that is a self-judged score, and .claude/global.md forbids "
            f"reporting one. Pass --judge-kind self, or judge with a different model."
        )

    prov = provenance(switches={
        "judge_model": args.judge_model, "judge_kind": args.judge_kind, "arm_dir": str(args.arm_dir),
        "arm_mode": arm_config.get("mode"), "agent_model": arm_config.get("model_id"),
        "max_new_tokens": args.max_new_tokens, "seed": args.seed,
    })
    print(f"harness {prov['git_sha'][:12]}"
          + (f"  job {prov['job_id']}" if prov["job_id"] else ""))
    print(f"{len(rows)} rows x {len(rows[0]['constraints'])} constraints = "
          f"{len(rows) * len(rows[0]['constraints'])} judge calls"
          + ("   [arm mode=debug: not evidence]" if arm_config.get("mode") == "debug" else ""))

    calls = [
        (i, j, CONSTRAINT_JUDGE_PROMPT_TEMPLATE.format(answer=row["agent_message"], constraint=constraint))
        for i, row in enumerate(rows)
        for j, constraint in enumerate(row["constraints"])
    ]

    from vllm import LLM, SamplingParams

    llm = LLM(model=args.judge_model, seed=args.seed, dtype="bfloat16",
              gpu_memory_utilization=args.gpu_memory_utilization, max_model_len=args.max_model_len)
    params = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens, seed=args.seed)

    t0 = time.time()
    outputs = llm.chat([[{"role": "user", "content": prompt}] for _, _, prompt in calls], params,
                       chat_template_kwargs={"enable_thinking": False})
    elapsed = time.time() - t0

    verdicts: dict[int, list[bool | None]] = {i: [None] * len(rows[i]["constraints"]) for i in range(len(rows))}
    truncated = 0
    for (i, j, _), out in zip(calls, outputs):
        verdicts[i][j] = extract_verdict(out.outputs[0].text)
        truncated += out.outputs[0].finish_reason == "length"

    scored = []
    for i, row in enumerate(rows):
        y, n_failed = graded_readout(verdicts[i])
        scored.append({
            "trajectory_id": row["trajectory_id"], "item_id": row["item_id"], "turn": row["turn"],
            "branch": row["branch"], "u_remind": row["u_remind"],
            "prefix_sha256": row["prefix_sha256"],
            "followed": verdicts[i], "y_graded": y, "y_binary": binary_turn_success(verdicts[i]),
            "n_judge_parse_failures": n_failed,
            "echo_jaccard_prev": row["echo_jaccard_prev"],
            "inserted_tokens": row["inserted_tokens"], "hit_token_cap": row["hit_token_cap"],
        })

    usable = [s for s in scored if s["y_graded"] is not None]
    per_turn = {}
    for s in usable:
        per_turn.setdefault((s["branch"], s["turn"]), []).append(s["y_graded"])
    range_by_turn = {
        f"{branch}_t{turn}": {
            "n": len(vals), "mean": statistics.fmean(vals),
            "sd": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "distinct": len(set(vals)),
        }
        for (branch, turn), vals in sorted(per_turn.items())
    }

    report = {
        "mode": arm_config.get("mode"), "backend": "vllm", "provenance": prov,
        "arm_dir": str(args.arm_dir), "agent_model": arm_config.get("model_id"),
        "judge_model": args.judge_model, "judge_kind": args.judge_kind,
        "k_constraints": len(rows[0]["constraints"]),
        "n_rows": len(rows), "n_judge_calls": len(calls),
        "n_rows_unusable": len(scored) - len(usable),
        "n_judge_truncated": truncated,
        "elapsed_s": elapsed, "seconds_per_call": elapsed / len(calls),
        "readout_range_by_turn": range_by_turn,
        "rows": scored,
    }
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print(f"\n{len(calls)} judge calls in {elapsed/60:.1f} min ({elapsed/len(calls):.2f} s/call)")
    print(f"rows with a parse failure (y undefined): {len(scored) - len(usable)}/{len(scored)}")
    print(f"judge outputs truncated at {args.max_new_tokens}: {truncated}")
    flat = [s["y_graded"] for s in usable]
    print(f"y over all usable rows: mean {statistics.fmean(flat):.4f}, distinct values {sorted(set(flat))}")
    print(f"judge_kind={args.judge_kind}"
          + ("  (SELECTION signal only -- never a reported number)" if args.judge_kind == "self" else ""))
    print(f"report written to {args.out_path}")


if __name__ == "__main__":
    main()
