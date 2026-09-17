#!/usr/bin/env python3
"""Drive a SECOND BACKBONE through the `core` collection protocol.

The `core` corpus was supplied finished and this repository has no generator for
it, so a second backbone can only be run if the protocol is recovered from the
landed rows. Four legs, each verified against every landed row before this
runner existed -- none of them is assumed here:

  1. turn structure -- verified by scripts/verify_core_protocol_reconstruction.py
     (gate B0): history is append-only, the assistant turn is `raw_generation`,
     the user turn is the previous row's `feedback_text`, on 12,840 turn-pairs
     across eight tasks. So TURN-1 PROMPTS ARE REPLAYED VERBATIM from the landed
     rows rather than rebuilt, and only turns >= 2 are constructed.
  2. the scorer -- same script, 1.0000 row-exact on every task this runner
     admits.
  3. the feedback rule -- scripts/reconstruct_core_feedback.py, byte-exact on
     6,732 landed feedback rows.
  4. the normalisation -- recovered here as an affine map with clipping, pinned
     below per task; residuals against the landed columns are at float epsilon
     (<= 2e-15), which is why the bounds are written as exact numbers.

WHAT IT WRITES is the schema configs/dataset/*.yaml declares -- the
`output_columns` / `target_columns` that scripts/eval_surrogate_rows.py reads --
plus the raw measurements and the full prompt, so the arm can be re-scored
without being re-generated.

EVERY ARM NEEDS ITS PAIRED CONTROL. A landed Qwen column is NOT a valid control
for an arm collected by this runner unless that task's scorer reproduces at
1.0000; and even then the decoding here is the landed `decoding_config`, not
whatever the new backbone defaults to. Run the same task twice, once per
backbone, and compare inside the group.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from reconstruct_core_feedback import BUILDERS  # noqa: E402
from verify_core_protocol_reconstruction import (  # noqa: E402
    average_word_length,
    character_count,
    first_integer,
    word_count,
)


def _clip01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def _affine(lo: float, hi: float):
    return lambda v: _clip01((v - lo) / (hi - lo))


# (measure -> {field: value}) and (fields -> normalised outputs). The affine
# bounds are FITTED against the landed columns, not read off the prompt text.
TASK_SPEC = {
    "sentence_length_t10": {
        "dir": "scalar/sentence_length_t10", "turns": 10,
        "measure": lambda t: {"measured_raw_count": word_count(t)},
        "norm": {"normalized_output": ("measured_raw_count", _affine(5.0, 100.0))},
        "targets": ["effective_norm"],
    },
    "character_length_t5": {
        "dir": "scalar/character_length_t5", "turns": 5,
        "measure": lambda t: {"measured_raw_count": character_count(t)},
        "norm": {"normalized_output": ("measured_raw_count", _affine(0.5, 10.5))},
        "targets": ["effective_norm"],
    },
    "average_word_length_t5": {
        "dir": "scalar/average_word_length_t5", "turns": 5,
        "measure": lambda t: {"measured_raw_value": average_word_length(t)},
        "norm": {"normalized_output": ("measured_raw_value", _affine(2.0, 10.0))},
        "targets": ["effective_norm"],
    },
    "even_odd_t5": {
        "dir": "scalar/even_odd_t5", "turns": 5,
        "measure": lambda t: {"integer_value": first_integer(t)},
        # y is the PARITY CATEGORY (0 even / 1 odd), not correctness -- verified
        # on the landed rows, where normalized_output splits 60/60 and equals
        # effective_norm on every row. That identity is exactly why Table 1
        # reports this task as having no range.
        "norm": {"normalized_output": ("integer_value",
                                       lambda v: None if v is None else float(int(v) % 2))},
        "targets": ["effective_norm"],
    },
    "vector_count_stage1_t10": {
        "dir": "vector/vector_count_stage1_t10", "turns": 10,
        "measure": lambda t: {"word_count": word_count(t),
                              "average_word_length": average_word_length(t)},
        "norm": {"word_count_norm": ("word_count", _affine(5.0, 30.0)),
                 "avg_word_length_norm": ("average_word_length", _affine(3.0, 6.0))},
        "targets": ["target_word_count_norm", "target_avg_word_length_norm"],
    },
}

# Fields the feedback builder for a task reads off the PREVIOUS row, beyond what
# `measure` produces. They are carried forward from the landed turn-1 row.
CARRIED = ("topic", "topic_split", "trajectory_id", "target_raw_count", "target_raw_value",
           "target_word_count", "target_avg_word_length", "target_category",
           "effective_norm", "target_word_count_norm", "target_avg_word_length_norm",
           "requested_norm", "seed")


def provenance() -> dict:
    def git(*a: str) -> str:
        return subprocess.run(["git", "-C", str(REPO), *a], check=True,
                              capture_output=True, text=True).stdout.strip()
    dirty = git("status", "--porcelain")
    return {"git_sha": git("rev-parse", "HEAD"), "git_dirty": bool(dirty),
            "n_dirty_paths": len(dirty.splitlines())}


def load_seed_rows(task: str) -> list[dict]:
    """Turn-1 rows, one per landed trajectory: the verbatim prompt plus the
    per-trajectory constants the protocol needs."""
    path = REPO / "datasets" / TASK_SPEC[task]["dir"] / "trajectories.jsonl"
    seeds = {}
    for line in path.open():
        r = json.loads(line)
        if int(r["turn"]) == 1:
            seeds[r["trajectory_id"]] = r
    return list(seeds.values())


def build_messages(seed_row: dict, history: list[tuple[str, str]]) -> list[dict]:
    """Turn 1 verbatim; later turns append (assistant, user) pairs -- leg 1."""
    messages = [dict(m) for m in seed_row["prompt_messages"]]
    for assistant_text, feedback in history:
        messages.append({"role": "assistant", "content": assistant_text})
        messages.append({"role": "user", "content": feedback})
    return messages


def score_row(task: str, seed_row: dict, text: str) -> dict:
    spec = TASK_SPEC[task]
    row = {k: seed_row.get(k) for k in CARRIED}
    row.update(spec["measure"](text))
    for out_col, (src, fn) in spec["norm"].items():
        value = row.get(src)
        row[out_col] = None if value is None else fn(value)
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", required=True, choices=sorted(TASK_SPEC))
    ap.add_argument("--agent-model", required=True)
    ap.add_argument("--out-dir", type=pathlib.Path, required=True)
    ap.add_argument("--max-new-tokens", type=int, default=None,
                    help="default: the landed decoding_config's value for this task")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--limit-trajectories", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="build every prompt and exercise scorer+feedback on the "
                         "landed turn-1 text; load no model and write no file")
    args = ap.parse_args()

    spec = TASK_SPEC[args.task]
    seeds = load_seed_rows(args.task)
    if args.limit_trajectories:
        seeds = seeds[: args.limit_trajectories]
    decoding = seeds[0]["decoding_config"]
    max_new = args.max_new_tokens or int(decoding["max_new_tokens"])
    builder = BUILDERS[args.task]
    n_gen = len(seeds) * spec["turns"]

    print(f"[{args.task}] {len(seeds)} trajectories x {spec['turns']} turns = {n_gen} generations")
    print(f"[{args.task}] decoding from landed rows: {decoding} -> max_new_tokens={max_new}")

    if args.dry_run:
        bad = 0
        for seed_row in seeds:
            msgs = build_messages(seed_row, [])
            assert msgs == list(seed_row["prompt_messages"]), "turn-1 replay must be verbatim"
            row = score_row(args.task, seed_row, seed_row["raw_generation"])
            try:
                builder({**seed_row, **row})
            except NotImplementedError as exc:
                bad += 1
                if bad <= 3:
                    print(f"  unwitnessed feedback branch on {seed_row['trajectory_id']}: {exc}")
        print(f"[dry-run] {len(seeds)} turn-1 prompts replay verbatim; scorer+feedback ran on "
              f"every one; {bad} hit an unwitnessed branch; no model loaded, no file written")
        return

    out_path = args.out_dir / "trajectories.jsonl"
    if out_path.exists():
        raise FileExistsError(f"{out_path} exists; outputs are never overwritten")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    from vllm import LLM, SamplingParams  # imported late so --dry-run needs no GPU stack

    llm = LLM(model=args.agent_model, dtype="bfloat16",
              gpu_memory_utilization=args.gpu_memory_utilization,
              max_model_len=args.max_model_len)
    sampling = SamplingParams(
        temperature=float(decoding["temperature"]), top_p=float(decoding["top_p"]),
        max_tokens=max_new)

    histories: dict[str, list[tuple[str, str]]] = {r["trajectory_id"]: [] for r in seeds}
    written = []
    for turn in range(1, spec["turns"] + 1):
        prompts = [build_messages(r, histories[r["trajectory_id"]]) for r in seeds]
        outputs = llm.chat(prompts, sampling)
        for seed_row, out in zip(seeds, outputs):
            text = out.outputs[0].text.strip()
            row = score_row(args.task, seed_row, text)
            row.update({"task": args.task, "turn": turn, "model": args.agent_model,
                        "raw_generation": text, "decoding_config": dict(decoding),
                        "prompt_messages": build_messages(seed_row, histories[seed_row["trajectory_id"]]),
                        "cap_hit": len(out.outputs[0].token_ids) >= max_new})
            row["feedback_text"] = builder({**seed_row, **row}) if turn < spec["turns"] else None
            histories[seed_row["trajectory_id"]].append((text, row["feedback_text"] or ""))
            written.append(row)
        hits = sum(1 for r in written if r["turn"] == turn and r["cap_hit"])
        print(f"  turn {turn}/{spec['turns']}: {len(seeds)} rows, cap hit {hits}")

    with out_path.open("w") as fh:
        for row in written:
            fh.write(json.dumps(row, default=float) + "\n")
    cap_share = sum(1 for r in written if r["cap_hit"]) / len(written)
    (args.out_dir / "run_config.json").write_text(json.dumps({
        "task": args.task, "agent_model": args.agent_model, "n_trajectories": len(seeds),
        "turns": spec["turns"], "max_new_tokens": max_new, "decoding": decoding,
        "cap_hit_share": cap_share, "provenance": provenance()}, indent=2) + "\n")
    print(f"[done] {len(written)} rows -> {out_path}; cap hit share {cap_share:.4f}")


if __name__ == "__main__":
    main()
