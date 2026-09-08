"""E2's C2 context guardrail (docs/experiments/two_task_success_plan.md
section 2 E2 / section 12.2 C2): append-mode prompts grow monotonically
because a reset appends the consolidated stimulus on top of the kept
history instead of replacing it. This script reconstructs, per trajectory
row, the exact token count of the prompt the agent actually saw at that
turn -- not a re-run, a read-only reconstruction from the trajectory log's
own `user_message`/`agent_message` fields -- and reports the maximum across
one or more arm directories, so a truncation confound (the defense line's
B1) can be ruled out cheaply before it is trusted anywhere downstream.

Reconstruction: `ergo_math_trajectory.run_ergo_math_trajectory` builds
`agent_history` as an alternating list of {"role": "user"/"assistant",
"content": ...} dicts, appending `user_message` then `agent_message` each
turn, and `chat_model._build_prompt_text` renders exactly that list
with `tokenizer.apply_chat_template(messages, tokenize=False,
add_generation_prompt=True, enable_thinking=enable_thinking)` before
`.generate()` ever sees it. `run_ergo_math_screening.py` never overrides
`enable_thinking`, so `ChatModel.__init__`'s default (False) applies --
consistent with every trajectory row's `agent_thinking` field being empty.
This script rebuilds the same messages list per turn (using rows 1..t-1's
user_message + agent_message, plus turn t's own user_message) and renders
it with the same tokenizer call, so the token count matches what the
agent's forward pass actually consumed, not a naive plain-text count.

VALID FOR reset_mode="append" ONLY. Under "overwrite" a reset replaces
`agent_history` with a single message (`ergo_math_trajectory.py:114`), so a
reconstruction that accumulates rows 1..t-1 systematically OVERSTATES the
prompt an overwrite-mode arm actually consumed -- it is an upper bound, not
the count. Pointing this script at `outputs/ergo_math_phaseC_*` therefore
produces a number that must not be quoted (the 2026-09-08 E2 session
measured 3282 tokens that way; it is an artifact). C2 gates append arms
only, so the gate itself is unaffected.

Does not modify ergo_math_trajectory.py or re-run any model -- CPU-only,
reads existing trajectories.jsonl files.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def _load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _group_by_trajectory(rows: list[dict]) -> dict[str, list[dict]]:
    by_traj: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_traj[row["trajectory_id"]].append(row)
    for traj_id, traj_rows in by_traj.items():
        traj_rows.sort(key=lambda r: r["turn"])
    return by_traj


def per_turn_prompt_token_counts(tokenizer, traj_rows: list[dict], enable_thinking: bool = False) -> list[int]:
    """Returns one token count per turn: the length of the exact rendered
    prompt (chat-template applied, add_generation_prompt=True) the agent
    saw immediately before generating that turn's agent_message."""

    messages: list[dict[str, str]] = []
    counts = []
    for row in traj_rows:
        messages.append({"role": "user", "content": row["user_message"]})
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
            )
        except TypeError:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        n_tokens = len(tokenizer(text, add_special_tokens=False)["input_ids"])
        counts.append(n_tokens)
        messages.append({"role": "assistant", "content": row["agent_message"]})
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm", action="append", required=True, help="name=path (path is a dir containing trajectories.jsonl)"
    )
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-4B")
    parser.add_argument("--context-limit", type=int, default=40960, help="model_config.max_position_embeddings")
    parser.add_argument("--output", default=None, help="write the JSON report here (refuses to overwrite)")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    report: dict = {
        "tokenizer": args.tokenizer,
        "context_limit_max_position_embeddings": args.context_limit,
        "arms": {},
    }

    for arm_spec in args.arm:
        name, path_str = arm_spec.split("=", 1)
        path = Path(path_str)
        rows = _load_rows(path / "trajectories.jsonl")
        by_traj = _group_by_trajectory(rows)

        max_tokens = -1
        max_traj_id = None
        max_turn = None
        n_trajectories = 0
        n_exceeding_limit = 0

        for traj_id, traj_rows in by_traj.items():
            n_trajectories += 1
            counts = per_turn_prompt_token_counts(tokenizer, traj_rows)
            for row, count in zip(traj_rows, counts):
                if count > args.context_limit:
                    n_exceeding_limit += 1
                if count > max_tokens:
                    max_tokens = count
                    max_traj_id = traj_id
                    max_turn = row["turn"]

        report["arms"][name] = {
            "directory": str(path),
            "n_trajectories": n_trajectories,
            "max_prompt_tokens": max_tokens,
            "max_at_trajectory_id": max_traj_id,
            "max_at_turn": max_turn,
            "n_turns_exceeding_context_limit": n_exceeding_limit,
        }
        print(
            f"{name}: max_prompt_tokens={max_tokens} "
            f"(trajectory={max_traj_id}, turn={max_turn}), "
            f"n_turns_exceeding_limit={n_exceeding_limit}"
        )

    if args.output:
        out_path = Path(args.output)
        if out_path.exists():
            raise FileExistsError(f"refusing to overwrite existing file: {out_path}")
        out_path.write_text(json.dumps(report, indent=2))
        print(f"\nreport written to {out_path}")


if __name__ == "__main__":
    main()
