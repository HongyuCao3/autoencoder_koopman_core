#!/usr/bin/env python3
"""E1 Step 1 (docs/experiments/readout_controllability_gate_plan.md section
5.1): a token-entropy readout for the ERGO/Laban line, tried because E0
(scripts/analyze_ergo_readout_state.py) found `y_task_success` fails RC-1
and RC-3 -- binary, sparse (13.7% success in the Phase B random-excitation
data), and its predictable part is almost entirely the deterministic
`shard_frac` ramp, not feedback-usable state.

For every one of the 666 rows in
outputs/ergo_math_phaseB_random_excite/trajectories.jsonl, rebuilds the
exact agent-facing chat history that turn actually saw (replaying each
trajectory's recorded `u_reset`/`user_message`/`agent_message` in turn
order -- NOT re-deriving stimulus text from the item bank, so this cannot
drift from what was actually shown to the model) and does one
teacher-forcing forward pass over that row's already-recorded
`agent_message`, reading off two per-row entropy columns:

- `entropy_mean`: mean predictive entropy (nats) over every generated
  content token.
- `entropy_answer_span`: mean predictive entropy over just the tokens at
  and after the literal "Current answer:" substring (the tokens that
  actually commit to a numeric answer) -- null when that substring isn't
  found in the row's `agent_message` (a parse-failure-adjacent row).

No re-sampling, no RNG: teacher forcing over an already-generated,
already-recorded string is a deterministic function of the frozen model
weights and that string, like `chat_model.ChatModel.next_token_logits`.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from persona_drift.chat_model import ChatModel, _build_prompt_text  # noqa: E402
from persona_drift.modeling.dataset import load_trajectories  # noqa: E402

ANSWER_MARKER = "Current answer:"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agent-model", default="Qwen/Qwen3-4B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--rows-path",
        type=pathlib.Path,
        default=pathlib.Path("outputs/ergo_math_phaseB_random_excite/trajectories.jsonl"),
    )
    parser.add_argument(
        "--out-path",
        type=pathlib.Path,
        default=pathlib.Path("outputs/ergo_math_phaseB_random_excite/entropy_readout.json"),
    )
    return parser.parse_args()


def _token_entropies(model: ChatModel, messages: list[dict[str, str]], completion_text: str) -> torch.Tensor:
    """Teacher-forcing forward pass: returns a 1D tensor of length
    len(completion token ids), entropy (nats) of the predictive
    distribution at each position that generated one of those tokens.
    Position convention matches `ChatModel.next_token_logits`: the logits
    at the last prompt-only position predict the first completion token."""

    prompt_text = _build_prompt_text(model.tokenizer, messages, enable_thinking=False)
    prompt_ids = model.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    completion_ids = model.tokenizer(completion_text, add_special_tokens=False)["input_ids"]
    if not completion_ids:
        return torch.zeros(0)

    full_ids = torch.tensor([prompt_ids + completion_ids], device=model.device)
    with torch.no_grad():
        logits = model.model(full_ids).logits[0].float()  # (seq_len, vocab)

    start = len(prompt_ids) - 1  # predicts completion_ids[0]
    end = start + len(completion_ids)  # predicts completion_ids[-1]
    completion_logits = logits[start:end]
    log_probs = torch.log_softmax(completion_logits, dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1)
    return entropy.cpu()


def _answer_span_entropy(model: ChatModel, completion_text: str, entropies: torch.Tensor) -> float | None:
    marker_pos = completion_text.find(ANSWER_MARKER)
    if marker_pos < 0 or entropies.numel() == 0:
        return None
    encoded = model.tokenizer(completion_text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoded["offset_mapping"]
    span_token_idx = [i for i, (s, e) in enumerate(offsets) if e > marker_pos]
    if not span_token_idx:
        return None
    span_start = span_token_idx[0]
    span_entropies = entropies[span_start:]
    if span_entropies.numel() == 0:
        return None
    return float(span_entropies.mean())


def main() -> None:
    args = parse_args()
    rows = load_trajectories(args.rows_path)
    model = ChatModel(args.agent_model, device=args.device, enable_thinking=False)

    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["trajectory_id"], []).append(row)
    for traj_rows in groups.values():
        traj_rows.sort(key=lambda r: r["turn"])

    entropy_by_row: dict[tuple[str, int], dict] = {}
    n_missing_span = 0
    for traj_rows in groups.values():
        agent_history: list[dict[str, str]] = []
        for row in traj_rows:
            if row["u_reset"]:
                agent_history = [{"role": "user", "content": row["user_message"]}]
            else:
                agent_history.append({"role": "user", "content": row["user_message"]})

            entropies = _token_entropies(model, agent_history, row["agent_message"])
            entropy_mean = float(entropies.mean()) if entropies.numel() > 0 else None
            entropy_answer_span = _answer_span_entropy(model, row["agent_message"], entropies)
            if entropy_answer_span is None:
                n_missing_span += 1
            entropy_by_row[(row["trajectory_id"], row["turn"])] = {
                "entropy_mean": entropy_mean,
                "entropy_answer_span": entropy_answer_span,
                "n_completion_tokens": int(entropies.numel()),
            }

            agent_history.append({"role": "assistant", "content": row["agent_message"]})

    out_rows = []
    for row in rows:
        readout = entropy_by_row[(row["trajectory_id"], row["turn"])]
        out_rows.append(
            {
                "trajectory_id": row["trajectory_id"],
                "item_id": row["item_id"],
                "turn": row["turn"],
                "num_shards": row["num_shards"],
                "u_reset": row["u_reset"],
                "y_task_success": row["y_task_success"],
                **readout,
            }
        )

    report = {
        "agent_model": args.agent_model,
        "rows_path": str(args.rows_path),
        "n_rows": len(out_rows),
        "n_missing_answer_span": n_missing_span,
        "missing_answer_span_rate": n_missing_span / len(out_rows),
        "rows": out_rows,
    }
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.out_path.exists():
        raise SystemExit(f"refusing to overwrite existing {args.out_path}")
    args.out_path.write_text(json.dumps(report, indent=2))

    print(f"n_rows={len(out_rows)}")
    print(f"n_missing_answer_span={n_missing_span} ({report['missing_answer_span_rate']*100:.2f}%)")
    print(f"report written to {args.out_path}")


if __name__ == "__main__":
    main()
