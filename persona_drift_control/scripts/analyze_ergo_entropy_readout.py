#!/usr/bin/env python3
"""NOTE (2026-09-07): the E0 verdict cited below was WITHDRAWN. E0's RC-3
controlled for u_{t+2} instead of u_{t+1}, and its RC-1 made the model
extrapolate a deterministic exogenous variable that the null baseline received
as ground truth; corrected, `y_task_success` passes all of RC-1/RC-2/RC-3. See
docs/experiments/signal_resolution_plan.md sections 0.1/0.2 and task F0. The
entropy readout this script computes is still not usable, but for a different
reason (RC-0: it is near-orthogonal to the evaluated objective).

E1 Step 1 (docs/experiments/backup/readout_controllability_gate_plan.md section
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
from persona_drift.ergo_math_trajectory import _initial_agent_history  # noqa: E402
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
    # 2026-09-08: the replay below rebuilds the agent-facing history from the
    # recorded rows, and under prompt_profile="upstream" that history STARTS
    # with a system message (ergo_math_trajectory._initial_agent_history).
    # Without this flag the replay silently drops it and the entropies are
    # read off a context the model never saw. Default is "legacy" -- the
    # existing behavior byte-for-byte, so the already-published
    # outputs/ergo_math_phaseB_random_excite/entropy_readout.json stays
    # reproducible from the bare command line.
    parser.add_argument("--prompt-profile", choices=("legacy", "upstream"), default="legacy")
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


def group_trajectories(rows: list[dict]) -> list[list[dict]]:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["trajectory_id"], []).append(row)
    for traj_rows in groups.values():
        traj_rows.sort(key=lambda r: r["turn"])
    return list(groups.values())


def replay_histories(traj_rows: list[dict], prompt_profile: str):
    """Rebuild the agent-facing history each recorded turn actually saw, and
    yield (row, messages) in turn order.

    Two guards, because both failure modes are invisible in the output --
    the entropies would just be read off the wrong context (.claude/global.md:
    a guard that passes silently hides the thing it exists to catch):

    * the rows' own `prompt_profile` must match the requested one, or the
      replay starts from the wrong initial history;
    * a reset is replayed with OVERWRITE semantics (history replaced by the
      consolidated turn). Rows recorded under `reset_mode="append"` keep the
      pre-reset history, so replaying those here would be wrong -- refuse.
    """

    for row in traj_rows:
        recorded = row.get("prompt_profile")
        if recorded is not None and recorded != prompt_profile:
            raise SystemExit(
                f"row {row['trajectory_id']} turn {row['turn']} was recorded under "
                f"prompt_profile={recorded!r} but --prompt-profile is {prompt_profile!r}"
            )
        if row["u_reset"] and row.get("reset_mode") == "append":
            raise SystemExit(
                f"row {row['trajectory_id']} turn {row['turn']} is a reset recorded under "
                'reset_mode="append"; this replay only implements overwrite semantics'
            )

    agent_history = _initial_agent_history(prompt_profile)
    for row in traj_rows:
        if row["u_reset"]:
            agent_history = [*_initial_agent_history(prompt_profile),
                             {"role": "user", "content": row["user_message"]}]
        else:
            agent_history.append({"role": "user", "content": row["user_message"]})
        yield row, agent_history
        agent_history = [*agent_history, {"role": "assistant", "content": row["agent_message"]}]


def main() -> None:
    args = parse_args()
    rows = load_trajectories(args.rows_path)
    model = ChatModel(args.agent_model, device=args.device, enable_thinking=False)

    entropy_by_row: dict[tuple[str, int], dict] = {}
    n_missing_span = 0
    for traj_rows in group_trajectories(rows):
        for row, agent_history in replay_histories(traj_rows, args.prompt_profile):
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
        "prompt_profile": args.prompt_profile,
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
