"""One ERGO/Laban sharded-GSM8K trajectory: a math item's clauses
(ergo_math_bank.GSM8KShardedItem.shards) are revealed one per turn on this
project's usual fixed schedule (shard `i` at turn `i` -- resources/
PROVENANCE.md's "known simplification" note explains why this project does
not replicate upstream's live user-simulator reveal/classification logic).
Every turn the agent is asked for its current best-guess numeric answer,
scored by ergo_math_judge.judge_math_answer against the item's gold answer
-- not just the upstream benchmark's single end-of-conversation grade -- so
this produces a per-turn `y_t` compatible with this project's other
trajectory conventions (docs/feasibility/
ERGO_MULTITURN_RELIABILITY_FEASIBILITY.md section 2.3's "path B", chosen
over the entropy-proxy "path A" because GSM8K's evaluator is cheap enough
to make path B free for this one task).

This does NOT reuse trajectory_runner.run_reminder_gated_trajectory: that
shared loop's `reminder_fn(level) -> str | None` only ever prefixes fixed
text onto the current turn's stimulus, and every existing caller's action
is exactly that shape (a constant sentence, independent of turn/history).
The "reset" actuator this module tests is a different kind of action --
it replaces the accumulated conversation history with a single fresh
message consolidating every shard revealed so far, which depends on how
many turns have happened and cannot be expressed as `reminder_fn(level)`
without changing that shared function's signature for every existing
caller. Reusing `control.Controller` (zero_control/constant_remind/
periodic/fixed_schedule/threshold all already implement
`next_u_remind(turn, history) -> int` and `.name`) for the reset/no-reset
decision costs nothing, though -- only the turn loop itself is new here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .chat_model import ChatModel, GenerationConfig
from .control import Controller, ZeroControlController
from .ergo_math_bank import GSM8KShardedItem
from .ergo_math_judge import _normalize, extract_answer_by_regex, judge_math_answer
from .reminder import count_inserted_tokens
from .safety_judge import looks_like_refusal

_ANSWER_FORMAT_INSTRUCTION = (
    'Give your current best-guess final numeric answer to the math problem, even if you are '
    'not fully confident yet or do not have all the details, on its own line in this exact '
    'format: "Current answer: X" (a single number).'
)


def _to_number(text: str | None) -> float | None:
    if text is None:
        return None
    try:
        return float(_normalize(text))
    except (ValueError, TypeError):
        return None


def closeness(agent_message: str, gold_answer: str) -> float:
    """docs/experiments/signal_resolution_plan.md section 2.2's
    de-thresholded state readout: the SAME extractor and gold answer
    `ergo_math_judge.judge_math_answer` already compares, just not chopped
    to 0/1 at exact equality. 0.0 on any parse failure or non-numeric
    extraction, matching the hard score's parse-failure convention."""

    extracted = extract_answer_by_regex(agent_message)
    a = _to_number(extracted) if extracted is not None else None
    g = _to_number(gold_answer)
    if a is None or g is None:
        return 0.0
    return 1.0 / (1.0 + abs(a - g) / max(abs(g), 1.0))


@dataclass(frozen=True)
class ErgoMathTrajectoryConfig:
    agent_gen: GenerationConfig = field(default_factory=lambda: GenerationConfig(max_new_tokens=512))
    # docs/experiments/two_task_success_plan.md section 2 E1: "overwrite" is
    # the existing behavior (a reset replaces the whole accumulated history
    # with one consolidated message) and must stay byte-for-byte identical;
    # "append" keeps the accumulated history and appends the consolidated
    # message on top of it, so the pre-reset history re-enters the agent's
    # context instead of being discarded.
    reset_mode: str = "overwrite"

    def __post_init__(self) -> None:
        if self.reset_mode not in ("overwrite", "append"):
            raise ValueError(f"reset_mode must be 'overwrite' or 'append', got {self.reset_mode!r}")


def _incremental_stimulus(shard: str) -> str:
    return f"{shard}\n\n{_ANSWER_FORMAT_INSTRUCTION}"


def _consolidated_stimulus(revealed_shards: list[str]) -> str:
    bullet_list = "\n".join(f"- {shard}" for shard in revealed_shards)
    return (
        "Here is the math problem, given as a list of clues (all the information you have "
        f"been given so far in this conversation):\n{bullet_list}\n\n{_ANSWER_FORMAT_INSTRUCTION}"
    )


def run_ergo_math_trajectory(
    agent: ChatModel,
    judge: ChatModel,
    entry: GSM8KShardedItem,
    seed: int,
    trajectory_id: str,
    config: ErgoMathTrajectoryConfig | None = None,
    run_id: str = "ergo_math_screening_v0.1",
    controller: Controller | None = None,
) -> list[dict[str, Any]]:
    config = config or ErgoMathTrajectoryConfig()
    controller = controller or ZeroControlController()
    num_turns = len(entry.shards)

    agent_history: list[dict[str, str]] = []
    rows: list[dict[str, Any]] = []
    revealed_shards: list[str] = []

    for turn in range(1, num_turns + 1):
        u_reset = controller.next_u_remind(turn, rows)
        revealed_shards.append(entry.shards[turn - 1])

        if u_reset:
            stimulus = _consolidated_stimulus(revealed_shards)
            if config.reset_mode == "overwrite":
                agent_history = [{"role": "user", "content": stimulus}]
            else:  # append
                agent_history.append({"role": "user", "content": stimulus})
        else:
            stimulus = _incremental_stimulus(entry.shards[turn - 1])
            agent_history.append({"role": "user", "content": stimulus})

        agent_seed = seed * 1_000_000 + turn * 100 + 1
        agent_text, agent_thinking = agent.generate(
            agent_history, seed=agent_seed, config=config.agent_gen, return_thinking=True
        )
        agent_history.append({"role": "assistant", "content": agent_text})

        score, parse_failure, raw_output = judge_math_answer(
            judge, entry, rows, turn, stimulus, agent_text, seed * 1_000_000 + turn * 100 + 2, config.agent_gen
        )

        row: dict[str, Any] = {
            "trajectory_id": trajectory_id,
            "item_id": entry.item_id,
            "gold_answer": entry.gold_answer,
            "turn": turn,
            "num_shards": num_turns,
            "shard_text": entry.shards[turn - 1],
            "user_message": stimulus,
            "agent_message": agent_text,
            "agent_thinking": agent_thinking,
            "u_reset": u_reset,
            "reset_mode": config.reset_mode,
            "excitation_design": controller.name,
            "run_id": run_id,
            "seed": seed,
            "model": agent.model_id,
            "decoding_config": {
                "temperature": config.agent_gen.temperature,
                "top_p": config.agent_gen.top_p,
                "max_new_tokens": config.agent_gen.max_new_tokens,
            },
            "refusal_flag": looks_like_refusal(agent_text),
            "inserted_tokens": count_inserted_tokens(agent.tokenizer, stimulus) if u_reset else 0,
            "y_task_success": score,
            "judge_parse_failure": parse_failure,
            "judge_raw_output": raw_output,
        }
        row["y_probe"] = row["y_task_success"]
        row["closeness"] = closeness(agent_text, entry.gold_answer)
        rows.append(row)

    return rows
