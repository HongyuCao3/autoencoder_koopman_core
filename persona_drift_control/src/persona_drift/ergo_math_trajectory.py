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

# docs/experiments/ergo_fidelity_restoration_plan.md section 3.1: under
# prompt_profile="upstream" the answer-format requirement lives here, delivered
# once before turn 1, instead of being concatenated onto every user turn. That
# is upstream's shape (microsoft/lost_in_conversation loads
# prompts/math/math_system_prompt.txt once via generate_system_prompt()), and
# the per-turn repetition is what section 0.1 attributes the append-mode
# terseness collapse to: the model reads k prior turns each carrying the
# instruction plus its own matching one-line reply, which is a k-shot
# demonstration of answering without deriving.
#
# The parseability requirement is KEPT (the regex judge is the only scoring
# instrument this harness has, so dropping it would leave R1 with no metric);
# it just appears once rather than every turn. The framing deliberately does
# NOT tell the assistant it is in a multi-turn underspecified conversation --
# upstream is explicit that "the assistant is not explicitly informed that it
# is participating in a multi-turn, underspecified conversation", and telling
# it would be a second changed variable.
_UPSTREAM_SYSTEM_PROMPT = (
    "You are a helpful assistant that solves math problems.\n\n"
    "The answer should be a single number (it could be decimal, or negative, or a fraction, "
    "etc.).\n\n"
    "End every reply with your current best-guess final numeric answer on its own line in this "
    'exact format: "Current answer: X" (a single number). Give it even if you are not fully '
    "confident yet or do not have all the details."
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
    # docs/experiments/ergo_fidelity_restoration_plan.md section 3.1: "legacy"
    # is the existing behavior byte-for-byte (no system message; the answer
    # format instruction concatenated onto every user turn); "upstream" moves
    # that instruction into a single system message and leaves the per-turn
    # stimuli as the bare shard / bare bullet list.
    prompt_profile: str = "legacy"

    def __post_init__(self) -> None:
        if self.reset_mode not in ("overwrite", "append"):
            raise ValueError(f"reset_mode must be 'overwrite' or 'append', got {self.reset_mode!r}")
        if self.prompt_profile not in ("legacy", "upstream"):
            raise ValueError(
                f"prompt_profile must be 'legacy' or 'upstream', got {self.prompt_profile!r}"
            )


def _incremental_stimulus(shard: str, prompt_profile: str = "legacy") -> str:
    if prompt_profile == "legacy":
        return f"{shard}\n\n{_ANSWER_FORMAT_INSTRUCTION}"
    return shard


def _consolidated_stimulus(revealed_shards: list[str], prompt_profile: str = "legacy") -> str:
    bullet_list = "\n".join(f"- {shard}" for shard in revealed_shards)
    head = (
        "Here is the math problem, given as a list of clues (all the information you have "
        f"been given so far in this conversation):\n{bullet_list}"
    )
    if prompt_profile == "legacy":
        return f"{head}\n\n{_ANSWER_FORMAT_INSTRUCTION}"
    return head


def _initial_agent_history(prompt_profile: str) -> list[dict[str, str]]:
    """The history an "empty" conversation starts from, and the base a
    reset_mode="overwrite" reset rebuilds on top of. Under "legacy" that is
    the empty list (existing behavior). Under "upstream" it is the single
    system message -- which an overwrite reset must NOT discard, or the
    system-prompt profile would silently revert to legacy from the first
    reset onward."""

    if prompt_profile == "legacy":
        return []
    return [{"role": "system", "content": _UPSTREAM_SYSTEM_PROMPT}]


def _run_one_turn(
    *,
    agent: ChatModel,
    judge: ChatModel,
    entry: GSM8KShardedItem,
    config: ErgoMathTrajectoryConfig,
    controller_name: str,
    run_id: str,
    trajectory_id: str,
    seed: int,
    turn: int,
    num_turns: int,
    agent_history: list[dict[str, str]],
    revealed_shards: list[str],
    u_reset: int,
    agent_seed: int,
    judge_seed: int,
    prior_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """One turn under action `u_reset`, starting from `agent_history`.

    Does NOT mutate `agent_history` -- it returns the continuation. That is
    what lets run_ergo_math_branch_trajectory run two actions from the SAME
    prefix (docs/experiments/ergo_fidelity_restoration_plan.md section 4).
    Extracted verbatim from run_ergo_math_trajectory's loop body; the
    byte-for-byte regression is tests/test_ergo_math_trajectory.py's
    test_branch_base_rows_are_identical_to_a_plain_run.
    """

    if u_reset:
        stimulus = _consolidated_stimulus(revealed_shards, config.prompt_profile)
        if config.reset_mode == "overwrite":
            history = _initial_agent_history(config.prompt_profile)
        else:  # append
            history = list(agent_history)
    else:
        stimulus = _incremental_stimulus(entry.shards[turn - 1], config.prompt_profile)
        history = list(agent_history)
    history.append({"role": "user", "content": stimulus})

    agent_text, agent_thinking = agent.generate(
        history, seed=agent_seed, config=config.agent_gen, return_thinking=True
    )
    history.append({"role": "assistant", "content": agent_text})

    score, parse_failure, raw_output = judge_math_answer(
        judge, entry, prior_rows, turn, stimulus, agent_text, judge_seed, config.agent_gen
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
        "prompt_profile": config.prompt_profile,
        "excitation_design": controller_name,
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
    return row, history


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

    agent_history: list[dict[str, str]] = _initial_agent_history(config.prompt_profile)
    rows: list[dict[str, Any]] = []
    revealed_shards: list[str] = []

    for turn in range(1, num_turns + 1):
        u_reset = controller.next_u_remind(turn, rows)
        revealed_shards.append(entry.shards[turn - 1])
        row, agent_history = _run_one_turn(
            agent=agent,
            judge=judge,
            entry=entry,
            config=config,
            controller_name=controller.name,
            run_id=run_id,
            trajectory_id=trajectory_id,
            seed=seed,
            turn=turn,
            num_turns=num_turns,
            agent_history=agent_history,
            revealed_shards=revealed_shards,
            u_reset=u_reset,
            agent_seed=seed * 1_000_000 + turn * 100 + 1,
            judge_seed=seed * 1_000_000 + turn * 100 + 2,
            prior_rows=rows,
        )
        rows.append(row)

    return rows


def run_ergo_math_branch_trajectory(
    agent: ChatModel,
    judge: ChatModel,
    entry: GSM8KShardedItem,
    seed: int,
    trajectory_id: str,
    config: ErgoMathTrajectoryConfig | None = None,
    run_id: str = "ergo_math_screening_v0.1",
    controller: Controller | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """EK-A (docs/experiments/ergo_fidelity_restoration_plan.md section 4):
    a base trajectory plus, at EVERY turn, one counterfactual turn generated
    from the byte-identical prefix under the opposite action.

    Why this and not another observational excitation arm: EK0 showed the
    i.i.d. `random_excite p=0.5` arm's minimum detectable endpoint effect is
    +0.49 while the effect R1 measured is +0.181, so it rules out nothing.
    A same-prefix pair measures the one-step causal gain directly instead of
    asking a regression to control for the prefix.

    The counterfactual reuses the base turn's `agent_seed` on purpose --
    common random numbers, so the pair shares its sampling draws and the
    difference is attributable to the action rather than to decoding noise.
    The judge gets a distinct seed (+3) so the two calls stay distinguishable.

    Returns `(base_rows, counterfactual_rows)`. They are written to SEPARATE
    files by the runner: counterfactual rows carry their own `trajectory_id`
    and are single-turn, so letting them into `trajectories.jsonl` would make
    every existing analyzer that groups by `trajectory_id` and selects
    `turn == num_shards` read them as one-row trajectories.
    """

    config = config or ErgoMathTrajectoryConfig()
    controller = controller or ZeroControlController()
    num_turns = len(entry.shards)

    agent_history: list[dict[str, str]] = _initial_agent_history(config.prompt_profile)
    base_rows: list[dict[str, Any]] = []
    counterfactual_rows: list[dict[str, Any]] = []
    revealed_shards: list[str] = []

    for turn in range(1, num_turns + 1):
        u_base = controller.next_u_remind(turn, base_rows)
        revealed_shards.append(entry.shards[turn - 1])
        prefix = agent_history
        shared = dict(
            agent=agent,
            judge=judge,
            entry=entry,
            config=config,
            controller_name=controller.name,
            run_id=run_id,
            seed=seed,
            turn=turn,
            num_turns=num_turns,
            agent_history=prefix,
            revealed_shards=revealed_shards,
            agent_seed=seed * 1_000_000 + turn * 100 + 1,
            prior_rows=base_rows,
        )
        base_row, agent_history = _run_one_turn(
            trajectory_id=trajectory_id,
            u_reset=u_base,
            judge_seed=seed * 1_000_000 + turn * 100 + 2,
            **shared,
        )
        cf_row, _ = _run_one_turn(
            trajectory_id=f"{trajectory_id}#cf_t{turn}",
            u_reset=1 - u_base,
            judge_seed=seed * 1_000_000 + turn * 100 + 3,
            **shared,
        )
        cf_row["is_counterfactual"] = True
        cf_row["base_trajectory_id"] = trajectory_id
        cf_row["branch_turn"] = turn
        cf_row["base_u_reset"] = u_base
        base_rows.append(base_row)
        counterfactual_rows.append(cf_row)

    return base_rows, counterfactual_rows
