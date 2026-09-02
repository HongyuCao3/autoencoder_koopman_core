"""Shared per-turn trajectory loop: the control-flow core of
attack_trajectory.run_attack_trajectory and benign_trajectory.run_benign_trajectory
was, before this module existed, two independently-maintained ~80-line copies
that differed only in (a) which judge(s) get called and with what extra
context, (b) which extra metadata fields land on each row, and (c) what the
row's "stimulus" field is named (attacker_query vs question). Everything
else -- building the reminder-prefixed agent-facing text, the agent
generation call and its seed derivation, the y_probe alias, refusal
detection, run/seed/model/decoding-config bookkeeping -- was character-for-
character identical. A third near-identical copy was about to be added for
the sycophancy-drift domain (see docs/task/SYCOPHANCY_DRIFT_TASK_FEASIBILITY.md),
which is the concrete "rule of three" trigger for extracting this: one tested
turn-loop shared by every domain means a bug (or the v-alignment timing
bug's cousin) gets fixed once for all domains instead of N times, which is
what "code reuse for reproducibility" concretely buys here -- see
group_stats.py and modeling/dataset.py's optional-column-name params for the
same kind of extraction done previously in this codebase, once, after real
duplication (not preemptively).

What is deliberately NOT pulled in here, and stays in each domain module
(attack_trajectory.py / benign_trajectory.py / sycophancy_trajectory.py):
judge rubrics and prompts, reminder text content, what counts as an
"entry"/bank, and the row's domain-specific metadata fields. Those differ
enough in meaning (not just in column name) that folding them in here would
recreate the anti-pattern analysis_adversarial.py's docstring already
warns against for a different pair of modules -- threading extra branches
through one function's logic to serve semantically different callers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .chat_model import ChatModel, GenerationConfig
from .control import Controller, ZeroControlController
from .reminder import count_inserted_tokens
from .safety_judge import looks_like_refusal


@dataclass(frozen=True)
class JudgeCall:
    """One per-turn judge invocation. `call` is a domain-supplied closure
    with signature (judge, entry, rows_so_far, turn, stimulus_text,
    response_text, seed, config) -> (score, parse_failure, raw_output) --
    `entry` and `rows_so_far` are passed through unused by callers that
    don't need them (e.g. benign's safety judge, which only needs a fixed
    placeholder goal) and used by callers that do (e.g. a sycophancy judge
    reading rows_so_far[0]["agent_message"] for the turn-1 original stance,
    the same "read prior rows" pattern control.py's Controller protocol
    already uses for next_u_remind).

    `seed_offset` reproduces each domain's pre-extraction seed derivation
    exactly (attack/benign's safety judge used turn*100+2, benign's
    helpfulness judge used turn*100+3) so refactoring this out changes no
    seed ever passed to a judge call.
    """

    score_field: str
    parse_failure_field: str
    raw_output_field: str
    call: Callable[[ChatModel, Any, list[dict[str, Any]], int, str, str, int, GenerationConfig], tuple[float, bool, str]]
    config: GenerationConfig
    seed_offset: int


def run_reminder_gated_trajectory(
    agent: ChatModel,
    judge: ChatModel,
    entry: Any,
    seed: int,
    trajectory_id: str,
    multi_turn_stimuli: tuple[str, ...],
    stimulus_field: str,
    reminder_fn: Callable[[int], str | None],
    judge_calls: list[JudgeCall],
    primary_score_field: str,
    extra_row_fields_fn: Callable[[Any], dict[str, Any]],
    agent_gen: GenerationConfig,
    run_id: str,
    controller: Controller | None = None,
) -> list[dict[str, Any]]:
    """One trajectory: replay `multi_turn_stimuli` turn by turn against
    `agent`, optionally prefixing each turn with a controller-gated reminder
    (channel A), scoring every turn with every judge in `judge_calls`.
    `primary_score_field` names which judge's score field becomes the
    `y_probe` alias that every control.py Controller reads."""

    controller = controller or ZeroControlController()
    num_turns = len(multi_turn_stimuli)

    agent_history: list[dict[str, str]] = []
    rows: list[dict[str, Any]] = []

    for turn in range(1, num_turns + 1):
        u_remind = controller.next_u_remind(turn, rows)
        reminder_text = reminder_fn(u_remind)

        stimulus = multi_turn_stimuli[turn - 1]
        agent_facing_text = f"{reminder_text}\n{stimulus}" if reminder_text else stimulus
        agent_history.append({"role": "user", "content": agent_facing_text})

        agent_seed = seed * 1_000_000 + turn * 100 + 1
        agent_text, agent_thinking = agent.generate(
            agent_history, seed=agent_seed, config=agent_gen, return_thinking=True
        )
        agent_history.append({"role": "assistant", "content": agent_text})

        row: dict[str, Any] = {
            "trajectory_id": trajectory_id,
            **extra_row_fields_fn(entry),
            "turn": turn,
            stimulus_field: stimulus,
            "agent_message": agent_text,
            "agent_thinking": agent_thinking,
            "u_remind": u_remind,
            "excitation_design": controller.name,
            "inserted_reminder_text": reminder_text,
            "inserted_tokens": count_inserted_tokens(agent.tokenizer, reminder_text) if reminder_text else 0,
            "run_id": run_id,
            "seed": seed,
            "model": agent.model_id,
            "judge_model": judge.model_id,
            "decoding_config": {
                "temperature": agent_gen.temperature,
                "top_p": agent_gen.top_p,
                "max_new_tokens": agent_gen.max_new_tokens,
            },
            "refusal_flag": looks_like_refusal(agent_text),
        }

        for jc in judge_calls:
            judge_seed = seed * 1_000_000 + turn * 100 + jc.seed_offset
            score, parse_failure, raw_output = jc.call(
                judge, entry, rows, turn, stimulus, agent_text, judge_seed, jc.config
            )
            row[jc.score_field] = score
            row[jc.parse_failure_field] = parse_failure
            row[jc.raw_output_field] = raw_output

        row["y_probe"] = row[primary_score_field]
        rows.append(row)

    return rows
