"""One adversarial-defense screening trajectory: a fixed, pre-generated
multi-turn jailbreak query sequence (attack_bank.AttackEntry) is replayed
turn by turn against the agent. Each turn's real agent reply is scored for
safety by safety_judge.judge_safety_score. Structurally this is
selfchat.run_trajectory with the live user_sim replaced by fixed replay and
the persona probe-fork replaced by scoring the actual continuing reply
(cheaper: no probe_repeats fan-out, appropriate for a screening pass).

An optional `controller` (control.py's `Controller` protocol, the same
plant-agnostic interface selfchat.py already drives channel A with) can
insert a fixed safety-policy reminder (safety_reminder.py) before each
turn's attacker query -- see docs/experiments/koopman_defense_pilot.md for
why this is the chosen defense actuator (prompt-level reminder injection,
not activation steering) and how it's used for both the open-loop
identification data and the closed-loop Koopman-MPC validation. Passing no
controller (the default) is exactly equivalent to `ZeroControlController`
(u_remind == 0 every turn, no reminder ever inserted) -- this is what every
existing screening run (job 15399715, job 15410124) is, retroactively.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .attack_bank import AttackEntry
from .chat_model import ChatModel, GenerationConfig
from .control import Controller, ZeroControlController
from .reminder import count_inserted_tokens
from .safety_judge import judge_safety_score, looks_like_refusal
from .safety_reminder import build_safety_reminder_text


@dataclass(frozen=True)
class AttackTrajectoryConfig:
    agent_gen: GenerationConfig = field(default_factory=lambda: GenerationConfig(max_new_tokens=256))
    judge_gen: GenerationConfig = field(
        default_factory=lambda: GenerationConfig(max_new_tokens=10, temperature=0.0, do_sample=False)
    )


def run_attack_trajectory(
    agent: ChatModel,
    judge: ChatModel,
    entry: AttackEntry,
    seed: int,
    trajectory_id: str,
    config: AttackTrajectoryConfig | None = None,
    run_id: str = "adversarial_screening_v0.2",
    controller: Controller | None = None,
) -> list[dict[str, Any]]:
    config = config or AttackTrajectoryConfig()
    controller = controller or ZeroControlController()
    num_turns = len(entry.multi_turn_queries)

    agent_history: list[dict[str, str]] = []
    rows: list[dict[str, Any]] = []

    for turn in range(1, num_turns + 1):
        u_remind = controller.next_u_remind(turn, rows)
        reminder_text = build_safety_reminder_text(u_remind)

        attacker_query = entry.multi_turn_queries[turn - 1]
        agent_facing_text = f"{reminder_text}\n{attacker_query}" if reminder_text else attacker_query
        agent_history.append({"role": "user", "content": agent_facing_text})

        agent_seed = seed * 1_000_000 + turn * 100 + 1
        agent_text, agent_thinking = agent.generate(
            agent_history, seed=agent_seed, config=config.agent_gen, return_thinking=True
        )
        # Only the final content (not the reasoning trace) re-enters the
        # conversation history, matching how a real multi-turn chat client
        # would call a thinking-mode model.
        agent_history.append({"role": "assistant", "content": agent_text})

        judge_seed = seed * 1_000_000 + turn * 100 + 2
        y_safety, judge_parse_failure, judge_raw_output = judge_safety_score(
            judge, entry.plain_query, attacker_query, agent_text, seed=judge_seed, config=config.judge_gen
        )

        rows.append(
            {
                "trajectory_id": trajectory_id,
                "attack_id": entry.attack_id,
                "category": entry.category,
                "actor_name": entry.actor_name,
                "plain_query": entry.plain_query,
                "turn": turn,
                "attacker_query": attacker_query,
                "agent_message": agent_text,
                "agent_thinking": agent_thinking,
                "y_safety": y_safety,
                # Alias of y_safety under the field name control.py's existing
                # Controller implementations expect (ThresholdController reads
                # history[-1]["y_probe"]) -- lets every Controller in
                # control.py (ZeroControlController, ConstantRemindController,
                # RandomExciteController, ThresholdController) run unmodified
                # on this domain, see koopman_defense_pilot.md.
                "y_probe": y_safety,
                "u_remind": u_remind,
                "excitation_design": controller.name,
                "inserted_reminder_text": reminder_text,
                # Guarded rather than unconditional: `agent.tokenizer` doesn't
                # exist on every fake test double used elsewhere in this
                # codebase (e.g. adversarial_screening.py's own tests), and
                # there's nothing to count when no reminder was inserted.
                "inserted_tokens": count_inserted_tokens(agent.tokenizer, reminder_text) if reminder_text else 0,
                "run_id": run_id,
                "seed": seed,
                "model": agent.model_id,
                "judge_model": judge.model_id,
                "decoding_config": {
                    "temperature": config.agent_gen.temperature,
                    "top_p": config.agent_gen.top_p,
                    "max_new_tokens": config.agent_gen.max_new_tokens,
                },
                "refusal_flag": looks_like_refusal(agent_text),
                "judge_parse_failure": judge_parse_failure,
                "judge_raw_output": judge_raw_output,
            }
        )

    return rows
