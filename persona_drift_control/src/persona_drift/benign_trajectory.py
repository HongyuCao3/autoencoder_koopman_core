"""One Phase F helpfulness-cost trajectory: a fixed, pre-generated benign
multi-turn session (benign_bank.BenignEntry) is replayed turn by turn against
the agent, with the exact same reminder-insertion mechanism
attack_trajectory.run_attack_trajectory uses (same controller.py Controller
protocol, same safety_reminder.py text, same reminder.count_inserted_tokens).

Two independent judge calls happen each turn, not one:

1. safety_judge.judge_safety_score, against a fixed BENIGN_PLACEHOLDER_GOAL --
   this is NOT a data-leak or a mistake: ThresholdController/KoopmanMPCController
   read history[-1]["y_probe"], and in a real deployment the safety monitor
   that drives them runs on ALL traffic, not just traffic already known to be
   an attack (the system has no such foreknowledge). Keeping this judge call
   is what makes this check "deploy Phase E's actual fielded controllers
   against benign traffic" rather than a different, easier experiment.
2. helpfulness_judge.judge_helpfulness_score -- the actual Phase F outcome
   metric (y_help). Purely a readout: nothing downstream of it feeds back
   into controller decisions.

See docs/experiments/koopman_defense_pilot.md for why safety_judge's own
score can't double as the outcome metric here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .benign_bank import BenignEntry
from .chat_model import ChatModel, GenerationConfig
from .control import Controller, ZeroControlController
from .helpfulness_judge import judge_helpfulness_score
from .reminder import count_inserted_tokens
from .safety_judge import judge_safety_score, looks_like_refusal
from .safety_reminder import build_safety_reminder_text

BENIGN_PLACEHOLDER_GOAL = "N/A: this is a benign conversation, no adversarial goal"


@dataclass(frozen=True)
class BenignTrajectoryConfig:
    agent_gen: GenerationConfig = field(default_factory=lambda: GenerationConfig(max_new_tokens=256))
    safety_judge_gen: GenerationConfig = field(
        default_factory=lambda: GenerationConfig(max_new_tokens=10, temperature=0.0, do_sample=False)
    )
    helpfulness_judge_gen: GenerationConfig = field(
        default_factory=lambda: GenerationConfig(max_new_tokens=10, temperature=0.0, do_sample=False)
    )


def run_benign_trajectory(
    agent: ChatModel,
    judge: ChatModel,
    entry: BenignEntry,
    seed: int,
    trajectory_id: str,
    config: BenignTrajectoryConfig | None = None,
    run_id: str = "benign_screening_v0.1",
    controller: Controller | None = None,
) -> list[dict[str, Any]]:
    config = config or BenignTrajectoryConfig()
    controller = controller or ZeroControlController()
    num_turns = len(entry.multi_turn_queries)

    agent_history: list[dict[str, str]] = []
    rows: list[dict[str, Any]] = []

    for turn in range(1, num_turns + 1):
        u_remind = controller.next_u_remind(turn, rows)
        reminder_text = build_safety_reminder_text(u_remind)

        question = entry.multi_turn_queries[turn - 1]
        agent_facing_text = f"{reminder_text}\n{question}" if reminder_text else question
        agent_history.append({"role": "user", "content": agent_facing_text})

        agent_seed = seed * 1_000_000 + turn * 100 + 1
        agent_text, agent_thinking = agent.generate(
            agent_history, seed=agent_seed, config=config.agent_gen, return_thinking=True
        )
        agent_history.append({"role": "assistant", "content": agent_text})

        safety_seed = seed * 1_000_000 + turn * 100 + 2
        y_safety, safety_parse_failure, safety_raw_output = judge_safety_score(
            judge, BENIGN_PLACEHOLDER_GOAL, question, agent_text, seed=safety_seed, config=config.safety_judge_gen
        )

        help_seed = seed * 1_000_000 + turn * 100 + 3
        y_help, help_parse_failure, help_raw_output = judge_helpfulness_score(
            judge, question, agent_text, seed=help_seed, config=config.helpfulness_judge_gen
        )

        rows.append(
            {
                "trajectory_id": trajectory_id,
                "benign_id": entry.benign_id,
                "category": entry.category,
                "turn": turn,
                "question": question,
                "agent_message": agent_text,
                "agent_thinking": agent_thinking,
                "y_safety": y_safety,
                # Same y_probe alias attack_trajectory.py writes, for the
                # exact same reason: lets ThresholdController/
                # KoopmanMPCController run unmodified on this domain too.
                "y_probe": y_safety,
                "y_help": y_help,
                "u_remind": u_remind,
                "excitation_design": controller.name,
                "inserted_reminder_text": reminder_text,
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
                "judge_parse_failure": safety_parse_failure,
                "judge_raw_output": safety_raw_output,
                "helpfulness_parse_failure": help_parse_failure,
                "helpfulness_raw_output": help_raw_output,
            }
        )

    return rows
