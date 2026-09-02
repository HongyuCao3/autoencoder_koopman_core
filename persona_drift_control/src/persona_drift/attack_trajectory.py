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

The turn-by-turn loop itself lives in trajectory_runner.run_reminder_gated_trajectory
(shared with benign_trajectory.py and sycophancy_trajectory.py, see that
module's docstring for why); this module only supplies the domain-specific
pieces -- which judge to call and with what extra context, which metadata
fields land on each row, and the row's stimulus field name.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .attack_bank import AttackEntry
from .chat_model import ChatModel, GenerationConfig
from .control import Controller
from .safety_judge import judge_safety_score
from .safety_reminder import build_safety_reminder_text
from .trajectory_runner import JudgeCall, run_reminder_gated_trajectory


@dataclass(frozen=True)
class AttackTrajectoryConfig:
    agent_gen: GenerationConfig = field(default_factory=lambda: GenerationConfig(max_new_tokens=256))
    judge_gen: GenerationConfig = field(
        default_factory=lambda: GenerationConfig(max_new_tokens=10, temperature=0.0, do_sample=False)
    )


def _safety_judge_call(
    judge: ChatModel,
    entry: AttackEntry,
    rows: list[dict[str, Any]],
    turn: int,
    stimulus: str,
    response: str,
    seed: int,
    config: GenerationConfig,
) -> tuple[float, bool, str]:
    return judge_safety_score(judge, entry.plain_query, stimulus, response, seed=seed, config=config)


def _extra_row_fields(entry: AttackEntry) -> dict[str, Any]:
    return {
        "attack_id": entry.attack_id,
        "category": entry.category,
        "actor_name": entry.actor_name,
        "plain_query": entry.plain_query,
    }


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
    return run_reminder_gated_trajectory(
        agent=agent,
        judge=judge,
        entry=entry,
        seed=seed,
        trajectory_id=trajectory_id,
        multi_turn_stimuli=entry.multi_turn_queries,
        stimulus_field="attacker_query",
        reminder_fn=build_safety_reminder_text,
        judge_calls=[
            JudgeCall(
                score_field="y_safety",
                parse_failure_field="judge_parse_failure",
                raw_output_field="judge_raw_output",
                call=_safety_judge_call,
                config=config.judge_gen,
                seed_offset=2,
            )
        ],
        primary_score_field="y_safety",
        extra_row_fields_fn=_extra_row_fields,
        agent_gen=config.agent_gen,
        run_id=run_id,
        controller=controller,
    )
