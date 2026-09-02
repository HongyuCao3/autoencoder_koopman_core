"""One Phase F helpfulness-cost trajectory: a fixed, pre-generated benign
multi-turn session (benign_bank.BenignEntry) is replayed turn by turn against
the agent, with the exact same reminder-insertion mechanism
attack_trajectory.run_attack_trajectory uses (same controller.py Controller
protocol, same safety_reminder.py text, same reminder.count_inserted_tokens
-- inherited here via trajectory_runner.run_reminder_gated_trajectory, the
turn-loop shared with attack_trajectory.py, see that module's docstring).

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
from .control import Controller
from .helpfulness_judge import judge_helpfulness_score
from .safety_judge import judge_safety_score
from .safety_reminder import build_safety_reminder_text
from .trajectory_runner import JudgeCall, run_reminder_gated_trajectory

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


def _safety_judge_call(
    judge: ChatModel,
    entry: BenignEntry,
    rows: list[dict[str, Any]],
    turn: int,
    stimulus: str,
    response: str,
    seed: int,
    config: GenerationConfig,
) -> tuple[float, bool, str]:
    return judge_safety_score(judge, BENIGN_PLACEHOLDER_GOAL, stimulus, response, seed=seed, config=config)


def _helpfulness_judge_call(
    judge: ChatModel,
    entry: BenignEntry,
    rows: list[dict[str, Any]],
    turn: int,
    stimulus: str,
    response: str,
    seed: int,
    config: GenerationConfig,
) -> tuple[float, bool, str]:
    return judge_helpfulness_score(judge, stimulus, response, seed=seed, config=config)


def _extra_row_fields(entry: BenignEntry) -> dict[str, Any]:
    return {"benign_id": entry.benign_id, "category": entry.category}


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
    return run_reminder_gated_trajectory(
        agent=agent,
        judge=judge,
        entry=entry,
        seed=seed,
        trajectory_id=trajectory_id,
        multi_turn_stimuli=entry.multi_turn_queries,
        stimulus_field="question",
        reminder_fn=build_safety_reminder_text,
        judge_calls=[
            JudgeCall(
                score_field="y_safety",
                parse_failure_field="judge_parse_failure",
                raw_output_field="judge_raw_output",
                call=_safety_judge_call,
                config=config.safety_judge_gen,
                seed_offset=2,
            ),
            JudgeCall(
                score_field="y_help",
                parse_failure_field="helpfulness_parse_failure",
                raw_output_field="helpfulness_raw_output",
                call=_helpfulness_judge_call,
                config=config.helpfulness_judge_gen,
                seed_offset=3,
            ),
        ],
        primary_score_field="y_safety",
        extra_row_fields_fn=_extra_row_fields,
        agent_gen=config.agent_gen,
        run_id=run_id,
        controller=controller,
    )
