"""One MMLU-sycophancy screening trajectory: a multiple-choice item
(mc_sycophancy_bank.MCQItem) is replayed turn by turn -- turn 1 is the bare
question (establishes the agent's initial answer), turns 2..5 replay the
fixed, generic escalating pushback script (mc_sycophancy_bank.PUSHBACK_TURNS).
Each turn's reply is scored by mc_answer_judge.judge_mc_answer against the
item's `correct_letter` (MMLU's own answer key), not against the agent's own
turn-1 answer or any free-text field a judge model has to adjudicate --
see mc_answer_judge.py's docstring for why that distinction is the point of
this module existing alongside sycophancy_trajectory.py rather than
replacing it in place.

No reminder/controller actuator exists for this domain yet (this module
only replaces the sycophancy-drift line's SYCON-Bench item pool with a
ground-truth-verified one, see mc_sycophancy_bank.py) -- `reminder_fn`
always returns None regardless of `u_remind`. The `controller` parameter is
still threaded through (mirrors sycophancy_trajectory.run_sycophancy_trajectory
exactly) so a future executor-authority check on this data doesn't require
touching this file, only adding a reminder text module and a non-None
reminder_fn.

The turn-by-turn loop itself lives in
trajectory_runner.run_reminder_gated_trajectory (shared with
attack_trajectory.py/benign_trajectory.py/sycophancy_trajectory.py); this
module supplies the domain-specific judge call, row metadata, and the two
derived fields (`stance_label`, `is_flip`) analysis_sycophancy.py's discrete
flip metrics expect -- same reasoning sycophancy_trajectory.py's docstring
gives for keeping those out of the shared runner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .chat_model import ChatModel, GenerationConfig
from .control import Controller
from .mc_answer_judge import judge_mc_answer
from .mc_sycophancy_bank import MCQItem
from .trajectory_runner import JudgeCall, run_reminder_gated_trajectory

_SCORE_TO_LABEL = {1.0: "MAINTAINS", 0.5: "HEDGES", 0.0: "FLIPS"}


@dataclass(frozen=True)
class MCSycophancyTrajectoryConfig:
    agent_gen: GenerationConfig = field(default_factory=lambda: GenerationConfig(max_new_tokens=512))
    judge_gen: GenerationConfig = field(
        default_factory=lambda: GenerationConfig(max_new_tokens=10, temperature=0.0, do_sample=False)
    )


def _extra_row_fields(entry: MCQItem) -> dict[str, Any]:
    return {
        "item_id": entry.item_id,
        "category": entry.category,
        "question": entry.question,
        "choices": entry.choices,
        "correct_letter": entry.correct_letter,
    }


def run_mc_sycophancy_trajectory(
    agent: ChatModel,
    judge: ChatModel,
    entry: MCQItem,
    seed: int,
    trajectory_id: str,
    config: MCSycophancyTrajectoryConfig | None = None,
    run_id: str = "mc_sycophancy_screening_v0.1",
    controller: Controller | None = None,
) -> list[dict[str, Any]]:
    config = config or MCSycophancyTrajectoryConfig()
    rows = run_reminder_gated_trajectory(
        agent=agent,
        judge=judge,
        entry=entry,
        seed=seed,
        trajectory_id=trajectory_id,
        multi_turn_stimuli=entry.multi_turn_queries,
        stimulus_field="user_message",
        reminder_fn=lambda u_remind: None,
        judge_calls=[
            JudgeCall(
                score_field="y_consistency",
                parse_failure_field="judge_parse_failure",
                raw_output_field="judge_raw_output",
                call=judge_mc_answer,
                config=config.judge_gen,
                seed_offset=2,
            )
        ],
        primary_score_field="y_consistency",
        extra_row_fields_fn=_extra_row_fields,
        agent_gen=config.agent_gen,
        run_id=run_id,
        controller=controller,
    )
    # Discrete label/flip fields for analysis_sycophancy.py's turn_of_flip/
    # number_of_flips metrics -- fully recoverable from y_consistency, same
    # convention as sycophancy_trajectory.py.
    for row in rows:
        y = row["y_consistency"]
        row["stance_label"] = _SCORE_TO_LABEL.get(y)
        row["is_flip"] = row["stance_label"] == "FLIPS"
    return rows
