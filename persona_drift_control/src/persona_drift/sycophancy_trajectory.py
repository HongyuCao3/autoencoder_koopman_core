"""One sycophancy-drift screening trajectory: a fixed SYCON-Bench
false-presupposition item (sycophancy_bank.SycophancyItem) is replayed
turn by turn -- turn 1 is the neutral question (establishes the agent's
initial position), turns 2..N replay the item's fixed escalating pushback
script. Each turn's reply is scored by sycophancy_judge.judge_sycophancy_score
against the item's ground-truth `correction`/`presupposition` fields (not
against the agent's own turn-1 answer -- see sycophancy_judge.py's docstring
for why the dataset's ground truth is used instead).

An optional `controller` can insert a fixed consistency reminder
(consistency_reminder.py) before each turn's user message, same channel-A
mechanism as attack_trajectory.py/benign_trajectory.py. Passing no
controller (the default) is exactly equivalent to `ZeroControlController`.

The turn-by-turn loop itself lives in
trajectory_runner.run_reminder_gated_trajectory (shared with
attack_trajectory.py/benign_trajectory.py); this module supplies the
domain-specific judge call, row metadata, and the two extra derived fields
(`stance_label`, `is_flip`) that the shared runner doesn't know about --
see trajectory_runner.py's docstring for why those stay out of the shared
code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .chat_model import ChatModel, GenerationConfig
from .consistency_reminder import build_consistency_reminder_text
from .control import Controller
from .sycophancy_bank import SycophancyItem
from .sycophancy_judge import judge_sycophancy_score
from .trajectory_runner import JudgeCall, run_reminder_gated_trajectory

_SCORE_TO_LABEL = {1.0: "MAINTAINS", 0.5: "HEDGES", 0.0: "FLIPS"}


@dataclass(frozen=True)
class SycophancyTrajectoryConfig:
    agent_gen: GenerationConfig = field(default_factory=lambda: GenerationConfig(max_new_tokens=256))
    judge_gen: GenerationConfig = field(
        default_factory=lambda: GenerationConfig(max_new_tokens=10, temperature=0.0, do_sample=False)
    )


def _sycophancy_judge_call(
    judge: ChatModel,
    entry: SycophancyItem,
    rows: list[dict[str, Any]],
    turn: int,
    stimulus: str,
    response: str,
    seed: int,
    config: GenerationConfig,
) -> tuple[float, bool, str]:
    return judge_sycophancy_score(
        judge, entry.question, entry.correction, entry.presupposition, response, seed=seed, config=config
    )


def _extra_row_fields(entry: SycophancyItem) -> dict[str, Any]:
    return {
        "item_id": entry.item_id,
        "category": entry.category,
        "question": entry.question,
        "presupposition": entry.presupposition,
        "correction": entry.correction,
    }


def run_sycophancy_trajectory(
    agent: ChatModel,
    judge: ChatModel,
    entry: SycophancyItem,
    seed: int,
    trajectory_id: str,
    config: SycophancyTrajectoryConfig | None = None,
    run_id: str = "sycophancy_screening_v0.1",
    controller: Controller | None = None,
) -> list[dict[str, Any]]:
    config = config or SycophancyTrajectoryConfig()
    rows = run_reminder_gated_trajectory(
        agent=agent,
        judge=judge,
        entry=entry,
        seed=seed,
        trajectory_id=trajectory_id,
        multi_turn_stimuli=entry.multi_turn_queries,
        stimulus_field="user_message",
        reminder_fn=build_consistency_reminder_text,
        judge_calls=[
            JudgeCall(
                score_field="y_consistency",
                parse_failure_field="judge_parse_failure",
                raw_output_field="judge_raw_output",
                call=_sycophancy_judge_call,
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
    # number_of_flips metrics (docs/task/SYCOPHANCY_DRIFT_TASK_FEASIBILITY.md
    # section 5) -- fully recoverable from y_consistency, kept out of
    # trajectory_runner/sycophancy_judge on purpose, see their docstrings.
    for row in rows:
        y = row["y_consistency"]
        row["stance_label"] = _SCORE_TO_LABEL.get(y)
        row["is_flip"] = row["stance_label"] == "FLIPS"
    return rows
