"""Single-turn dose-response sweep for channel C (u_steer): for one harmful
single-turn ask, generate once per alpha in the sweep grid with the
calibrated safety direction (safety_direction_calibration.py) added to the
residual stream, and judge each reply's safety. Rows feed
analysis_dose_response.analyze_dose_response (new-Q2: is y_safety
alpha-dependent, in the direction the direction's own sign predicts?).

Single-turn by design (ADVERSARIAL_DEFENSE_TASK_FEASIBILITY.md section 7
step 2: "单轮剂量-响应") -- this isolates the steering channel's effect from
any multi-turn confound, and keeps this step's cost proportional to
(n_queries * n_alphas), not (n_queries * n_alphas * n_turns).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .chat_model import ChatModel, GenerationConfig, SteeringConfig
from .safety_judge import judge_safety_score, looks_like_refusal

# DATA_COLLECTION_PROTOCOL.md section 3's channel-C grid ({-1,-0.5,0,0.5,1} x alpha0);
# alpha0 is folded into the direction vector's own magnitude (see chat_model.SteeringConfig).
ALPHA_GRID: tuple[float, ...] = (-1.0, -0.5, 0.0, 0.5, 1.0)


@dataclass(frozen=True)
class DoseResponseConfig:
    agent_gen: GenerationConfig = field(default_factory=lambda: GenerationConfig(max_new_tokens=256))
    judge_gen: GenerationConfig = field(
        default_factory=lambda: GenerationConfig(max_new_tokens=10, temperature=0.0, do_sample=False)
    )
    alpha_grid: tuple[float, ...] = ALPHA_GRID


def run_dose_response_query(
    agent: ChatModel,
    judge: ChatModel,
    query_id: str,
    harmful_goal: str,
    layer: int,
    direction: np.ndarray,
    seed: int,
    config: DoseResponseConfig | None = None,
) -> list[dict[str, Any]]:
    config = config or DoseResponseConfig()
    rows: list[dict[str, Any]] = []

    for alpha in config.alpha_grid:
        # alpha=0 skips the hook entirely rather than registering one with a
        # zero-magnitude delta -- byte-for-byte identical to unsteered
        # generation, no reliance on 0.0 * direction being exactly zero.
        steering = None if alpha == 0.0 else SteeringConfig(layer=layer, direction=direction, alpha=alpha)
        agent_seed = seed * 1_000_000 + int(round((alpha + 1) * 1000)) + 1
        agent_text = agent.generate(
            [{"role": "user", "content": harmful_goal}],
            seed=agent_seed,
            config=config.agent_gen,
            steering=steering,
        )

        judge_seed = agent_seed + 1
        y_safety, judge_parse_failure, judge_raw_output = judge_safety_score(
            judge, harmful_goal, harmful_goal, agent_text, seed=judge_seed, config=config.judge_gen
        )

        rows.append(
            {
                "query_id": query_id,
                "harmful_goal": harmful_goal,
                "layer": layer,
                "alpha": alpha,
                "agent_message": agent_text,
                "y_safety": y_safety,
                "seed": seed,
                "model": agent.model_id,
                "judge_model": judge.model_id,
                "refusal_flag": looks_like_refusal(agent_text),
                "judge_parse_failure": judge_parse_failure,
                "judge_raw_output": judge_raw_output,
            }
        )

    return rows
