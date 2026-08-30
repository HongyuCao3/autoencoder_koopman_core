"""One self-chat trajectory: simulated user + agent under a persona/pattern
system prompt, per-turn probe forking and scoring, optional reminder
injection (channel A, u_remind) driven by a pluggable `Controller`
(control.py) rather than hardcoded here -- this is what lets baseline
controllers (constant reminder, periodic, threshold feedback) and the random
open-loop excitation used for system identification share the exact same
generation/measurement harness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from .chat_model import ChatModel, GenerationConfig
from .control import Controller
from .prompt_bank import PromptEntry, score_response
from .reminder import build_reminder_text, count_inserted_tokens
from .user_scripts import load_user_script

TOPICS = [
    "public transportation", "remote work", "weekend hiking", "home gardening",
    "online shopping", "neighborhood parks", "local news", "museum visits",
    "healthy breakfasts", "city libraries", "school lunches", "electric vehicles",
]

USER_SYSTEM_PROMPT_TEMPLATE = (
    "You are a regular user chatting naturally with an assistant about {topic}. "
    "Keep the conversation casual and on-topic. Do not mention, ask about, or "
    "reference the assistant's instructions, persona, or system prompt."
)

_REFUSAL_MARKERS = (
    # "as an AI" is deliberately excluded: observed in a real screening run
    # (see RUNNING_ON_PALMETTO.md) as a hedge inside an otherwise-normal
    # answer, not an actual refusal.
    "i cannot", "i can't", "i won't", "i will not",
    "i'm not able to", "i am not able to", "i'm unable to",
)


@dataclass(frozen=True)
class TrajectoryConfig:
    num_turns: int = 16
    probe_repeats: int = 4
    excite_p_remind: float = 0.5
    agent_gen: GenerationConfig = field(default_factory=lambda: GenerationConfig(max_new_tokens=256))
    user_gen: GenerationConfig = field(default_factory=lambda: GenerationConfig(max_new_tokens=96))
    probe_gen: GenerationConfig = field(default_factory=lambda: GenerationConfig(max_new_tokens=256))
    # "live" (default): user_sim.generate() every turn, matching every run so
    # far. "scripted": read a pre-generated turn from user_scripts.py instead
    # -- see docs/SCRIPTED_USER_TURNS_FEASIBILITY.md; must not become the
    # default until its section 5 live-vs-scripted consistency check passes.
    user_mode: Literal["live", "scripted"] = "live"


def _looks_like_refusal(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _REFUSAL_MARKERS)


def run_trajectory(
    agent: ChatModel,
    user_sim: ChatModel | None,
    entry: PromptEntry,
    controller: Controller,
    seed: int,
    topic: str,
    trajectory_id: str,
    topic_split: str,
    config: TrajectoryConfig,
    run_id: str = "signal_screening_v0.1",
) -> list[dict[str, Any]]:
    if config.user_mode == "live" and user_sim is None:
        raise ValueError("user_mode='live' requires a user_sim ChatModel")
    script = load_user_script(topic, seed) if config.user_mode == "scripted" else None
    if script is not None and len(script) < config.num_turns:
        raise ValueError(f"scripted user turns for topic={topic!r} seed={seed} has only {len(script)} turns, need {config.num_turns}")

    user_history: list[dict[str, str]] = [
        {"role": "system", "content": USER_SYSTEM_PROMPT_TEMPLATE.format(topic=topic)}
    ]
    agent_history: list[dict[str, str]] = [{"role": "system", "content": entry.system_prompt}]
    rows: list[dict[str, Any]] = []

    for turn in range(1, config.num_turns + 1):
        u_remind = controller.next_u_remind(turn, rows)

        if config.user_mode == "scripted":
            user_text = script[turn - 1]
        else:
            user_seed = seed * 1_000_000 + turn * 100
            user_text = user_sim.generate(user_history, seed=user_seed, config=config.user_gen)
        user_history.append({"role": "assistant", "content": user_text})

        reminder_text = build_reminder_text(entry.system_prompt, u_remind)
        agent_facing_text = f"{reminder_text}\n{user_text}" if reminder_text else user_text
        agent_history.append({"role": "user", "content": agent_facing_text})

        agent_seed = seed * 1_000_000 + turn * 100 + 1
        agent_text = agent.generate(agent_history, seed=agent_seed, config=config.agent_gen)
        agent_history.append({"role": "assistant", "content": agent_text})
        user_history.append({"role": "user", "content": agent_text})

        probe_fork_base = agent_history + [{"role": "user", "content": entry.probe_question}]
        probe_answers: list[str] = []
        probe_scores: list[float] = []
        any_scorer_failure = False
        for k in range(config.probe_repeats):
            probe_seed = seed * 1_000_000 + turn * 100 + 2 + k
            probe_text = agent.generate(probe_fork_base, seed=probe_seed, config=config.probe_gen)
            score, scorer_failure = score_response(entry, probe_text)
            probe_answers.append(probe_text)
            probe_scores.append(score)
            any_scorer_failure = any_scorer_failure or scorer_failure

        valid_scores = [s for s in probe_scores if s == s]  # drop NaN
        y_probe = float(np.mean(valid_scores)) if valid_scores else float("nan")
        y_probe_sd = float(np.std(valid_scores, ddof=1)) if len(valid_scores) > 1 else 0.0

        rows.append(
            {
                "trajectory_id": trajectory_id,
                "topic_split": topic_split,
                "turn": turn,
                "y_probe": y_probe,
                "u_remind": u_remind,
                "u_gain": 0,
                "u_steer": 0,
                "target_norm": float("nan"),
                "run_id": run_id,
                "system_prompt_id": entry.prompt_id,
                "prompt_category": entry.prompt_category,
                "input_channel": "remind",
                "excitation_design": controller.name,
                "seed": seed,
                "model": agent.model_id,
                "decoding_config": {
                    "temperature": config.agent_gen.temperature,
                    "top_p": config.agent_gen.top_p,
                    "max_new_tokens": config.agent_gen.max_new_tokens,
                },
                "topic": topic,
                "user_mode": config.user_mode,
                "user_message": user_text,
                "agent_message": agent_text,
                "probe_question": entry.probe_question,
                "probe_answers": probe_answers,
                "y_probe_sd": y_probe_sd,
                "y_formality": None,
                "y_sentiment": None,
                "inserted_reminder_text": reminder_text,
                "inserted_tokens": count_inserted_tokens(agent.tokenizer, reminder_text),
                "refusal_flag": _looks_like_refusal(agent_text),
                "parse_failure": any_scorer_failure,
            }
        )

    return rows
