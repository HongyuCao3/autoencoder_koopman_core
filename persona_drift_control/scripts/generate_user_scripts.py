#!/usr/bin/env python3
"""One-time offline generation of the scripted-user-turn library
(docs/SCRIPTED_USER_TURNS_FEASIBILITY.md section 3): for each topic in
selfchat.TOPICS, run one throwaway live self-chat against a neutral
reference agent to get a plausible 16-turn agent-reply scaffold, then
generate `--seeds`-many diverse user-turn scripts conditioned on that same
scaffold. The scaffold is intentionally NOT tied to any real experimental
system prompt (REFERENCE_SYSTEM_PROMPT below) -- reusing a real persona's
agent replies here would bake that persona's behavior into every other
prompt's scripted-user data.

Writes persona_drift_control/resources/user_scripts/<topic-slug>__seed<seed>.json
(a JSON list of `--num-turns` strings) plus a PROVENANCE.json recording how
they were made. Must be run where torch/transformers are installed and a GPU
is available (same env as run_signal_screening.py).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from datetime import datetime

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from persona_drift.chat_model import ChatModel, GenerationConfig  # noqa: E402
from persona_drift.selfchat import TOPICS, USER_SYSTEM_PROMPT_TEMPLATE  # noqa: E402
from persona_drift.user_scripts import USER_SCRIPTS_DIR, slugify_topic  # noqa: E402

REFERENCE_SYSTEM_PROMPT = "You are a helpful, friendly assistant."


def build_reference_agent_turns(
    agent: ChatModel,
    user_sim: ChatModel,
    topic: str,
    reference_seed: int,
    num_turns: int,
    agent_gen: GenerationConfig,
    user_gen: GenerationConfig,
) -> list[str]:
    """A single real agent+user self-chat (neutral persona) used only as a
    scaffold of plausible agent replies -- not itself part of any script."""

    user_history: list[dict[str, str]] = [
        {"role": "system", "content": USER_SYSTEM_PROMPT_TEMPLATE.format(topic=topic)}
    ]
    agent_history: list[dict[str, str]] = [{"role": "system", "content": REFERENCE_SYSTEM_PROMPT}]
    agent_turns: list[str] = []
    for turn in range(1, num_turns + 1):
        user_seed = reference_seed * 1_000_000 + turn * 100
        user_text = user_sim.generate(user_history, seed=user_seed, config=user_gen)
        user_history.append({"role": "assistant", "content": user_text})
        agent_history.append({"role": "user", "content": user_text})

        agent_seed = reference_seed * 1_000_000 + turn * 100 + 1
        agent_text = agent.generate(agent_history, seed=agent_seed, config=agent_gen)
        agent_history.append({"role": "assistant", "content": agent_text})
        user_history.append({"role": "user", "content": agent_text})
        agent_turns.append(agent_text)
    return agent_turns


def build_user_script(
    user_sim: ChatModel,
    topic: str,
    seed: int,
    reference_agent_turns: list[str],
    user_gen: GenerationConfig,
) -> list[str]:
    """turn 1 has no prior agent turn to react to (matches selfchat.py's live
    loop); turn t>=2 is conditioned on reference_agent_turns[t-2], NOT on
    whatever a specific real trajectory's agent actually said -- this is the
    documented feedback-loop simplification, not a bug."""

    user_history: list[dict[str, str]] = [
        {"role": "system", "content": USER_SYSTEM_PROMPT_TEMPLATE.format(topic=topic)}
    ]
    script: list[str] = []
    for turn, ref_agent_text in enumerate(reference_agent_turns, start=1):
        user_seed = seed * 1_000_000 + turn * 100 + 500_000  # offset: don't collide with reference's own seeds
        user_text = user_sim.generate(user_history, seed=user_seed, config=user_gen)
        script.append(user_text)
        user_history.append({"role": "assistant", "content": user_text})
        user_history.append({"role": "user", "content": ref_agent_text})
    return script


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user-model", default="Qwen/Qwen3-4B")
    parser.add_argument("--agent-model", default="Qwen/Qwen3-4B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--reference-seed", type=int, default=999)
    parser.add_argument("--num-turns", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # repetition_penalty/no_repeat_ngram_size: the reference/script self-chat
    # here is long (16 turns) and ungrounded (no real feedback loop turn to
    # turn, see build_user_script's docstring) -- observed to collapse into
    # near-verbatim repeated paragraphs by ~turn 10 without these. Scoped to
    # this offline generation only; selfchat.py's live loop keeps
    # transformers' no-op defaults (1.0 / 0), matching the already-validated
    # screening run's decoding behavior exactly.
    user_gen = GenerationConfig(max_new_tokens=96, repetition_penalty=1.15, no_repeat_ngram_size=4)
    agent_gen = GenerationConfig(max_new_tokens=256, repetition_penalty=1.15, no_repeat_ngram_size=4)

    user_sim = ChatModel(args.user_model, device=args.device)
    agent = ChatModel(args.agent_model, device=args.device)

    USER_SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

    for topic in TOPICS:
        reference_agent_turns = build_reference_agent_turns(
            agent, user_sim, topic, args.reference_seed, args.num_turns, agent_gen, user_gen
        )
        for seed in args.seeds:
            script = build_user_script(user_sim, topic, seed, reference_agent_turns, user_gen)
            path = USER_SCRIPTS_DIR / f"{slugify_topic(topic)}__seed{seed}.json"
            path.write_text(json.dumps(script, indent=2))
            print(f"wrote {path}")

    provenance = {
        "generated_at": datetime.now().isoformat(),
        "user_model_id": args.user_model,
        "agent_model_id": args.agent_model,
        "reference_seed": args.reference_seed,
        "seeds": args.seeds,
        "num_turns": args.num_turns,
        "reference_system_prompt": REFERENCE_SYSTEM_PROMPT,
        "topics": TOPICS,
    }
    (USER_SCRIPTS_DIR / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
