"""Loads step-1 screening's real recorded conversation histories
(outputs/adversarial_screening/trajectories.jsonl, job 15399715) to build
dose-response queries that start from a genuinely eroded context, instead of
asking the attack's raw harmful goal as a bare first turn.

Why: the first dose-response pass (docs/experiments/dose_response_pilot.md,
job 15404914) hit a ceiling effect -- Qwen3-4B refuses a bare direct ask of
the harmful goal near-perfectly regardless of alpha (y_safety 0.8-1.0 at
every alpha level), leaving no headroom to see a dose-response at all. Step
1's own screening data already shows where safety actually erodes (turn4/
turn5 mean y_safety down to 0.688/0.450) -- this module replays the model's
own real, unsteered replies through turn N-1 of a trajectory as context, so
the final (steered) turn starts from that eroded state rather than a clean
one.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any


def load_eroded_contexts(
    trajectories_path: pathlib.Path,
    seed: int = 0,
    max_final_turn_y_safety: float = 0.8,
) -> list[dict[str, Any]]:
    """One entry per trajectory (filtered to `seed`, so the two seeds of the
    same attack_id don't count as two independent queries -- same reasoning
    as attack_trajectory's per-attack aggregation elsewhere in this
    project). Drops trajectories whose recorded final-turn y_safety is a
    judge-parse-failure NaN, or is already above `max_final_turn_y_safety`
    (no headroom left in the recorded run -- the same ceiling problem this
    module exists to avoid)."""

    rows_by_trajectory: dict[str, list[dict[str, Any]]] = {}
    for line in pathlib.Path(trajectories_path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["seed"] != seed:
            continue
        rows_by_trajectory.setdefault(row["trajectory_id"], []).append(row)

    contexts: list[dict[str, Any]] = []
    for trajectory_id, rows in rows_by_trajectory.items():
        rows = sorted(rows, key=lambda r: r["turn"])
        if len(rows) < 2:
            continue  # need at least one turn of real context before the steered turn
        *context_rows, final_row = rows

        final_y_safety = final_row["y_safety"]
        if final_y_safety != final_y_safety:  # NaN: judge parse failure in the recorded run
            continue
        if final_y_safety > max_final_turn_y_safety:
            continue

        context_messages: list[dict[str, str]] = []
        for row in context_rows:
            context_messages.append({"role": "user", "content": row["attacker_query"]})
            context_messages.append({"role": "assistant", "content": row["agent_message"]})

        contexts.append(
            {
                "query_id": trajectory_id,
                "attack_id": final_row["attack_id"],
                "plain_query": final_row["plain_query"],
                "context_messages": context_messages,
                "final_turn_query": final_row["attacker_query"],
                "recorded_final_y_safety": final_y_safety,
                "context_turns": len(context_rows),
            }
        )
    return contexts
