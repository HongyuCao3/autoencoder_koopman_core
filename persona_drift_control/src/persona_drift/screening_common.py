"""Shared pieces between adversarial_screening.run_adversarial_screening and
benign_screening.run_benign_screening: the two screening loops (fixed attack
sequences vs fixed benign sessions) are otherwise structurally identical --
resumability bookkeeping, agent/judge lazy-loading order, per-trajectory
logging/writing -- so this module holds the parts that carry no behavioral
difference between them. Agent/judge construction still goes through a
caller-supplied `chat_model_cls` (rather than importing ChatModel here)
so each caller's own module-level `ChatModel` name stays the one tests
monkeypatch.

Not shared with screening.py's own `_prepare_resumable_trajectories_file`:
that one tracks a single fixed expected-rows-per-trajectory constant, while
this one is keyed per trajectory_id (attack/benign sequences have variable
length) -- see adversarial_screening.py's module docstring.
"""

from __future__ import annotations

import json
import pathlib
import time
from typing import Any, Callable

from loguru import logger


def prepare_resumable_trajectories_file(
    trajectories_path: pathlib.Path, expected_rows_by_trajectory_id: dict[str, int]
) -> dict[str, list[dict[str, Any]]]:
    """Truncates trajectories_path down to only the trajectory_ids whose row
    count already matches expected_rows_by_trajectory_id, and returns those
    completed rows keyed by trajectory_id (sorted by turn) so the caller can
    skip re-running them."""

    completed_by_tid: dict[str, list[dict[str, Any]]] = {}
    if trajectories_path.exists():
        rows_by_tid: dict[str, list[dict[str, Any]]] = {}
        for line in trajectories_path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            rows_by_tid.setdefault(row["trajectory_id"], []).append(row)
        completed_by_tid = {
            tid: sorted(rs, key=lambda r: r["turn"])
            for tid, rs in rows_by_tid.items()
            if len(rs) == expected_rows_by_trajectory_id.get(tid)
        }

    with trajectories_path.open("w") as handle:
        for rs in completed_by_tid.values():
            for row in rs:
                handle.write(json.dumps(row) + "\n")

    return completed_by_tid


def load_agent_and_judge(
    chat_model_cls: Callable[..., Any],
    agent_model_id: str,
    judge_model_id: str,
    device: str,
    enable_thinking: bool,
    needed: bool,
) -> tuple[Any, Any]:
    """`needed` should be False when every trajectory is already completed
    (resumability) -- skips loading either model at all in that case."""

    agent = None
    judge = None
    if needed:
        logger.info("loading agent model {} (enable_thinking={})", agent_model_id, enable_thinking)
        agent = chat_model_cls(agent_model_id, device=device, enable_thinking=enable_thinking)
        if judge_model_id == agent_model_id:
            logger.info("judge_model == agent_model: reusing the loaded agent instance as judge")
            judge = agent
        else:
            logger.info("loading judge model {}", judge_model_id)
            # enable_thinking=False regardless of the agent's setting: judge
            # calls always pin enable_thinking=False per-call (see
            # safety_judge.judge_safety_score / helpfulness_judge.judge_helpfulness_score),
            # so this instance default is never actually used, but it
            # documents intent even when the judge model happens to be a
            # different, separately-loaded instance from the agent.
            judge = chat_model_cls(judge_model_id, device=device, enable_thinking=False)
    return agent, judge


def run_trajectories_loop(
    entries: list[Any],
    id_fn: Callable[[Any], str],
    seeds: tuple[int, ...],
    controller_factory: Callable[[int], Any],
    trajectory_config: Any,
    agent: Any,
    judge: Any,
    trajectory_runner: Callable[..., list[dict[str, Any]]],
    trajectories_path: pathlib.Path,
    completed_by_tid: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Iterates entries x seeds, skipping anything already in
    completed_by_tid, appending each trajectory's rows to trajectories_path
    as they're produced (so a crash mid-run only loses the in-flight
    trajectory, not earlier ones)."""

    total_trajectories = len(entries) * len(seeds)
    rows: list[dict[str, Any]] = [row for rs in completed_by_tid.values() for row in rs]
    completed = len(completed_by_tid)
    run_start = time.monotonic()
    with trajectories_path.open("a") as handle:
        for entry in entries:
            for seed in seeds:
                trajectory_id = f"{id_fn(entry)}__seed{seed}"
                if trajectory_id in completed_by_tid:
                    logger.info("[{}/{}] skipping already-completed {}", completed, total_trajectories, trajectory_id)
                    continue
                logger.info(
                    "[{}/{}] starting {} (+{:.0f}s)",
                    completed,
                    total_trajectories,
                    trajectory_id,
                    time.monotonic() - run_start,
                )
                trajectory_start = time.monotonic()
                trajectory_rows = trajectory_runner(
                    agent=agent,
                    judge=judge,
                    entry=entry,
                    seed=seed,
                    trajectory_id=trajectory_id,
                    config=trajectory_config,
                    controller=controller_factory(seed),
                )
                for row in trajectory_rows:
                    handle.write(json.dumps(row) + "\n")
                handle.flush()
                rows.extend(trajectory_rows)
                completed += 1
                logger.info(
                    "[{}/{}] finished {} in {:.0f}s (+{:.0f}s total)",
                    completed,
                    total_trajectories,
                    trajectory_id,
                    time.monotonic() - trajectory_start,
                    time.monotonic() - run_start,
                )
    return rows
