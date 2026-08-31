"""Orchestrates the escalating persona-pressure confirmation pilot
(docs/experiments/drift_confirmation_pilot.md, "下一步的小范围实验建议"): a
handful of prompts x seeds, compared under two conditions --

- "baseline": user_mode="live" (natural self-chat, no pressure) -- the same
  free-drift condition as signal_screening_pilot.md/drift_confirmation_pilot.md's
  zero_control, expected to replicate their clean null as a sanity check.
- "escalating_pressure": user_mode="pressure" (fixed, category-keyed
  escalating scripts from pressure_scripts.py) -- tests whether the null was
  a stimulus-strength artifact rather than a Qwen3-4B capacity/training
  property, by reusing the "progressive external pressure" design that
  already produced a clean signal on the same model in a different domain
  (adversarial_screening_pilot.md).

Both conditions use ZeroControlController: channel A (u_remind) plays no
role in either -- the pressure/no-pressure distinction lives entirely in
`user_mode`, not in `excitation_design` (see analysis_pressure.py's
docstring). Structurally mirrors screening.py, including its resumability
helper (imported directly since the logic -- fixed expected-rows-per-
trajectory equal to num_turns -- is identical here, unlike
adversarial_screening.py's variable-length variant).
"""

from __future__ import annotations

import json
import pathlib
import random
import time
from dataclasses import replace
from datetime import datetime
from typing import Any

from loguru import logger

from .analysis_pressure import analyze_pressure_screening
from .chat_model import ChatModel
from .control import ZeroControlController
from .logging_setup import configure_run_logger
from .prompt_bank import load_prompt_bank, select_screening_prompts
from .screening import _prepare_resumable_trajectories_file
from .selfchat import TOPICS, TrajectoryConfig, run_trajectory

CONDITIONS = ("baseline", "escalating_pressure")
_USER_MODE_BY_CONDITION = {"baseline": "live", "escalating_pressure": "pressure"}


def run_pressure_screening(
    agent_model_id: str,
    user_model_id: str,
    output_dir: pathlib.Path,
    num_prompts: int = 4,
    seeds: tuple[int, ...] = (0, 1),
    prompt_rng_seed: int = 0,
    device: str = "cuda",
    trajectory_config: TrajectoryConfig | None = None,
) -> dict[str, Any]:
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_config = trajectory_config or TrajectoryConfig()

    bank = load_prompt_bank()
    prompts = select_screening_prompts(bank, num_prompts=num_prompts, rng_seed=prompt_rng_seed)

    run_id = f"{output_dir.name}_{datetime.now():%Y%m%d_%H%M%S}"
    run_config = {
        "agent_model_id": agent_model_id,
        "user_model_id": user_model_id,
        "num_prompts": num_prompts,
        "seeds": list(seeds),
        "prompt_rng_seed": prompt_rng_seed,
        "device": device,
        "conditions": list(CONDITIONS),
        "trajectory_config": {
            "num_turns": trajectory_config.num_turns,
            "probe_repeats": trajectory_config.probe_repeats,
        },
        "prompt_ids": [entry.prompt_id for entry in prompts],
        "output_dir": str(output_dir),
    }
    configure_run_logger(run_id, run_config)

    trajectories_path = output_dir / "trajectories.jsonl"
    completed_by_tid, topic_by_prompt_id = _prepare_resumable_trajectories_file(
        trajectories_path, trajectory_config.num_turns
    )
    if completed_by_tid:
        logger.info(
            "resuming: {} already-completed trajectories found in {}",
            len(completed_by_tid),
            trajectories_path,
        )

    total_trajectories = len(prompts) * len(seeds) * len(CONDITIONS)
    remaining_needs_live = any(
        f"{entry.prompt_id}__seed{seed}__baseline" not in completed_by_tid
        for entry in prompts
        for seed in seeds
    )

    agent = None
    user_sim = None
    if len(completed_by_tid) < total_trajectories:
        logger.info("loading agent model {}", agent_model_id)
        agent = ChatModel(agent_model_id, device=device)
        if remaining_needs_live:
            logger.info("loading user-simulator model {}", user_model_id)
            user_sim = ChatModel(user_model_id, device=device)
        else:
            logger.info("all remaining baseline trajectories already completed: skipping user-simulator model load")

    topic_rng = random.Random(prompt_rng_seed)
    rows: list[dict[str, Any]] = [row for rs in completed_by_tid.values() for row in rs]
    completed = len(completed_by_tid)
    run_start = time.monotonic()
    with trajectories_path.open("a") as handle:
        for entry in prompts:
            drawn_topic = topic_rng.choice(TOPICS)
            topic = topic_by_prompt_id.get(entry.prompt_id, drawn_topic)
            for seed in seeds:
                for condition in CONDITIONS:
                    trajectory_id = f"{entry.prompt_id}__seed{seed}__{condition}"
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
                    condition_config = replace(trajectory_config, user_mode=_USER_MODE_BY_CONDITION[condition])
                    trajectory_start = time.monotonic()
                    trajectory_rows = run_trajectory(
                        agent=agent,
                        user_sim=user_sim,
                        entry=entry,
                        controller=ZeroControlController(),
                        seed=seed,
                        topic=topic,
                        trajectory_id=trajectory_id,
                        topic_split="pressure_screening",
                        config=condition_config,
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

    report = analyze_pressure_screening(rows)
    report["config"] = run_config

    report_path = output_dir / "pressure_screening_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    (output_dir / "pressure_screening_report.md").write_text(_render_markdown(report))
    logger.info(
        "q1_baseline_no_pressure.pass={} q1_escalating_pressure.pass={} report written to {}",
        report["q1_baseline_no_pressure"]["pass"],
        report["q1_escalating_pressure"]["pass"],
        output_dir,
    )

    return report


def _render_markdown(report: dict[str, Any]) -> str:
    baseline, pressure = report["q1_baseline_no_pressure"], report["q1_escalating_pressure"]
    diag = report["diagnostics"]
    lines = [
        "# Persona-pressure confirmation pilot report",
        "",
        "## Baseline (user_mode=live, no pressure -- expected to replicate the prior clean null)",
        f"- prompts with negative slope: {baseline['n_negative_slope_prompts']}/{baseline['n_prompts']}",
        f"- t-test of mean per-prompt slope vs 0: t={baseline['t_test_mean_slope_vs_zero']['t']:.4f}, "
        f"p={baseline['t_test_mean_slope_vs_zero']['p']:.4f}",
        f"- pass (t<0 and p<0.05): {baseline['pass']}",
        "",
        "## Escalating pressure (user_mode=pressure -- the new condition being tested)",
        f"- prompts with negative slope: {pressure['n_negative_slope_prompts']}/{pressure['n_prompts']}",
        f"- t-test of mean per-prompt slope vs 0: t={pressure['t_test_mean_slope_vs_zero']['t']:.4f}, "
        f"p={pressure['t_test_mean_slope_vs_zero']['p']:.4f}",
        f"- pass (t<0 and p<0.05): {pressure['pass']}",
        "",
        "## Diagnostics",
        f"- refusal_rate: {diag['refusal_rate']:.4f}",
        f"- scorer_failure_rate: {diag['scorer_failure_rate']:.4f}",
        "- y_probe by condition/category:",
    ]
    for key, stats in diag["y_probe_by_condition_and_category"].items():
        lines.append(f"  - {key}: mean={stats['mean']:.4f}, sd={stats['sd']:.4f}")
    return "\n".join(lines) + "\n"
