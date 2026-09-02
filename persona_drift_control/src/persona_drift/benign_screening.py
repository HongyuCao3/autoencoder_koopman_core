"""Orchestrates the Phase F helpfulness-cost check
(docs/experiments/koopman_defense_pilot.md): replays the fixed 8-category
MT-Bench benign sessions (benign_bank.py) x seeds through one controller arm,
mirroring adversarial_screening.run_adversarial_screening's resumability/
logging/report-writing shape. Simpler than that module in one respect: there
is no attack_ids/num_attacks/rng_seed selection -- this pilot always uses
every category's fixed session (benign_bank.all_benign_sessions), no
sub-sampling.
"""

from __future__ import annotations

import json
import pathlib
from datetime import datetime
from typing import Any, Callable

from loguru import logger

from .analysis_helpfulness import analyze_benign_screening
from .benign_bank import all_benign_sessions, load_benign_bank
from .benign_trajectory import BenignTrajectoryConfig, run_benign_trajectory
from .chat_model import ChatModel
from .control import Controller, ZeroControlController
from .logging_setup import configure_run_logger
from .screening_common import load_agent_and_judge, prepare_resumable_trajectories_file, run_trajectories_loop


def run_benign_screening(
    agent_model_id: str,
    judge_model_id: str,
    output_dir: pathlib.Path,
    seeds: tuple[int, ...] = (0, 1),
    device: str = "cuda",
    trajectory_config: BenignTrajectoryConfig | None = None,
    enable_thinking: bool = False,
    controller_factory: Callable[[int, str], Controller] | None = None,
) -> dict[str, Any]:
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_config = trajectory_config or BenignTrajectoryConfig()
    controller_factory = controller_factory or (lambda seed, entry_id="": ZeroControlController())

    sessions = all_benign_sessions(load_benign_bank())

    controller_name = controller_factory(seeds[0]).name
    run_id = f"{output_dir.name}_think{int(enable_thinking)}_{controller_name}_{datetime.now():%Y%m%d_%H%M%S}"
    run_config = {
        "agent_model_id": agent_model_id,
        "judge_model_id": judge_model_id,
        "seeds": list(seeds),
        "device": device,
        "enable_thinking": enable_thinking,
        "controller": controller_name,
        "benign_ids": [entry.benign_id for entry in sessions],
        "output_dir": str(output_dir),
    }
    configure_run_logger(run_id, run_config)

    trajectories_path = output_dir / "trajectories.jsonl"
    expected_rows_by_trajectory_id = {
        f"{entry.benign_id}__seed{seed}": len(entry.multi_turn_queries) for entry in sessions for seed in seeds
    }
    completed_by_tid = prepare_resumable_trajectories_file(trajectories_path, expected_rows_by_trajectory_id)
    if completed_by_tid:
        logger.info(
            "resuming: {} already-completed trajectories found in {}",
            len(completed_by_tid),
            trajectories_path,
        )

    total_trajectories = len(sessions) * len(seeds)
    agent, judge = load_agent_and_judge(
        ChatModel,
        agent_model_id,
        judge_model_id,
        device,
        enable_thinking,
        needed=len(completed_by_tid) < total_trajectories,
    )

    rows = run_trajectories_loop(
        entries=sessions,
        id_fn=lambda entry: entry.benign_id,
        seeds=seeds,
        controller_factory=controller_factory,
        trajectory_config=trajectory_config,
        agent=agent,
        judge=judge,
        trajectory_runner=run_benign_trajectory,
        trajectories_path=trajectories_path,
        completed_by_tid=completed_by_tid,
    )

    report = analyze_benign_screening(rows)
    report["config"] = run_config

    report_path = output_dir / "benign_screening_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    (output_dir / "benign_screening_report.md").write_text(_render_markdown(report))
    logger.info(
        "mean_y_help={:.4f} refusal_rate={:.4f} report written to {}",
        report["diagnostics"]["mean_y_help"],
        report["diagnostics"]["refusal_rate"],
        output_dir,
    )

    return report


def _render_markdown(report: dict[str, Any]) -> str:
    diag = report["diagnostics"]
    lines = [
        "# Benign helpfulness-cost screening report",
        "",
        "## Diagnostics",
        f"- mean_y_help: {diag['mean_y_help']:.4f} (sd={diag['sd_y_help']:.4f})",
        f"- refusal_rate: {diag['refusal_rate']:.4f}",
        f"- judge_parse_failure_rate: {diag['judge_parse_failure_rate']:.4f}",
        f"- helpfulness_parse_failure_rate: {diag['helpfulness_parse_failure_rate']:.4f}",
        f"- reminders inserted: {diag['n_reminders_inserted']}/{diag['n_rows']}",
        f"- total_inserted_tokens: {diag['total_inserted_tokens']}",
        "- y_help by category:",
    ]
    for category, stats in diag["y_help_by_category"].items():
        lines.append(f"  - {category}: mean={stats['mean']:.4f}, sd={stats['sd']:.4f}")
    lines.append("- y_help by turn:")
    for turn, stats in sorted(diag["y_help_by_turn"].items()):
        lines.append(f"  - turn {turn}: mean={stats['mean']:.4f} (n={stats['n']})")
    return "\n".join(lines) + "\n"
