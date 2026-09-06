"""Orchestrates the ERGO/Laban sharded-GSM8K minimal executor-authority
check (docs/feasibility/ERGO_MULTITURN_RELIABILITY_FEASIBILITY.md's step 2)
-- structurally mirrors mc_sycophancy_screening.py/sycophancy_screening.py
(same controller_factory-per-trajectory pattern, same resumability, same
run_id convention), reusing screening_common.py's shared plumbing.
"""

from __future__ import annotations

import json
import pathlib
from datetime import datetime
from typing import Any, Callable

from loguru import logger

from .analysis_ergo_math import analyze_ergo_math_screening
from .chat_model import ChatModel
from .control import Controller, ZeroControlController
from .ergo_math_bank import load_ergo_math_bank, select_items_by_id, select_screening_items
from .ergo_math_trajectory import ErgoMathTrajectoryConfig, run_ergo_math_trajectory
from .logging_setup import configure_run_logger
from .screening_common import load_agent_and_judge, prepare_resumable_trajectories_file, run_trajectories_loop


def run_ergo_math_screening(
    agent_model_id: str,
    judge_model_id: str,
    output_dir: pathlib.Path,
    num_items: int = 20,
    seeds: tuple[int, ...] = (0, 1),
    item_rng_seed: int = 0,
    device: str = "cuda",
    trajectory_config: ErgoMathTrajectoryConfig | None = None,
    enable_thinking: bool = False,
    controller_factory: Callable[[int, str], Controller] | None = None,
    item_ids: list[str] | None = None,
) -> dict[str, Any]:
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_config = trajectory_config or ErgoMathTrajectoryConfig()
    controller_factory = controller_factory or (lambda seed, entry_id="": ZeroControlController())

    bank = load_ergo_math_bank()
    items = (
        select_items_by_id(bank, item_ids)
        if item_ids is not None
        else select_screening_items(bank, num_items=num_items, rng_seed=item_rng_seed)
    )

    controller_name = controller_factory(seeds[0]).name
    run_id = f"{output_dir.name}_think{int(enable_thinking)}_{controller_name}_{datetime.now():%Y%m%d_%H%M%S}"
    run_config = {
        "agent_model_id": agent_model_id,
        "judge_model_id": judge_model_id,
        "num_items": num_items,
        "seeds": list(seeds),
        "item_rng_seed": item_rng_seed,
        "device": device,
        "enable_thinking": enable_thinking,
        "controller": controller_name,
        "item_ids": [item.item_id for item in items],
        "output_dir": str(output_dir),
    }
    configure_run_logger(run_id, run_config)

    trajectories_path = output_dir / "trajectories.jsonl"
    expected_rows_by_trajectory_id = {
        f"{item.item_id}__seed{seed}": len(item.shards) for item in items for seed in seeds
    }
    completed_by_tid = prepare_resumable_trajectories_file(trajectories_path, expected_rows_by_trajectory_id)
    if completed_by_tid:
        logger.info(
            "resuming: {} already-completed trajectories found in {}",
            len(completed_by_tid),
            trajectories_path,
        )

    total_trajectories = len(items) * len(seeds)
    agent, judge = load_agent_and_judge(
        ChatModel,
        agent_model_id,
        judge_model_id,
        device,
        enable_thinking,
        needed=len(completed_by_tid) < total_trajectories,
    )

    rows = run_trajectories_loop(
        entries=items,
        id_fn=lambda entry: entry.item_id,
        seeds=seeds,
        controller_factory=controller_factory,
        trajectory_config=trajectory_config,
        agent=agent,
        judge=judge,
        trajectory_runner=run_ergo_math_trajectory,
        trajectories_path=trajectories_path,
        completed_by_tid=completed_by_tid,
    )

    report = analyze_ergo_math_screening(rows)
    report["config"] = run_config

    report_path = output_dir / "ergo_math_screening_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    (output_dir / "ergo_math_screening_report.md").write_text(_render_markdown(report))
    logger.info(
        "final_turn_success.mean={:.4f} (n={}) report written to {}",
        report["final_turn_success"]["mean"],
        report["final_turn_success"]["n"],
        output_dir,
    )

    return report


def _render_markdown(report: dict[str, Any]) -> str:
    final = report["final_turn_success"]
    q1 = report["new_q1_escalation"]
    q3 = report["new_q3_autocorrelation"]
    diag = report["diagnostics"]
    t_test = q1["t_test_mean_slope_vs_zero"]
    lines = [
        "# ERGO/Laban sharded-GSM8K minimal authority-check report",
        "",
        "## Primary metric: final-turn task success (turn == num_shards for that item)",
        f"- mean: {final['mean']:.4f} (n={final['n']})",
        "",
        "## new-Q1 analog: progressive change, continuous (per-trajectory OLS slope of y_task_success vs turn)",
        f"- items with negative slope: {q1['n_negative_slope_items']}/{q1['n_items']}",
        f"- items with positive slope: {q1['n_positive_slope_items']}/{q1['n_items']}",
        f"- one-sample t-test of mean per-item slope vs 0: t={t_test['t']:.4f}, p={t_test['p']:.4f}, df={t_test['df']}",
        "",
        "## new-Q3 analog: turn-to-turn inertia (y_task_success_t vs y_task_success_{t+1}, OLS)",
        f"- slope: {q3['slope']:.4f}, r={q3['r']:.4f}, p={q3['p_value']:.4f} (n_pairs={q3['n_pairs']})",
        "",
        "## Diagnostics",
        f"- refusal_rate: {diag['refusal_rate']:.4f}",
        f"- judge_parse_failure_rate: {diag['judge_parse_failure_rate']:.4f}",
        "- success by turn:",
    ]
    for turn, stats in sorted(report["success_by_turn"].items()):
        lines.append(f"  - turn {turn}: mean={stats['mean']:.4f} (n={stats['n']})")
    return "\n".join(lines) + "\n"
