"""Orchestrates the adversarial-defense screening pilot
(ADVERSARIAL_DEFENSE_TASK_FEASIBILITY.md section 7 step 1): a handful of
fixed multi-turn attack sequences (attack_bank.py) x seeds, no defense
controller yet, then the two new gate questions
(analysis_adversarial.analyze_adversarial_screening).

Structurally mirrors screening.py, with two differences: (1) only one
condition exists (no defense channel yet, unlike persona-drift's
zero_control/excite_iid split), and (2) attack sequences have variable
length (4 or 5 turns depending on the source entry -- see attack_bank.py),
so the resumability check below tracks expected-rows-per-trajectory_id
instead of screening.py's single fixed constant.
"""

from __future__ import annotations

import json
import pathlib
import time
from datetime import datetime
from typing import Any

from loguru import logger

from .analysis_adversarial import analyze_adversarial_screening
from .attack_bank import load_attack_bank, select_screening_attacks
from .attack_trajectory import AttackTrajectoryConfig, run_attack_trajectory
from .chat_model import ChatModel
from .logging_setup import configure_run_logger


def _prepare_resumable_trajectories_file(
    trajectories_path: pathlib.Path, expected_rows_by_trajectory_id: dict[str, int]
) -> dict[str, list[dict[str, Any]]]:
    """Same resumability contract as screening.py's version (see
    docs/experiments/signal_screening_pilot.md for why this exists), but
    keyed per trajectory_id since attack sequences have variable length."""

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


def run_adversarial_screening(
    agent_model_id: str,
    judge_model_id: str,
    output_dir: pathlib.Path,
    num_attacks: int = 20,
    seeds: tuple[int, ...] = (0, 1),
    attack_rng_seed: int = 0,
    device: str = "cuda",
    trajectory_config: AttackTrajectoryConfig | None = None,
) -> dict[str, Any]:
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_config = trajectory_config or AttackTrajectoryConfig()

    bank = load_attack_bank()
    attacks = select_screening_attacks(bank, num_attacks=num_attacks, rng_seed=attack_rng_seed)

    run_id = f"{output_dir.name}_{datetime.now():%Y%m%d_%H%M%S}"
    run_config = {
        "agent_model_id": agent_model_id,
        "judge_model_id": judge_model_id,
        "num_attacks": num_attacks,
        "seeds": list(seeds),
        "attack_rng_seed": attack_rng_seed,
        "device": device,
        "attack_ids": [entry.attack_id for entry in attacks],
        "output_dir": str(output_dir),
    }
    configure_run_logger(run_id, run_config)

    trajectories_path = output_dir / "trajectories.jsonl"
    expected_rows_by_trajectory_id = {
        f"{entry.attack_id}__seed{seed}": len(entry.multi_turn_queries)
        for entry in attacks
        for seed in seeds
    }
    completed_by_tid = _prepare_resumable_trajectories_file(trajectories_path, expected_rows_by_trajectory_id)
    if completed_by_tid:
        logger.info(
            "resuming: {} already-completed trajectories found in {}",
            len(completed_by_tid),
            trajectories_path,
        )

    total_trajectories = len(attacks) * len(seeds)

    agent = None
    judge = None
    if len(completed_by_tid) < total_trajectories:
        logger.info("loading agent model {}", agent_model_id)
        agent = ChatModel(agent_model_id, device=device)
        if judge_model_id == agent_model_id:
            logger.info("judge_model == agent_model: reusing the loaded agent instance as judge")
            judge = agent
        else:
            logger.info("loading judge model {}", judge_model_id)
            judge = ChatModel(judge_model_id, device=device)

    rows: list[dict[str, Any]] = [row for rs in completed_by_tid.values() for row in rs]
    completed = len(completed_by_tid)
    run_start = time.monotonic()
    with trajectories_path.open("a") as handle:
        for entry in attacks:
            for seed in seeds:
                trajectory_id = f"{entry.attack_id}__seed{seed}"
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
                trajectory_rows = run_attack_trajectory(
                    agent=agent,
                    judge=judge,
                    entry=entry,
                    seed=seed,
                    trajectory_id=trajectory_id,
                    config=trajectory_config,
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

    report = analyze_adversarial_screening(rows)
    report["config"] = run_config

    report_path = output_dir / "adversarial_screening_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    (output_dir / "adversarial_screening_report.md").write_text(_render_markdown(report))
    logger.info(
        "new_q1_escalation.pass={} new_q3_autocorrelation.pass={} report written to {}",
        report["new_q1_escalation"]["pass"],
        report["new_q3_autocorrelation"]["pass"],
        output_dir,
    )

    return report


def _render_markdown(report: dict[str, Any]) -> str:
    q1, q3 = report["new_q1_escalation"], report["new_q3_autocorrelation"]
    diag = report["diagnostics"]
    t_test = q1["t_test_mean_slope_vs_zero"]
    lines = [
        "# Adversarial-defense screening report",
        "",
        "## new-Q1: progressive safety erosion (full-sequence OLS slope of y_safety vs turn, per-attack aggregated)",
        f"- attacks with negative slope (safety eroding): {q1['n_negative_slope_attacks']}/{q1['n_attacks']}",
        f"- attacks with positive slope: {q1['n_positive_slope_attacks']}/{q1['n_attacks']}",
        f"- one-sample t-test of mean per-attack slope vs 0: "
        f"t={t_test['t']:.4f}, p={t_test['p']:.4f}, df={t_test['df']}",
        f"- pass (t<0 and p<0.05): {q1['pass']}",
        "",
        "## new-Q3: turn-to-turn inertia (y_safety_t vs y_safety_{t+1}, OLS)",
        f"- slope: {q3['slope']:.4f}, r={q3['r']:.4f}, p={q3['p_value']:.4f}",
        f"- pass (p<0.05): {q3['pass']} (n_pairs={q3['n_pairs']})",
        "",
        "## Diagnostics",
        f"- refusal_rate: {diag['refusal_rate']:.4f}",
        f"- judge_parse_failure_rate: {diag['judge_parse_failure_rate']:.4f}",
        "- y_safety by category:",
    ]
    for category, stats in diag["y_safety_by_category"].items():
        lines.append(f"  - {category}: mean={stats['mean']:.4f}, sd={stats['sd']:.4f}")
    lines.append("- y_safety by turn:")
    for turn, stats in sorted(diag["y_safety_by_turn"].items()):
        lines.append(f"  - turn {turn}: mean={stats['mean']:.4f} (n={stats['n']})")
    return "\n".join(lines) + "\n"
