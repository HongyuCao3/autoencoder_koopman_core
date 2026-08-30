"""Orchestrates the pre-experiment signal screening pilot
(DATA_COLLECTION_PROTOCOL.md section 7): a handful of prompts x seeds x
turns, in both the zero_control and excite_iid conditions, then the three
gate-question analysis.

This intentionally runs a SUPERSET of the protocol's literal "5 prompt x 2
seed x 16 turns" (10 trajectories): gate question 1 needs u==0 trajectories
and gate questions 2/3 need u_remind to vary, so each (prompt, seed) pair is
run under BOTH conditions here (20 trajectories total), roughly doubling the
protocol's "~1 hour" estimate. See screening_report.md's header for the
actual wall-clock this run took.
"""

from __future__ import annotations

import json
import pathlib
import random
import time
from datetime import datetime
from typing import Any

from loguru import logger

from .analysis import analyze_screening
from .chat_model import ChatModel
from .control import Controller, RandomExciteController, ZeroControlController
from .logging_setup import configure_run_logger
from .prompt_bank import load_prompt_bank, select_screening_prompts
from .selfchat import TOPICS, TrajectoryConfig, run_trajectory

CONDITIONS = ("zero_control", "excite_iid")


def _make_controller(condition: str, seed: int, trajectory_config: TrajectoryConfig) -> Controller:
    if condition == "zero_control":
        return ZeroControlController()
    if condition == "excite_iid":
        return RandomExciteController(p=trajectory_config.excite_p_remind, seed=seed)
    raise ValueError(f"unknown condition: {condition!r}")


def _prepare_resumable_trajectories_file(
    trajectories_path: pathlib.Path, expected_rows_per_trajectory: int
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str]]:
    """Lets a killed/resubmitted job (e.g. hit --time before finishing a big
    prompt sweep, see docs/experiments/signal_screening_pilot.md) continue
    instead of discarding already-completed trajectories.

    Reads any existing `trajectories_path`, keeps only trajectory_ids with
    exactly `expected_rows_per_trajectory` rows ("completed"), and rewrites
    the file to contain just those (dropping any partial trajectory from a
    run that was killed mid-trajectory, so a retry of that trajectory_id
    doesn't end up with duplicate/interleaved turns on top of the old
    partial rows). Returns (completed_rows_by_trajectory_id,
    recorded_topic_by_prompt_id) -- the latter so a resumed run reuses the
    same topic for any not-yet-run seed/condition under a prompt that
    already has some completed trajectories (topic is chosen once per
    prompt and shared across its seeds/conditions)."""

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
            if len(rs) == expected_rows_per_trajectory
        }

    with trajectories_path.open("w") as handle:
        for rs in completed_by_tid.values():
            for row in rs:
                handle.write(json.dumps(row) + "\n")

    topic_by_prompt_id = {rs[0]["system_prompt_id"]: rs[0]["topic"] for rs in completed_by_tid.values()}
    return completed_by_tid, topic_by_prompt_id


def run_screening(
    agent_model_id: str,
    user_model_id: str,
    output_dir: pathlib.Path,
    num_prompts: int = 5,
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
            "excite_p_remind": trajectory_config.excite_p_remind,
            "user_mode": trajectory_config.user_mode,
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
    remaining_needs_live = trajectory_config.user_mode == "live" and any(
        f"{entry.prompt_id}__seed{seed}__{condition}" not in completed_by_tid
        for entry in prompts
        for seed in seeds
        for condition in CONDITIONS
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
            logger.info("user_mode={} for all remaining trajectories: skipping user-simulator model load", trajectory_config.user_mode)

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
                    controller = _make_controller(condition, seed, trajectory_config)
                    trajectory_start = time.monotonic()
                    trajectory_rows = run_trajectory(
                        agent=agent,
                        user_sim=user_sim,
                        entry=entry,
                        controller=controller,
                        seed=seed,
                        topic=topic,
                        trajectory_id=trajectory_id,
                        topic_split="screening",
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

    report = analyze_screening(rows)
    report["config"] = run_config

    report_path = output_dir / "screening_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    (output_dir / "screening_report.md").write_text(_render_markdown(report))
    logger.info("overall_pass={} report written to {}", report["overall_pass"], output_dir)

    return report


def _render_markdown(report: dict[str, Any]) -> str:
    q1, q2, q3 = report["q1_drift_exists"], report["q2_input_effective"], report["q3_inertia"]
    q1_trend = report["q1_drift_trend"]
    diag = report["diagnostics"]
    t_test = q1_trend["t_test_mean_slope_vs_zero"]
    lines = [
        "# Pre-experiment signal screening report",
        "",
        f"Overall: {'PASS' if report['overall_pass'] else 'FAIL'}",
        "",
        "## Q1 drift exists (zero_control, turn 1 vs last)",
        f"- mean drop: {q1['mean_drop_turn1_to_last']:.4f}",
        f"- threshold (2 x mean y_probe_sd): {q1['threshold']:.4f}",
        f"- pass: {q1['pass']} (n_trajectories={q1['n_trajectories']})",
        "",
        "## Q1 drift trend (supplementary: full-sequence OLS slope, per-prompt aggregated)",
        f"- prompts with negative slope: {q1_trend['n_negative_slope_prompts']}/{q1_trend['n_prompts']}",
        f"- one-sample t-test of mean per-prompt slope vs 0: "
        f"t={t_test['t']:.4f}, p={t_test['p']:.4f}, df={t_test['df']}",
        "",
        "## Q2 input effective (excite_iid, u_remind_t vs y_probe_{t+1})",
        f"- mean y_probe_next | u=1: {q2['mean_y_probe_next_given_u1']:.4f}",
        f"- mean y_probe_next | u=0: {q2['mean_y_probe_next_given_u0']:.4f}",
        f"- diff: {q2['diff']:.4f}, threshold: {q2['threshold']:.4f}",
        f"- pass: {q2['pass']} (n_pairs={q2['n_pairs']})",
        "",
        "## Q3 inertia (excite_iid, u_remind_t vs y_probe_{t+2}, OLS)",
        f"- slope: {q3['slope_u_on_y_lag2']:.4f}, p={q3['p_value']:.4f}, r={q3['r']:.4f}",
        f"- pass (p<0.05): {q3['pass']} (n_pairs={q3['n_pairs']})",
        "",
        "## Diagnostics",
        f"- refusal_rate: {diag['refusal_rate']:.4f}",
        f"- scorer_failure_rate: {diag['scorer_failure_rate']:.4f}",
        f"- saturated_prompt_ids (zero y_probe variance all run, likely a degenerate scorer): "
        f"{diag['saturated_prompt_ids'] or 'none'}",
        "- y_probe by category:",
    ]
    for category, stats in diag["y_probe_by_category"].items():
        lines.append(f"  - {category}: mean={stats['mean']:.4f}, sd={stats['sd']:.4f}")
    return "\n".join(lines) + "\n"
