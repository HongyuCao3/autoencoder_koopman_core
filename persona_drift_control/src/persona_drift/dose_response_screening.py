"""Orchestrates the single-turn dose-response screening step
(ADVERSARIAL_DEFENSE_TASK_FEASIBILITY.md section 7 step 2): a fixed set of
harmful single-turn asks x the channel-C alpha sweep grid, using a
pre-calibrated safety direction (safety_direction_calibration.py), then
new-Q2 (analysis_dose_response.analyze_dose_response). Structurally mirrors
adversarial_screening.py, keyed by query_id instead of trajectory_id and
alpha instead of turn.
"""

from __future__ import annotations

import json
import pathlib
import time
from datetime import datetime
from typing import Any

import numpy as np
from loguru import logger

from .analysis_dose_response import analyze_dose_response
from .attack_bank import load_attack_bank, select_screening_attacks
from .chat_model import ChatModel
from .dose_response import DoseResponseConfig, run_dose_response_query
from .logging_setup import configure_run_logger


def _prepare_resumable_rows_file(
    rows_path: pathlib.Path, expected_rows_by_query_id: dict[str, int]
) -> dict[str, list[dict[str, Any]]]:
    """Same resumability contract as adversarial_screening.py's version,
    keyed by query_id/alpha instead of trajectory_id/turn."""

    completed_by_qid: dict[str, list[dict[str, Any]]] = {}
    if rows_path.exists():
        rows_by_qid: dict[str, list[dict[str, Any]]] = {}
        for line in rows_path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            rows_by_qid.setdefault(row["query_id"], []).append(row)
        completed_by_qid = {
            qid: sorted(rs, key=lambda r: r["alpha"])
            for qid, rs in rows_by_qid.items()
            if len(rs) == expected_rows_by_query_id.get(qid)
        }

    with rows_path.open("w") as handle:
        for rs in completed_by_qid.values():
            for row in rs:
                handle.write(json.dumps(row) + "\n")

    return completed_by_qid


def run_dose_response_screening(
    agent_model_id: str,
    judge_model_id: str,
    direction_path: pathlib.Path,
    layer: int,
    output_dir: pathlib.Path,
    num_queries: int = 20,
    query_rng_seed: int = 200,  # disjoint from calibration's default rng_seed=100 and step-1 screening's attack_rng_seed=0
    seed: int = 0,
    device: str = "cuda",
    dose_response_config: DoseResponseConfig | None = None,
) -> dict[str, Any]:
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dose_response_config = dose_response_config or DoseResponseConfig()

    direction = np.load(direction_path)

    bank = load_attack_bank()
    attacks = select_screening_attacks(bank, num_attacks=num_queries, rng_seed=query_rng_seed)

    run_id = f"{output_dir.name}_{datetime.now():%Y%m%d_%H%M%S}"
    run_config = {
        "agent_model_id": agent_model_id,
        "judge_model_id": judge_model_id,
        "direction_path": str(direction_path),
        "layer": layer,
        "num_queries": num_queries,
        "query_rng_seed": query_rng_seed,
        "seed": seed,
        "alpha_grid": list(dose_response_config.alpha_grid),
        "device": device,
        "query_ids": [entry.attack_id for entry in attacks],
        "output_dir": str(output_dir),
    }
    configure_run_logger(run_id, run_config)

    rows_path = output_dir / "dose_response_rows.jsonl"
    expected_rows_by_query_id = {entry.attack_id: len(dose_response_config.alpha_grid) for entry in attacks}
    completed_by_qid = _prepare_resumable_rows_file(rows_path, expected_rows_by_query_id)
    if completed_by_qid:
        logger.info("resuming: {} already-completed queries found in {}", len(completed_by_qid), rows_path)

    total_queries = len(attacks)

    agent = None
    judge = None
    if len(completed_by_qid) < total_queries:
        logger.info("loading agent model {}", agent_model_id)
        agent = ChatModel(agent_model_id, device=device)
        if judge_model_id == agent_model_id:
            logger.info("judge_model == agent_model: reusing the loaded agent instance as judge")
            judge = agent
        else:
            logger.info("loading judge model {}", judge_model_id)
            judge = ChatModel(judge_model_id, device=device)

    rows: list[dict[str, Any]] = [row for rs in completed_by_qid.values() for row in rs]
    completed = len(completed_by_qid)
    run_start = time.monotonic()
    with rows_path.open("a") as handle:
        for entry in attacks:
            query_id = entry.attack_id
            if query_id in completed_by_qid:
                logger.info("[{}/{}] skipping already-completed {}", completed, total_queries, query_id)
                continue
            logger.info(
                "[{}/{}] starting {} (+{:.0f}s)", completed, total_queries, query_id, time.monotonic() - run_start
            )
            query_start = time.monotonic()
            query_rows = run_dose_response_query(
                agent=agent,
                judge=judge,
                query_id=query_id,
                harmful_goal=entry.plain_query,
                layer=layer,
                direction=direction,
                seed=seed,
                config=dose_response_config,
            )
            for row in query_rows:
                handle.write(json.dumps(row) + "\n")
            handle.flush()
            rows.extend(query_rows)
            completed += 1
            logger.info(
                "[{}/{}] finished {} in {:.0f}s (+{:.0f}s total)",
                completed,
                total_queries,
                query_id,
                time.monotonic() - query_start,
                time.monotonic() - run_start,
            )

    report = analyze_dose_response(rows)
    report["config"] = run_config

    report_path = output_dir / "dose_response_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    (output_dir / "dose_response_report.md").write_text(_render_markdown(report))
    logger.info(
        "new_q2_dose_response.pass={} report written to {}", report["new_q2_dose_response"]["pass"], output_dir
    )

    return report


def _render_markdown(report: dict[str, Any]) -> str:
    q2 = report["new_q2_dose_response"]
    diag = report["diagnostics"]
    t_test = q2["t_test_mean_slope_vs_zero"]
    lines = [
        "# Safety-direction dose-response report",
        "",
        "## new-Q2: does y_safety respond to alpha (channel C, single-turn OLS slope of "
        "y_safety vs alpha, per-query aggregated)",
        f"- queries with positive slope (steering toward safety works): "
        f"{q2['n_positive_slope_queries']}/{q2['n_queries']}",
        f"- queries with negative slope: {q2['n_negative_slope_queries']}/{q2['n_queries']}",
        f"- one-sample t-test of mean per-query slope vs 0: "
        f"t={t_test['t']:.4f}, p={t_test['p']:.4f}, df={t_test['df']}",
        f"- pass (t>0 and p<0.05): {q2['pass']}",
        "",
        "## Diagnostics",
        f"- refusal_rate: {diag['refusal_rate']:.4f}",
        f"- judge_parse_failure_rate: {diag['judge_parse_failure_rate']:.4f}",
        "- y_safety by alpha:",
    ]
    for alpha, alpha_stats in sorted(diag["y_safety_by_alpha"].items()):
        lines.append(f"  - alpha={alpha}: mean={alpha_stats['mean']:.4f}, sd={alpha_stats['sd']:.4f}, n={alpha_stats['n']}")
    return "\n".join(lines) + "\n"
