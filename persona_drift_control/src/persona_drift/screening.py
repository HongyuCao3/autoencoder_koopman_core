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
from typing import Any

from .analysis import analyze_screening
from .chat_model import ChatModel
from .prompt_bank import load_prompt_bank, select_screening_prompts
from .selfchat import TOPICS, TrajectoryConfig, run_trajectory

CONDITIONS = ("zero_control", "excite_iid")


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

    agent = ChatModel(agent_model_id, device=device)
    user_sim = ChatModel(user_model_id, device=device)

    topic_rng = random.Random(prompt_rng_seed)
    rows: list[dict[str, Any]] = []
    trajectories_path = output_dir / "trajectories.jsonl"
    with trajectories_path.open("w") as handle:
        for entry in prompts:
            topic = topic_rng.choice(TOPICS)
            for seed in seeds:
                for condition in CONDITIONS:
                    trajectory_id = f"{entry.prompt_id}__seed{seed}__{condition}"
                    trajectory_rows = run_trajectory(
                        agent=agent,
                        user_sim=user_sim,
                        entry=entry,
                        condition=condition,
                        seed=seed,
                        topic=topic,
                        trajectory_id=trajectory_id,
                        topic_split="screening",
                        config=trajectory_config,
                    )
                    for row in trajectory_rows:
                        handle.write(json.dumps(row) + "\n")
                    rows.extend(trajectory_rows)

    report = analyze_screening(rows)
    report["config"] = {
        "agent_model_id": agent_model_id,
        "user_model_id": user_model_id,
        "num_prompts": num_prompts,
        "seeds": list(seeds),
        "conditions": list(CONDITIONS),
        "trajectory_config": {
            "num_turns": trajectory_config.num_turns,
            "probe_repeats": trajectory_config.probe_repeats,
            "excite_p_remind": trajectory_config.excite_p_remind,
        },
        "prompt_ids": [entry.prompt_id for entry in prompts],
    }

    report_path = output_dir / "screening_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    (output_dir / "screening_report.md").write_text(_render_markdown(report))

    return report


def _render_markdown(report: dict[str, Any]) -> str:
    q1, q2, q3 = report["q1_drift_exists"], report["q2_input_effective"], report["q3_inertia"]
    diag = report["diagnostics"]
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
        "- y_probe by category:",
    ]
    for category, stats in diag["y_probe_by_category"].items():
        lines.append(f"  - {category}: mean={stats['mean']:.4f}, sd={stats['sd']:.4f}")
    return "\n".join(lines) + "\n"
