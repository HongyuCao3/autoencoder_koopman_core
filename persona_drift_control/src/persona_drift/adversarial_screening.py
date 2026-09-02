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
from datetime import datetime
from typing import Any, Callable

from loguru import logger

from .analysis_adversarial import analyze_adversarial_screening
from .attack_bank import load_attack_bank, select_attacks_by_id, select_screening_attacks
from .attack_trajectory import AttackTrajectoryConfig, run_attack_trajectory
from .chat_model import ChatModel
from .control import Controller, ZeroControlController
from .logging_setup import configure_run_logger
from .screening_common import load_agent_and_judge, prepare_resumable_trajectories_file, run_trajectories_loop


def run_adversarial_screening(
    agent_model_id: str,
    judge_model_id: str,
    output_dir: pathlib.Path,
    num_attacks: int = 20,
    seeds: tuple[int, ...] = (0, 1),
    attack_rng_seed: int = 0,
    device: str = "cuda",
    trajectory_config: AttackTrajectoryConfig | None = None,
    enable_thinking: bool = False,
    controller_factory: Callable[[int, str], Controller] | None = None,
    attack_ids: list[str] | None = None,
) -> dict[str, Any]:
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_config = trajectory_config or AttackTrajectoryConfig()
    # A *factory* (one fresh Controller per trajectory, given that
    # trajectory's own seed), not a single shared Controller instance --
    # mirrors screening.py's `_make_controller(condition, seed, ...)`. This
    # matters for stateful controllers like RandomExciteController: a single
    # shared instance's RNG would run continuously across trajectories,
    # which breaks both per-trajectory reproducibility from its own seed and
    # this function's resumability (a resumed run would draw a different
    # sequence than an uninterrupted one). Every existing call site (no
    # `controller_factory` argument) is exactly the "无控制" screening this
    # module's docstring describes -- see
    # docs/experiments/koopman_defense_pilot.md.
    controller_factory = controller_factory or (lambda seed, entry_id="": ZeroControlController())

    bank = load_attack_bank()
    # An explicit attack_ids list (e.g. Phase C's held-out split from a
    # Koopman identification run) takes precedence over the random
    # rng_seed-based sample -- see docs/experiments/koopman_defense_pilot.md
    # for why Phase E's closed-loop validation needs this rather than
    # relying on a different attack_rng_seed to merely reduce overlap odds.
    attacks = (
        select_attacks_by_id(bank, attack_ids)
        if attack_ids is not None
        else select_screening_attacks(bank, num_attacks=num_attacks, rng_seed=attack_rng_seed)
    )

    # One throwaway construction just to read .name for the run_id/config
    # below, before the real per-trajectory instances are built inside the
    # loop -- harmless for every Controller in control.py (construction has
    # no side effects; RandomExciteController's actual per-trajectory RNGs
    # are still freshly seeded independently of this peek).
    controller_name = controller_factory(seeds[0]).name
    # think{0,1} and the controller name in the run id (not just a timestamp)
    # make both ablations self-describing straight from the logs/ filename,
    # on top of the timestamp already making every run's log file distinct
    # on its own -- see docs/experiments/adversarial_screening_thinking_pilot.md
    # and docs/experiments/koopman_defense_pilot.md.
    run_id = f"{output_dir.name}_think{int(enable_thinking)}_{controller_name}_{datetime.now():%Y%m%d_%H%M%S}"
    run_config = {
        "agent_model_id": agent_model_id,
        "judge_model_id": judge_model_id,
        "num_attacks": num_attacks,
        "seeds": list(seeds),
        "attack_rng_seed": attack_rng_seed,
        "device": device,
        "enable_thinking": enable_thinking,
        "controller": controller_name,
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
    completed_by_tid = prepare_resumable_trajectories_file(trajectories_path, expected_rows_by_trajectory_id)
    if completed_by_tid:
        logger.info(
            "resuming: {} already-completed trajectories found in {}",
            len(completed_by_tid),
            trajectories_path,
        )

    total_trajectories = len(attacks) * len(seeds)
    agent, judge = load_agent_and_judge(
        ChatModel,
        agent_model_id,
        judge_model_id,
        device,
        enable_thinking,
        needed=len(completed_by_tid) < total_trajectories,
    )

    rows = run_trajectories_loop(
        entries=attacks,
        id_fn=lambda entry: entry.attack_id,
        seeds=seeds,
        controller_factory=controller_factory,
        trajectory_config=trajectory_config,
        agent=agent,
        judge=judge,
        trajectory_runner=run_attack_trajectory,
        trajectories_path=trajectories_path,
        completed_by_tid=completed_by_tid,
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
