#!/usr/bin/env python3
"""Hydra-driven CLI for the adversarial-defense screening pilot
(ADVERSARIAL_DEFENSE_TASK_FEASIBILITY.md section 7 step 1), with
enable_thinking as a first-class, swappable config group (`generation`)
instead of a hardcoded default -- see
docs/experiments/adversarial_screening_thinking_pilot.md for why this rerun
exists and conf/adversarial_screening.yaml for how the output directory /
log run_id are kept from colliding between generation regimes.

Usage:
    python scripts/run_adversarial_screening_hydra.py                    # no_thinking baseline (default)
    python scripts/run_adversarial_screening_hydra.py generation=thinking
    python scripts/run_adversarial_screening_hydra.py generation=thinking num_attacks=20 seeds=[0,1]

Must be run where torch/transformers are installed and a GPU (or patient
CPU) is available -- see environment/setup_env.sh. Writes trajectories.jsonl,
adversarial_screening_report.json, and adversarial_screening_report.md under
Hydra's per-run output directory (printed at startup by Hydra itself).
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import hydra  # noqa: E402
from hydra.core.hydra_config import HydraConfig  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

from persona_drift.adversarial_screening import run_adversarial_screening  # noqa: E402
from persona_drift.attack_trajectory import AttackTrajectoryConfig  # noqa: E402
from persona_drift.chat_model import GenerationConfig  # noqa: E402

CONFIG_DIR = str(pathlib.Path(__file__).resolve().parents[1] / "conf")


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="adversarial_screening")
def main(cfg: DictConfig) -> None:
    output_dir = pathlib.Path(HydraConfig.get().runtime.output_dir)
    judge_model = cfg.judge_model or cfg.agent_model
    trajectory_config = AttackTrajectoryConfig(
        agent_gen=GenerationConfig(max_new_tokens=cfg.generation.agent_max_new_tokens),
    )
    report = run_adversarial_screening(
        agent_model_id=cfg.agent_model,
        judge_model_id=judge_model,
        output_dir=output_dir,
        num_attacks=cfg.num_attacks,
        seeds=tuple(cfg.seeds),
        attack_rng_seed=cfg.attack_rng_seed,
        device=cfg.device,
        trajectory_config=trajectory_config,
        enable_thinking=cfg.generation.enable_thinking,
    )
    print(f"generation={cfg.generation.name} enable_thinking={cfg.generation.enable_thinking}")
    print(f"new_q1_escalation.pass={report['new_q1_escalation']['pass']}")
    print(f"new_q3_autocorrelation.pass={report['new_q3_autocorrelation']['pass']}")
    print(f"report written to {output_dir}/adversarial_screening_report.md")


if __name__ == "__main__":
    main()
