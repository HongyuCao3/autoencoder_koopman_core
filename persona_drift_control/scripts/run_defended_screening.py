#!/usr/bin/env python3
"""CLI for the adversarial-defense screening pilot with a defense
controller active (docs/experiments/koopman_defense_pilot.md) -- the
channel-A-style safety-reminder actuator (safety_reminder.py) driven by one
of control.py's `Controller` implementations, instead of the "无控制"
baseline `scripts/run_adversarial_screening.py` runs.

Used for:
- Phase A (executor authority check): --controller constant_remind
- Phase B (open-loop excitation for Koopman identification): --controller random_excite
- Phase E (closed-loop validation classical baseline): --controller threshold

Must be run where torch/transformers are installed and a GPU (or patient
CPU) is available - see environment/setup_env.sh. Writes trajectories.jsonl,
adversarial_screening_report.json, and adversarial_screening_report.md under
--output-dir.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Callable

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from persona_drift.adversarial_screening import run_adversarial_screening  # noqa: E402
from persona_drift.attack_trajectory import AttackTrajectoryConfig  # noqa: E402
from persona_drift.chat_model import GenerationConfig  # noqa: E402
from persona_drift.control import (  # noqa: E402
    ConstantRemindController,
    Controller,
    RandomExciteController,
    ThresholdController,
    ZeroControlController,
)

CONTROLLER_CHOICES = ("zero_control", "constant_remind", "threshold", "random_excite")


def _make_controller_factory(name: str, threshold_y_min: float, random_excite_p: float) -> Callable[[int], Controller]:
    if name == "zero_control":
        return lambda seed: ZeroControlController()
    if name == "constant_remind":
        return lambda seed: ConstantRemindController()
    if name == "threshold":
        return lambda seed: ThresholdController(y_min=threshold_y_min)
    if name == "random_excite":
        return lambda seed: RandomExciteController(p=random_excite_p, seed=seed)
    raise ValueError(f"unknown controller: {name!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-model", default="Qwen/Qwen3-4B")
    parser.add_argument(
        "--judge-model",
        default=None,
        help="defaults to --agent-model (self-judging, no second model load)",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--num-attacks", type=int, default=20)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--attack-rng-seed", type=int, default=0)
    parser.add_argument("--agent-max-new-tokens", type=int, default=256)
    parser.add_argument("--controller", choices=CONTROLLER_CHOICES, required=True)
    parser.add_argument("--threshold-y-min", type=float, default=0.7)
    parser.add_argument("--random-excite-p", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    judge_model = args.judge_model or args.agent_model
    trajectory_config = AttackTrajectoryConfig(
        agent_gen=GenerationConfig(max_new_tokens=args.agent_max_new_tokens),
    )
    controller_factory = _make_controller_factory(args.controller, args.threshold_y_min, args.random_excite_p)
    report = run_adversarial_screening(
        agent_model_id=args.agent_model,
        judge_model_id=judge_model,
        output_dir=args.output_dir,
        num_attacks=args.num_attacks,
        seeds=tuple(args.seeds),
        attack_rng_seed=args.attack_rng_seed,
        device=args.device,
        trajectory_config=trajectory_config,
        controller_factory=controller_factory,
    )
    print(f"controller={args.controller}")
    print(f"new_q1_escalation.pass={report['new_q1_escalation']['pass']}")
    print(f"new_q3_autocorrelation.pass={report['new_q3_autocorrelation']['pass']}")
    print(f"report written to {args.output_dir}/adversarial_screening_report.md")


if __name__ == "__main__":
    main()
