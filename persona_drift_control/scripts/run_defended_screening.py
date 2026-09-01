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
- Phase G (fixed-schedule baseline, docs/BASELINES.md's "周期性/事件触发重提醒"
  candidate): --controller periodic --periodic-period 2

Must be run where torch/transformers are installed and a GPU (or patient
CPU) is available - see environment/setup_env.sh. Writes trajectories.jsonl,
adversarial_screening_report.json, and adversarial_screening_report.md under
--output-dir.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from persona_drift.adversarial_screening import run_adversarial_screening  # noqa: E402
from persona_drift.attack_trajectory import AttackTrajectoryConfig  # noqa: E402
from persona_drift.chat_model import GenerationConfig  # noqa: E402
from persona_drift.controller_cli import (  # noqa: E402
    EXTRA_FEATURES_FNS,
    load_koopman_mpc_controller,
    make_controller_factory,
)

CONTROLLER_CHOICES = ("zero_control", "constant_remind", "threshold", "periodic", "random_excite", "koopman_mpc")
_EXTRA_FEATURES_FNS = EXTRA_FEATURES_FNS


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
    parser.add_argument("--periodic-period", type=int, default=2, help="only used when --controller periodic")
    parser.add_argument(
        "--attack-ids",
        nargs="+",
        default=None,
        help="replay this exact set of attack ids instead of a random --num-attacks/--attack-rng-seed sample "
        "(e.g. Phase E's held-out split from koopman_fit_report.json)",
    )
    parser.add_argument(
        "--koopman-model-path",
        type=pathlib.Path,
        default=pathlib.Path("outputs/koopman_defense_phaseB_random_excite/koopman_fit_report.json"),
        help="only used when --controller koopman_mpc",
    )
    parser.add_argument("--koopman-model-key", choices=list(_EXTRA_FEATURES_FNS), default="richer_abs_sign")
    parser.add_argument("--koopman-nu", type=int, default=1)
    parser.add_argument("--koopman-mu", type=int, default=2)
    parser.add_argument("--koopman-horizon", type=int, default=2)
    parser.add_argument("--koopman-repeat-penalty", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    judge_model = args.judge_model or args.agent_model
    trajectory_config = AttackTrajectoryConfig(
        agent_gen=GenerationConfig(max_new_tokens=args.agent_max_new_tokens),
    )
    koopman_mpc_controller = (
        load_koopman_mpc_controller(
            args.koopman_model_path,
            args.koopman_model_key,
            args.koopman_nu,
            args.koopman_mu,
            args.koopman_horizon,
            args.koopman_repeat_penalty,
        )
        if args.controller == "koopman_mpc"
        else None
    )
    controller_factory = make_controller_factory(
        args.controller,
        args.threshold_y_min,
        koopman_mpc_controller,
        random_excite_p=args.random_excite_p,
        periodic_period=args.periodic_period,
    )
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
        attack_ids=args.attack_ids,
    )
    print(f"controller={args.controller}")
    print(f"new_q1_escalation.pass={report['new_q1_escalation']['pass']}")
    print(f"new_q3_autocorrelation.pass={report['new_q3_autocorrelation']['pass']}")
    print(f"report written to {args.output_dir}/adversarial_screening_report.md")


if __name__ == "__main__":
    main()
