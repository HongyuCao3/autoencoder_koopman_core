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
import json
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
    KoopmanMPCController,
    RandomExciteController,
    ThresholdController,
    ZeroControlController,
)
from persona_drift.modeling.dataset import ReducedStateConfig  # noqa: E402
from persona_drift.modeling.koopman import abs_sign_extra_features, no_extra_features, surrogate_from_arrays  # noqa: E402

CONTROLLER_CHOICES = ("zero_control", "constant_remind", "threshold", "random_excite", "koopman_mpc")
_EXTRA_FEATURES_FNS = {"arx": no_extra_features, "richer_abs_sign": abs_sign_extra_features}


def _load_koopman_mpc_controller(
    model_path: pathlib.Path,
    model_key: str,
    nu: int,
    mu: int,
    horizon: int,
    repeat_penalty: float,
) -> KoopmanMPCController:
    report = json.loads(model_path.read_text())
    fit = report[model_key]
    surrogate = surrogate_from_arrays(
        fit["A"],
        fit["B"],
        fit["b"],
        fit["C"],
        state_dim=ReducedStateConfig(nu=nu, mu=mu).state_dim,
        extra_features_fn=_EXTRA_FEATURES_FNS[model_key],
    )
    return KoopmanMPCController(
        surrogate=surrogate,
        state_config=ReducedStateConfig(nu=nu, mu=mu),
        horizon=horizon,
        repeat_penalty=repeat_penalty,
    )


def _make_controller_factory(
    name: str,
    threshold_y_min: float,
    random_excite_p: float,
    koopman_mpc_controller: KoopmanMPCController | None,
) -> Callable[[int], Controller]:
    if name == "zero_control":
        return lambda seed: ZeroControlController()
    if name == "constant_remind":
        return lambda seed: ConstantRemindController()
    if name == "threshold":
        return lambda seed: ThresholdController(y_min=threshold_y_min)
    if name == "random_excite":
        return lambda seed: RandomExciteController(p=random_excite_p, seed=seed)
    if name == "koopman_mpc":
        # Stateless given a fixed fitted surrogate -- safe to hand out the
        # same instance to every trajectory (unlike RandomExciteController,
        # there's no per-trajectory RNG state to keep independent).
        return lambda seed: koopman_mpc_controller
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
        _load_koopman_mpc_controller(
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
    controller_factory = _make_controller_factory(
        args.controller, args.threshold_y_min, args.random_excite_p, koopman_mpc_controller
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
