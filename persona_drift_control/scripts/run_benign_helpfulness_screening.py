#!/usr/bin/env python3
"""CLI for the Phase F benign helpfulness-cost check
(docs/experiments/koopman_defense_pilot.md): deploys one of Phase E's
controller arms against the fixed 8-category MT-Bench benign session set
(benign_bank.py) instead of the adversarial attack bank. Controller
construction (including --controller koopman_mpc's fitted-surrogate loading)
is shared with scripts/run_defended_screening.py via controller_cli.py.

Must be run where torch/transformers are installed and a GPU (or patient
CPU) is available. Writes trajectories.jsonl, benign_screening_report.json,
and benign_screening_report.md under --output-dir.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from persona_drift.benign_screening import run_benign_screening  # noqa: E402
from persona_drift.benign_trajectory import BenignTrajectoryConfig  # noqa: E402
from persona_drift.chat_model import GenerationConfig  # noqa: E402
from persona_drift.controller_cli import (  # noqa: E402
    EXTRA_FEATURES_FNS,
    load_koopman_mpc_controller,
    load_koopman_mpc_interaction_controller,
    make_controller_factory,
)

CONTROLLER_CHOICES = ("zero_control", "constant_remind", "threshold", "periodic", "koopman_mpc", "koopman_mpc_interaction")
_EXTRA_FEATURES_FNS = EXTRA_FEATURES_FNS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-model", default="Qwen/Qwen3-4B")
    parser.add_argument("--judge-model", default=None, help="defaults to --agent-model")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--agent-max-new-tokens", type=int, default=256)
    parser.add_argument("--controller", choices=CONTROLLER_CHOICES, required=True)
    parser.add_argument("--threshold-y-min", type=float, default=0.7)
    parser.add_argument("--periodic-period", type=int, default=2, help="only used when --controller periodic")
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
    parser.add_argument(
        "--koopman-contemporaneous-v",
        action="store_true",
        help="the saved koopman_mpc/koopman_mpc_interaction model was fit with ReducedStateConfig.contemporaneous_v=True "
        "(the corrected v-alignment, docs/next_step_diagnosis.md 2026-09-02) -- must match how --koopman-model-path/"
        "--koopman-interaction-model-path was actually fit, or decisions silently use the wrong action slot again.",
    )
    parser.add_argument(
        "--koopman-interaction-model-path",
        type=pathlib.Path,
        default=pathlib.Path("outputs/koopman_case_study/interaction_model_report.json"),
        help="only used when --controller koopman_mpc_interaction",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    judge_model = args.judge_model or args.agent_model
    trajectory_config = BenignTrajectoryConfig(
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
            contemporaneous_v=args.koopman_contemporaneous_v,
        )
        if args.controller == "koopman_mpc"
        else None
    )
    koopman_mpc_interaction_controller = (
        load_koopman_mpc_interaction_controller(
            args.koopman_interaction_model_path,
            args.koopman_nu,
            args.koopman_mu,
            args.koopman_horizon,
            args.koopman_repeat_penalty,
            contemporaneous_v=args.koopman_contemporaneous_v,
        )
        if args.controller == "koopman_mpc_interaction"
        else None
    )
    controller_factory = make_controller_factory(
        args.controller,
        args.threshold_y_min,
        koopman_mpc_controller,
        periodic_period=args.periodic_period,
        koopman_mpc_interaction_controller=koopman_mpc_interaction_controller,
    )
    report = run_benign_screening(
        agent_model_id=args.agent_model,
        judge_model_id=judge_model,
        output_dir=args.output_dir,
        seeds=tuple(args.seeds),
        device=args.device,
        trajectory_config=trajectory_config,
        controller_factory=controller_factory,
    )
    print(f"controller={args.controller}")
    print(f"mean_y_help={report['diagnostics']['mean_y_help']:.4f}")
    print(f"refusal_rate={report['diagnostics']['refusal_rate']:.4f}")
    print(f"report written to {args.output_dir}/benign_screening_report.md")


if __name__ == "__main__":
    main()
