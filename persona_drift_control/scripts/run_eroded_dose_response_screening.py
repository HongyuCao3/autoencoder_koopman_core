#!/usr/bin/env python3
"""CLI for the eroded-context dose-response screening variant
(docs/experiments/dose_response_pilot.md "下一步" candidate 1). Requires
both a calibrated direction (calibrate_safety_direction.py) and step-1
screening's trajectories.jsonl (run_adversarial_screening.py, job 15399715)
to already exist. Must be run where torch/transformers are installed and a
GPU (or patient CPU) is available - see environment/setup_env.sh. Writes
dose_response_rows.jsonl, dose_response_report.json, and
dose_response_report.md under --output-dir.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from persona_drift.chat_model import GenerationConfig  # noqa: E402
from persona_drift.dose_response import ALPHA_GRID, DoseResponseConfig  # noqa: E402
from persona_drift.eroded_dose_response_screening import run_eroded_dose_response_screening  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-model", default="Qwen/Qwen3-4B")
    parser.add_argument(
        "--judge-model",
        default=None,
        help="defaults to --agent-model (self-judging, no second model load)",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--direction-path",
        type=pathlib.Path,
        default=pathlib.Path("outputs/safety_direction/safety_direction.npy"),
    )
    parser.add_argument(
        "--layer",
        type=int,
        required=True,
        help="must match the layer calibrate_safety_direction.py computed --direction-path at",
    )
    parser.add_argument(
        "--trajectories-path",
        type=pathlib.Path,
        default=pathlib.Path("outputs/adversarial_screening/trajectories.jsonl"),
        help="step-1 screening output to source eroded conversation contexts from",
    )
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("outputs/dose_response_eroded"))
    parser.add_argument(
        "--context-seed",
        type=int,
        default=0,
        help="which of step-1's two seeds per attack to use as the query set (avoids counting both as independent)",
    )
    parser.add_argument(
        "--max-final-turn-y-safety",
        type=float,
        default=0.8,
        help="drop trajectories whose recorded final-turn y_safety is already above this (no headroom left)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--agent-max-new-tokens", type=int, default=256)
    parser.add_argument("--alpha-grid", type=float, nargs="+", default=list(ALPHA_GRID))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    judge_model = args.judge_model or args.agent_model
    dose_response_config = DoseResponseConfig(
        agent_gen=GenerationConfig(max_new_tokens=args.agent_max_new_tokens),
        alpha_grid=tuple(args.alpha_grid),
    )
    report = run_eroded_dose_response_screening(
        agent_model_id=args.agent_model,
        judge_model_id=judge_model,
        direction_path=args.direction_path,
        layer=args.layer,
        trajectories_path=args.trajectories_path,
        output_dir=args.output_dir,
        context_seed=args.context_seed,
        max_final_turn_y_safety=args.max_final_turn_y_safety,
        seed=args.seed,
        device=args.device,
        dose_response_config=dose_response_config,
    )
    print(f"new_q2_dose_response.pass={report['new_q2_dose_response']['pass']}")
    print(f"report written to {args.output_dir}/dose_response_report.md")


if __name__ == "__main__":
    main()
