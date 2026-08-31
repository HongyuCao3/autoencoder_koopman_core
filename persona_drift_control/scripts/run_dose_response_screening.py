#!/usr/bin/env python3
"""CLI for the single-turn safety-direction dose-response screening step
(ADVERSARIAL_DEFENSE_TASK_FEASIBILITY.md section 7 step 2). Requires a
direction already calibrated by calibrate_safety_direction.py. Must be run
where torch/transformers are installed and a GPU (or patient CPU) is
available - see environment/setup_env.sh. Writes dose_response_rows.jsonl,
dose_response_report.json, and dose_response_report.md under --output-dir.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from persona_drift.chat_model import GenerationConfig  # noqa: E402
from persona_drift.dose_response import ALPHA_GRID, DoseResponseConfig  # noqa: E402
from persona_drift.dose_response_screening import run_dose_response_screening  # noqa: E402


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
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("outputs/dose_response"))
    parser.add_argument("--num-queries", type=int, default=20)
    parser.add_argument("--query-rng-seed", type=int, default=200)
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
    report = run_dose_response_screening(
        agent_model_id=args.agent_model,
        judge_model_id=judge_model,
        direction_path=args.direction_path,
        layer=args.layer,
        output_dir=args.output_dir,
        num_queries=args.num_queries,
        query_rng_seed=args.query_rng_seed,
        seed=args.seed,
        device=args.device,
        dose_response_config=dose_response_config,
    )
    print(f"new_q2_dose_response.pass={report['new_q2_dose_response']['pass']}")
    print(f"report written to {args.output_dir}/dose_response_report.md")


if __name__ == "__main__":
    main()
