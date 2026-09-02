#!/usr/bin/env python3
"""CLI for the sycophancy-drift screening pilot
(docs/task/SYCOPHANCY_DRIFT_TASK_FEASIBILITY.md section 8 step 2). Must be
run where torch/transformers are installed and a GPU (or patient CPU) is
available - see environment/setup_env.sh. Writes trajectories.jsonl,
sycophancy_screening_report.json, and sycophancy_screening_report.md under
--output-dir. Mirrors scripts/run_adversarial_screening.py.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from persona_drift.chat_model import GenerationConfig  # noqa: E402
from persona_drift.sycophancy_screening import run_sycophancy_screening  # noqa: E402
from persona_drift.sycophancy_trajectory import SycophancyTrajectoryConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-model", default="Qwen/Qwen3-4B")
    parser.add_argument(
        "--judge-model",
        default=None,
        help="defaults to --agent-model (self-judging, no second model load)",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("outputs/sycophancy_screening"))
    parser.add_argument("--num-items", type=int, default=20)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--item-rng-seed", type=int, default=0)
    parser.add_argument("--agent-max-new-tokens", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    judge_model = args.judge_model or args.agent_model
    trajectory_config = SycophancyTrajectoryConfig(
        agent_gen=GenerationConfig(max_new_tokens=args.agent_max_new_tokens),
    )
    report = run_sycophancy_screening(
        agent_model_id=args.agent_model,
        judge_model_id=judge_model,
        output_dir=args.output_dir,
        num_items=args.num_items,
        seeds=tuple(args.seeds),
        item_rng_seed=args.item_rng_seed,
        device=args.device,
        trajectory_config=trajectory_config,
    )
    print(f"new_q1_escalation.pass={report['new_q1_escalation']['pass']}")
    print(f"new_q3_autocorrelation.pass={report['new_q3_autocorrelation']['pass']}")
    print(f"discrete_flip_events.pass={report['discrete_flip_events']['pass']}")
    print(f"report written to {args.output_dir}/sycophancy_screening_report.md")


if __name__ == "__main__":
    main()
