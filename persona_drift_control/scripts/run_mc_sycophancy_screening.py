#!/usr/bin/env python3
"""CLI for the MMLU-based sycophancy screening
(mc_sycophancy_bank.py/mc_answer_judge.py), which replaces
scripts/run_sycophancy_screening.py's SYCON-Bench item pool -- see
docs/experiments/sycophancy_screening_pilot.md's ground-truth audit and
resources/PROVENANCE.md's mmlu_sycophancy_mc.jsonl entry for why. Must be
run where torch/transformers are installed and a GPU (or patient CPU) is
available - see environment/setup_env.sh. Writes trajectories.jsonl,
mc_sycophancy_screening_report.json, and mc_sycophancy_screening_report.md
under --output-dir.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from persona_drift.chat_model import GenerationConfig  # noqa: E402
from persona_drift.mc_sycophancy_screening import run_mc_sycophancy_screening  # noqa: E402
from persona_drift.mc_sycophancy_trajectory import MCSycophancyTrajectoryConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-model", default="Qwen/Qwen3-4B")
    parser.add_argument(
        "--judge-model",
        default=None,
        help="defaults to --agent-model (self-judging). Unlike sycophancy_judge.py this judge only "
        "does answer-letter extraction (mostly via regex, an LLM call only on regex failure), not "
        "factual adjudication, so self-judging bias is a smaller concern here -- still recommended "
        "to pass an independent model for anything that will be reported on, for consistency with "
        "the rest of this project's default.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("outputs/mc_sycophancy_screening"))
    parser.add_argument("--num-items", type=int, default=60)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--item-rng-seed", type=int, default=0)
    parser.add_argument(
        "--item-ids",
        nargs="+",
        default=None,
        help="replay this exact set of item ids instead of a random --num-items/--item-rng-seed sample",
    )
    parser.add_argument("--agent-max-new-tokens", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    judge_model = args.judge_model or args.agent_model
    trajectory_config = MCSycophancyTrajectoryConfig(
        agent_gen=GenerationConfig(max_new_tokens=args.agent_max_new_tokens),
    )
    report = run_mc_sycophancy_screening(
        agent_model_id=args.agent_model,
        judge_model_id=judge_model,
        output_dir=args.output_dir,
        num_items=args.num_items,
        seeds=tuple(args.seeds),
        item_rng_seed=args.item_rng_seed,
        device=args.device,
        trajectory_config=trajectory_config,
        item_ids=args.item_ids,
    )
    print(f"new_q1_escalation.pass={report['new_q1_escalation']['pass']}")
    print(f"new_q3_autocorrelation.pass={report['new_q3_autocorrelation']['pass']}")
    print(f"discrete_flip_events.pass={report['discrete_flip_events']['pass']}")
    print(f"report written to {args.output_dir}/mc_sycophancy_screening_report.md")


if __name__ == "__main__":
    main()
