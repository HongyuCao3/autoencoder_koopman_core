#!/usr/bin/env python3
"""CLI for the MMLU-sycophancy line's executor-authority check (Phase A
analog): does consistency_reminder.py's channel-A reminder actually move
y_consistency on this (ground-truth-verified) item pool? Mirrors
scripts/run_sycophancy_defended_screening.py exactly -- see that script's
docstring for the reasoning; the only difference is the underlying item pool
and judge (mc_sycophancy_bank.py/mc_answer_judge.py instead of
sycophancy_bank.py/sycophancy_judge.py).

The earlier attempt at this check on the SYCON-Bench item pool
(docs/experiments/sycophancy_screening_pilot.md's "Phase A" section) came
back inconclusive -- underpowered (n=10-11 clean items) and confounded by
one item whose ground truth was itself in question. This run doesn't fix
the sample-size problem by itself, but it does remove the second
confound: mc_answer_judge.py's correctness check traces to MMLU's answer
key, not to a free-text field anything could have gotten wrong.

Only zero_control/constant_remind are exposed (not threshold/periodic/
koopman_mpc): same reasoning as run_sycophancy_defended_screening.py --
premature before this gate itself passes.

Must be run where torch/transformers are installed and a GPU (or patient
CPU) is available - see environment/setup_env.sh.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from persona_drift.chat_model import GenerationConfig  # noqa: E402
from persona_drift.controller_cli import make_controller_factory  # noqa: E402
from persona_drift.mc_sycophancy_screening import run_mc_sycophancy_screening  # noqa: E402
from persona_drift.mc_sycophancy_trajectory import MCSycophancyTrajectoryConfig  # noqa: E402

CONTROLLER_CHOICES = ("zero_control", "constant_remind")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agent-model", default="Qwen/Qwen3-4B")
    parser.add_argument("--judge-model", default=None, help="defaults to --agent-model")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--controller", choices=CONTROLLER_CHOICES, required=True)
    parser.add_argument("--num-items", type=int, default=30)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--item-rng-seed", type=int, default=0)
    parser.add_argument(
        "--item-ids",
        nargs="+",
        default=None,
        help="replay this exact set of item ids instead of a random --num-items/--item-rng-seed sample "
        "(e.g. to reuse the same items across the zero_control/constant_remind pair, or to reuse the "
        "60-item sample-expansion run's item set for a directly comparable executor-authority check)",
    )
    parser.add_argument("--agent-max-new-tokens", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    judge_model = args.judge_model or args.agent_model
    trajectory_config = MCSycophancyTrajectoryConfig(
        agent_gen=GenerationConfig(max_new_tokens=args.agent_max_new_tokens),
    )
    controller_factory = make_controller_factory(args.controller, threshold_y_min=0.7, koopman_mpc_controller=None)
    report = run_mc_sycophancy_screening(
        agent_model_id=args.agent_model,
        judge_model_id=judge_model,
        output_dir=args.output_dir,
        num_items=args.num_items,
        seeds=tuple(args.seeds),
        item_rng_seed=args.item_rng_seed,
        device=args.device,
        trajectory_config=trajectory_config,
        controller_factory=controller_factory,
        item_ids=args.item_ids,
    )
    print(f"controller={args.controller}")
    print(f"new_q1_escalation.pass={report['new_q1_escalation']['pass']}")
    print(f"new_q3_autocorrelation.pass={report['new_q3_autocorrelation']['pass']}")
    print(f"discrete_flip_events.pass={report['discrete_flip_events']['pass']}")
    print(f"report written to {args.output_dir}/mc_sycophancy_screening_report.md")


if __name__ == "__main__":
    main()
