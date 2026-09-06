#!/usr/bin/env python3
"""CLI for the ERGO/Laban sharded-GSM8K minimal executor-authority check
(docs/feasibility/ERGO_MULTITURN_RELIABILITY_FEASIBILITY.md's step 2):
does a simplified "reset" actuator (consolidate every shard revealed so far
into one fresh message, control.py's generic zero_control/constant_remind/
fixed_schedule controllers reused for the reset/no-reset decision) move
`y_task_success` on this project's own model, before any further engineering
(entropy readout, more tasks, upstream's live user-simulator reveal logic).

Only zero_control/constant_remind/fixed_schedule are exposed -- same
reasoning as run_mc_sycophancy_defended_screening.py: nothing past a
minimal authority check is warranted before this gate itself passes.

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
from persona_drift.ergo_math_screening import run_ergo_math_screening  # noqa: E402
from persona_drift.ergo_math_trajectory import ErgoMathTrajectoryConfig  # noqa: E402

CONTROLLER_CHOICES = ("zero_control", "constant_remind", "fixed_schedule")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agent-model", default="Qwen/Qwen3-4B")
    parser.add_argument("--judge-model", default=None, help="unused (no model-based judging in this domain); "
                        "kept for CLI-shape consistency with the other run_*_screening.py scripts")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--controller", choices=CONTROLLER_CHOICES, required=True)
    parser.add_argument("--num-items", type=int, default=20)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--item-rng-seed", type=int, default=0)
    parser.add_argument(
        "--item-ids",
        nargs="+",
        default=None,
        help="replay this exact set of item ids instead of a random --num-items/--item-rng-seed sample "
        "(e.g. to reuse the same items across the zero_control/constant_remind pair)",
    )
    parser.add_argument("--agent-max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--fixed-schedule-turns",
        type=int,
        nargs="+",
        default=None,
        help="required for --controller fixed_schedule",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trajectory_config = ErgoMathTrajectoryConfig(
        agent_gen=GenerationConfig(max_new_tokens=args.agent_max_new_tokens),
    )
    controller_factory = make_controller_factory(
        args.controller,
        threshold_y_min=0.7,
        koopman_mpc_controller=None,
        fixed_schedule_turns=tuple(args.fixed_schedule_turns) if args.fixed_schedule_turns else None,
    )
    report = run_ergo_math_screening(
        agent_model_id=args.agent_model,
        judge_model_id=args.agent_model,  # no separate judge model needed, see ergo_math_judge.py
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
    print(f"final_turn_success.mean={report['final_turn_success']['mean']:.4f} (n={report['final_turn_success']['n']})")
    print(f"report written to {args.output_dir}/ergo_math_screening_report.md")


if __name__ == "__main__":
    main()
