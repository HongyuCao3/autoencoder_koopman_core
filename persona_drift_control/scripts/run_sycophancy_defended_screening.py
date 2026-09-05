#!/usr/bin/env python3
"""CLI for the sycophancy-drift line's executor-authority check
(docs/feasibility/SYCOPHANCY_KOOPMAN_LOOP_FEASIBILITY.md section 5, step 2):
does `consistency_reminder.py`'s channel-A reminder actually move
y_consistency at all? `--controller zero_control` reproduces
scripts/run_sycophancy_screening.py's baseline exactly (same
ZeroControlController default); `--controller constant_remind` inserts the
reminder before every turn. Mirrors scripts/run_defended_screening.py
(the adversarial line's Phase A), reusing the same generic
controller_cli.make_controller_factory -- ConstantRemindController/
ZeroControlController only ever touch `u_remind`, they don't know which
reminder text sycophancy_trajectory.py's `reminder_fn` will substitute in.

Only zero_control/constant_remind are exposed here (not threshold/periodic/
koopman_mpc): those need a fitted state to react to or a model to plan
with, neither of which exists yet for this line -- premature before this
gate itself passes.

Must be run where torch/transformers are installed and a GPU (or patient
CPU) is available - see environment/setup_env.sh. Writes trajectories.jsonl,
sycophancy_screening_report.json, and sycophancy_screening_report.md under
--output-dir.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from persona_drift.chat_model import GenerationConfig  # noqa: E402
from persona_drift.controller_cli import make_controller_factory  # noqa: E402
from persona_drift.sycophancy_screening import run_sycophancy_screening  # noqa: E402
from persona_drift.sycophancy_trajectory import SycophancyTrajectoryConfig  # noqa: E402

CONTROLLER_CHOICES = ("zero_control", "constant_remind")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agent-model", default="Qwen/Qwen3-4B")
    parser.add_argument(
        "--judge-model",
        default=None,
        help="defaults to --agent-model (self-judging, no second model load) -- since "
        "sycophancy_screening_pilot.md found self-judging misses real capitulations, pass an "
        "independent model explicitly for anything that will be reported on",
    )
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
        "(e.g. the 11 ground-truth-clean items from "
        "persona_drift_control/scripts/audit_ground_truth_quality.py, recommended for this check so a "
        "no-effect result can't be blamed on the ground-truth issues docs/experiments/"
        "sycophancy_screening_pilot.md's audit found)",
    )
    parser.add_argument("--agent-max-new-tokens", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    judge_model = args.judge_model or args.agent_model
    trajectory_config = SycophancyTrajectoryConfig(
        agent_gen=GenerationConfig(max_new_tokens=args.agent_max_new_tokens),
    )
    controller_factory = make_controller_factory(args.controller, threshold_y_min=0.7, koopman_mpc_controller=None)
    report = run_sycophancy_screening(
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
    print(f"report written to {args.output_dir}/sycophancy_screening_report.md")


if __name__ == "__main__":
    main()
