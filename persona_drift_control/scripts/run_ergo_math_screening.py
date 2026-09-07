#!/usr/bin/env python3
"""CLI for the ERGO/Laban sharded-GSM8K minimal executor-authority check
(docs/feasibility/ERGO_MULTITURN_RELIABILITY_FEASIBILITY.md's step 2):
does a simplified "reset" actuator (consolidate every shard revealed so far
into one fresh message, control.py's generic zero_control/constant_remind/
fixed_schedule controllers reused for the reset/no-reset decision) move
`y_task_success` on this project's own model, before any further engineering
(entropy readout, more tasks, upstream's live user-simulator reveal logic).

Originally only zero_control/constant_remind/fixed_schedule were exposed --
same reasoning as run_mc_sycophancy_defended_screening.py: nothing past a
minimal authority check was warranted before that gate itself passed. That
gate has now passed twice (20-item pilot + 60-item expansion, see
docs/experiments/ergo_multiturn_reliability_pilot.md), so --controller
random_excite is added for the open-loop-excitation data collection a
Koopman fit needs (same role as run_defended_screening.py's Phase B) --
`u_reset` drawn i.i.d. Bernoulli(p) each turn instead of the all-0/all-1
extremes the authority check used.

Must be run where torch/transformers are installed and a GPU (or patient
CPU) is available - see environment/setup_env.sh.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from persona_drift.chat_model import GenerationConfig  # noqa: E402
from persona_drift.control import FixedScheduleController, RandomScheduleController  # noqa: E402
from persona_drift.controller_cli import _excitation_seed, make_controller_factory  # noqa: E402
from persona_drift.ergo_koopman_mpc import load_ergo_koopman_mpc_controller  # noqa: E402
from persona_drift.ergo_math_bank import load_ergo_math_bank  # noqa: E402
from persona_drift.ergo_math_screening import run_ergo_math_screening  # noqa: E402
from persona_drift.ergo_math_trajectory import ErgoMathTrajectoryConfig  # noqa: E402

CONTROLLER_CHOICES = (
    "zero_control",
    "constant_remind",
    "fixed_schedule",
    "random_excite",
    "random_schedule",
    "ergo_koopman_mpc",
    "fixed_last",
)


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
    parser.add_argument(
        "--random-excite-p",
        type=float,
        default=None,
        help="required for --controller random_excite: Bernoulli(p) probability of u_reset=1 each turn",
    )
    parser.add_argument(
        "--random-schedule-spend-prob",
        type=float,
        default=1.0,
        help="--controller random_schedule only: probability of spending the single reset "
        "(1.0 = p100, matches signal_resolution_plan.md section 4.3's equal-cost arm)",
    )
    parser.add_argument(
        "--remind-budget",
        type=int,
        default=None,
        help="--controller random_schedule/fixed_schedule/ergo_koopman_mpc: max resets per trajectory "
        "(signal_resolution_plan.md section 4.2 fixes this at 1 for Phase C)",
    )
    parser.add_argument("--koopman-model-path", type=pathlib.Path, default=None, help="required for --controller ergo_koopman_mpc")
    parser.add_argument("--koopman-model-key", default="arx", help="--controller ergo_koopman_mpc only")
    parser.add_argument("--koopman-nu", type=int, default=1)
    parser.add_argument("--koopman-mu", type=int, default=1)
    parser.add_argument("--koopman-horizon", type=int, default=2)
    parser.add_argument("--koopman-repeat-penalty", type=float, default=0.0)
    parser.add_argument(
        "--koopman-y-col",
        default="closeness",
        help="--controller ergo_koopman_mpc only: state readout column (default closeness, F2's fitted "
        "readout; the reported metric stays final_turn_success regardless)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trajectory_config = ErgoMathTrajectoryConfig(
        agent_gen=GenerationConfig(max_new_tokens=args.agent_max_new_tokens),
    )

    # signal_resolution_plan.md section 4.3.1: `random_schedule` and
    # `ergo_koopman_mpc` need per-item customization (num_shards varies by
    # item) that controller_cli.make_controller_factory's fixed-turns-tuple
    # design can't express -- handled here as local branches instead of
    # touching that shared file (control.py/controller_cli.py must stay
    # untouched, see docs/experiments/signal_resolution_plan.md's
    # open-before-work rules).
    if args.controller == "random_schedule":
        shards_by_item = {item.item_id: len(item.shards) for item in load_ergo_math_bank()}
        default_num_shards = max(shards_by_item.values())

        def controller_factory(seed: int, entry_id: str = "") -> RandomScheduleController:
            num_shards = shards_by_item.get(entry_id, default_num_shards)
            return RandomScheduleController(
                turns=tuple(range(1, num_shards + 1)),
                spend_prob=args.random_schedule_spend_prob,
                seed=_excitation_seed(seed, entry_id),
            )

    elif args.controller == "fixed_last":
        # G4 (docs/experiments/measurement_validity_plan.md section 7.1): the
        # missing optimal fixed arm -- reset on each item's own last shard
        # turn, not an absolute turn like fixed_t1..t4.
        shards_by_item = {item.item_id: len(item.shards) for item in load_ergo_math_bank()}

        def controller_factory(seed: int, entry_id: str = "") -> FixedScheduleController:
            return FixedScheduleController(
                turns=(shards_by_item[entry_id],),
                name="fixed_schedule_t_last",
            )

    elif args.controller == "ergo_koopman_mpc":
        if args.koopman_model_path is None:
            raise ValueError("--koopman-model-path is required for --controller ergo_koopman_mpc")
        mpc_controller = load_ergo_koopman_mpc_controller(
            model_path=args.koopman_model_path,
            model_key=args.koopman_model_key,
            nu=args.koopman_nu,
            mu=args.koopman_mu,
            horizon=args.koopman_horizon,
            repeat_penalty=args.koopman_repeat_penalty,
            remind_budget=args.remind_budget,
            y_col=args.koopman_y_col,
        )

        def controller_factory(seed: int, entry_id: str = ""):
            return mpc_controller

    else:
        controller_factory = make_controller_factory(
            args.controller,
            threshold_y_min=0.7,
            koopman_mpc_controller=None,
            fixed_schedule_turns=tuple(args.fixed_schedule_turns) if args.fixed_schedule_turns else None,
            random_excite_p=args.random_excite_p,
            remind_budget=args.remind_budget,
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
