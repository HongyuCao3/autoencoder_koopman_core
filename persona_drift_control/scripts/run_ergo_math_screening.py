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
from persona_drift.ergo_controllers import (  # noqa: E402
    FixedTAndLastController,
    RandSchedTAndLastController,
)
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
    "fixed_t_and_last",
    "randsched_t_and_last",
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
        "--reset-mode",
        choices=("overwrite", "append"),
        default="overwrite",
        help="docs/experiments/two_task_success_plan.md section 2 E1: 'overwrite' (default) is the "
        "existing byte-for-byte-unchanged behavior; 'append' keeps the accumulated history instead "
        "of replacing it with the reset's consolidated message.",
    )
    parser.add_argument(
        "--prompt-profile",
        choices=("legacy", "upstream"),
        default="legacy",
        help="docs/experiments/ergo_fidelity_restoration_plan.md section 3.1: 'legacy' (default) is "
        "the existing byte-for-byte-unchanged prompt structure (no system message; the answer "
        "format instruction concatenated onto every user turn); 'upstream' delivers that "
        "instruction once in a system message and leaves the per-turn stimuli bare, matching "
        "microsoft/lost_in_conversation's shape.",
    )
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
    parser.add_argument(
        "--koopman-objective",
        choices=("terminal", "sum"),
        default="terminal",
        help="--controller ergo_koopman_mpc only: passed through to "
        "load_ergo_koopman_mpc_controller/ErgoKoopmanMPCController (docs/experiments/"
        "two_task_success_plan.md section 12.2 B4 / section 13.6). Default here is the "
        "pre-registered 'terminal' (C1/B4) -- a CLI default is fine to be the pre-registered "
        "value since it is visible on the command line/sbatch; the *class's own* default "
        "stays 'sum' (section 13.6) so a bare ErgoKoopmanMPCController() still reproduces "
        "the existing outputs/ergo_math_phaseC_mpc arm unchanged.",
    )
    parser.add_argument(
        "--koopman-forced-last-reset",
        action="store_true",
        help="--controller ergo_koopman_mpc only: mode-B (docs/experiments/two_task_success_plan.md "
        "section 2 E4 item 3 / section 12.2 B3) -- the trajectory's own last turn is an "
        "unconditional reset (u=1), not a decision; --remind-budget must then be >= 2 for a "
        "non-degenerate mid-trajectory reset to also be available.",
    )
    parser.add_argument(
        "--koopman-pad-short-history",
        action="store_true",
        help="--controller ergo_koopman_mpc only: passed through to "
        "load_ergo_koopman_mpc_controller/ErgoKoopmanMPCController.pad_short_history. Default "
        "here is False (matches the class's own default, section 13.6); the new mode-A/mode-B "
        "arms pass this explicitly (C1's pre-registered True) in their sbatch invocation rather "
        "than relying on an implicit class default.",
    )
    parser.add_argument(
        "--fixed-t",
        type=int,
        default=None,
        help="required for --controller fixed_t_and_last: the fixed absolute turn t for the "
        "first reset (the item's own last turn is always the second, forced reset -- see "
        "ergo_controllers.FixedTAndLastController). A dedicated flag rather than reusing "
        "--fixed-schedule-turns[0]: that flag's nargs='+' already means 'reset on every one of "
        "these turns' for --controller fixed_schedule (k = len(turns)), so silently taking only "
        "its first element here and dropping any rest would be exactly the kind of silent "
        "degradation section 12.2/13.2 keep flagging elsewhere in this plan.",
    )
    return parser.parse_args()


def build_controller_factory(args: argparse.Namespace):
    """Extracted from `main()` (E4c, docs/experiments/two_task_success_plan.md
    section 2 E4c) purely so tests can drive the CLI's controller-construction
    branches (including the `--reset-mode append` name-suffix wrapper) with a
    hand-built `argparse.Namespace`, without going through `main()`'s GPU-bound
    `run_ergo_math_screening` call. No behavior change from what used to be
    inlined in `main()`."""

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
        default_num_shards = max(shards_by_item.values())

        def controller_factory(seed: int, entry_id: str = "") -> FixedScheduleController:
            # entry_id=="" only happens on ergo_math_screening.py's one naming-only
            # call (controller_factory(seeds[0]).name for the run_id string); the
            # per-trajectory calls always pass a real item_id. Same fallback pattern
            # as the random_schedule branch above.
            num_shards = shards_by_item.get(entry_id, default_num_shards)
            return FixedScheduleController(
                turns=(num_shards,),
                name="fixed_schedule_t_last",
            )

    elif args.controller == "fixed_t_and_last":
        # B2/E4c (docs/experiments/two_task_success_plan.md section 12.2 B2
        # / section 2 E4 item 6): mode-B's fixed-schedule opponent -- reset
        # on absolute turn --fixed-t AND this item's own last turn.
        if args.fixed_t is None:
            raise ValueError("--fixed-t is required for --controller fixed_t_and_last")
        shards_by_item = {item.item_id: len(item.shards) for item in load_ergo_math_bank()}
        default_num_shards = max(shards_by_item.values())

        def controller_factory(seed: int, entry_id: str = "") -> FixedTAndLastController:
            num_shards = shards_by_item.get(entry_id, default_num_shards)
            return FixedTAndLastController(t=args.fixed_t, num_shards=num_shards)

    elif args.controller == "randsched_t_and_last":
        # B2/E4c: mode-B's random-allocation opponent -- reset on a turn
        # drawn uniformly from {1, ..., num_shards - 1} AND this item's own
        # last turn. 13.2's B2-continuation hard requirement: the seed
        # passed in MUST be the per-(seed, entry_id) `_excitation_seed`, not
        # the raw trajectory-level `seed` -- otherwise every item sharing a
        # `num_shards` value draws the identical `t` (all items with the
        # same shard count collapse onto one shared schedule, silently
        # defeating P1's random-allocation opponent), exactly the failure
        # mode `random_schedule` above and `random_excite`
        # (controller_cli.py) already guard against the same way.
        shards_by_item = {item.item_id: len(item.shards) for item in load_ergo_math_bank()}
        default_num_shards = max(shards_by_item.values())

        def controller_factory(seed: int, entry_id: str = "") -> RandSchedTAndLastController:
            num_shards = shards_by_item.get(entry_id, default_num_shards)
            return RandSchedTAndLastController(
                num_shards=num_shards,
                seed=_excitation_seed(seed, entry_id),
            )

    elif args.controller == "ergo_koopman_mpc":
        if args.koopman_model_path is None:
            raise ValueError("--koopman-model-path is required for --controller ergo_koopman_mpc")
        if args.koopman_forced_last_reset and (args.remind_budget is None or args.remind_budget < 2):
            # Opus's follow-up finding on this same E4c task (docs/experiments/
            # two_task_success_plan.md section 2 E4 item 3 / section 12.2 B3):
            # forced_last_reset reserves exactly one unit of the budget for
            # the unconditional last-turn reset on every turn before the
            # last (ergo_koopman_mpc.py's _remaining_budget). At k=1 that
            # reservation leaves 0 spendable before the last turn, so the
            # planner's own (freely chosen) first reset can never fire --
            # the arm silently collapses into `fixed_last` (identical
            # decisions on all 6 verified turns) while still carrying the
            # `mpc_..._forcedlast_...` name, the same silent-degradation
            # shape as B3/the Phase C `mpc` == `fixed_t2` collapse. Caught
            # only by Opus's independent recompute, not by any gate in this
            # module -- so it is stopped here, at the CLI, rather than left
            # to the controller (which should not have to guess the
            # caller's intent about what counts as "still meaningfully
            # mode B").
            raise ValueError(
                "--koopman-forced-last-reset requires --remind-budget >= 2: forced_last_reset "
                "reserves 1 spend for the unconditional last-turn reset, so at k=1 (remind_budget "
                "None or 1) the planner's own first reset can never fire and this arm collapses "
                "into fixed_last (same decisions on every turn) while still being named "
                "mpc_..._forcedlast_..."
            )
        # C1 (docs/experiments/two_task_success_plan.md section 12.3): the
        # pre-registered objective/pad_short_history values must be visible
        # in the arm name, not just passed silently -- built here rather
        # than left to load_ergo_koopman_mpc_controller's own "koopman_mpc"
        # default name.
        mpc_name = f"mpc_{args.koopman_objective}"
        if args.koopman_forced_last_reset:
            mpc_name += "_forcedlast"
        if args.koopman_pad_short_history:
            mpc_name += "_pad"
        mpc_controller = load_ergo_koopman_mpc_controller(
            model_path=args.koopman_model_path,
            model_key=args.koopman_model_key,
            nu=args.koopman_nu,
            mu=args.koopman_mu,
            horizon=args.koopman_horizon,
            repeat_penalty=args.koopman_repeat_penalty,
            remind_budget=args.remind_budget,
            y_col=args.koopman_y_col,
            name=mpc_name,
            objective=args.koopman_objective,
            forced_last_reset=args.koopman_forced_last_reset,
            pad_short_history=args.koopman_pad_short_history,
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
    if args.prompt_profile == "upstream":
        # docs/experiments/ergo_fidelity_restoration_plan.md section 3.1: suffix
        # excitation_design/controller.name with "_up" for the same reason
        # --reset-mode append suffixes with "_append" -- two prompt profiles of
        # the same controller must never collide under one name. Applied before
        # the append suffix so names read e.g. "constant_remind_up_append";
        # same double-suffix guard, since controllers may be singletons.
        base_profile_factory = controller_factory

        def controller_factory(seed: int, entry_id: str = ""):
            controller = base_profile_factory(seed, entry_id)
            if not controller.name.endswith("_up") and "_up_" not in controller.name:
                controller.name = f"{controller.name}_up"
            return controller

    if args.reset_mode == "append":
        # docs/experiments/two_task_success_plan.md section 2 E1 item 2:
        # suffix excitation_design/controller.name with "_append" so the
        # overwrite- and append-mode products of the same controller never
        # collide under the same name. Controllers may be a singleton
        # returned by reference on every call (e.g. ergo_koopman_mpc), so
        # guard against double-suffixing on repeated controller_factory calls.
        base_controller_factory = controller_factory

        def controller_factory(seed: int, entry_id: str = ""):
            controller = base_controller_factory(seed, entry_id)
            if not controller.name.endswith("_append"):
                controller.name = f"{controller.name}_append"
            return controller

    return controller_factory


def main() -> None:
    args = parse_args()
    trajectory_config = ErgoMathTrajectoryConfig(
        agent_gen=GenerationConfig(max_new_tokens=args.agent_max_new_tokens),
        reset_mode=args.reset_mode,
        prompt_profile=args.prompt_profile,
    )
    controller_factory = build_controller_factory(args)

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
