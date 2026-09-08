#!/usr/bin/env python3
"""EK-A (docs/experiments/ergo_fidelity_restoration_plan.md section 4): the
counterfactual-branch identification arm.

For every item x seed this runs ONE base trajectory and, at every turn,
one extra turn generated from the byte-identical prefix under the opposite
action. Each turn therefore yields a same-prefix pair whose difference in
`closeness` is a direct measurement of that turn's one-step causal gain --
the quantity EK0 showed the observational `random_excite p=0.5` arm cannot
resolve (its endpoint MDE is +0.49 against an effect of +0.181).

Deliberately a separate script rather than a flag on
run_ergo_math_screening.py: that driver's resumability keys off
`expected_rows_by_trajectory_id`, and the counterfactual rows are
single-turn with their own trajectory_ids, so feeding them through it would
corrupt both the bookkeeping and every analyzer that groups
trajectories.jsonl by trajectory_id. The two row families go to two files.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from datetime import datetime

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from loguru import logger  # noqa: E402

from persona_drift.chat_model import ChatModel, GenerationConfig  # noqa: E402
from persona_drift.control import ConstantRemindController, ZeroControlController  # noqa: E402
from persona_drift.ergo_math_bank import load_ergo_math_bank, select_items_by_id, select_screening_items  # noqa: E402
from persona_drift.ergo_math_trajectory import (  # noqa: E402
    ErgoMathTrajectoryConfig,
    run_ergo_math_branch_trajectory,
)
from persona_drift.logging_setup import configure_run_logger  # noqa: E402
from persona_drift.screening_common import load_agent_and_judge  # noqa: E402

BASE_CONTROLLERS = {"zero_control": ZeroControlController, "constant_remind": ConstantRemindController}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--agent-model", default="Qwen/Qwen3-4B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-dir", type=pathlib.Path, required=True)
    p.add_argument("--base-controller", choices=sorted(BASE_CONTROLLERS), default="zero_control")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--num-items", type=int, default=45)
    p.add_argument("--item-rng-seed", type=int, default=0)
    p.add_argument("--item-ids", nargs="+", default=None,
                   help="explicit item ids; overrides --num-items/--item-rng-seed")
    p.add_argument("--reset-mode", choices=("overwrite", "append"), default="append")
    p.add_argument("--prompt-profile", choices=("legacy", "upstream"), default="upstream")
    p.add_argument("--agent-max-new-tokens", type=int, default=512)
    return p.parse_args()


def _load_done(path: pathlib.Path) -> set[str]:
    """Resume by base trajectory_id: a trajectory counts as done only when both
    its files already hold rows for it, so a crash between the two writes
    re-runs that trajectory instead of leaving a half pair."""
    if not path.exists():
        return set()
    ids = set()
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            ids.add(row.get("base_trajectory_id", row["trajectory_id"]))
    return ids


def main() -> None:
    args = parse_args()
    out = pathlib.Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    base_path = out / "trajectories.jsonl"
    pair_path = out / "counterfactual_pairs.jsonl"

    bank = load_ergo_math_bank()
    items = (
        select_items_by_id(bank, args.item_ids)
        if args.item_ids
        else select_screening_items(bank, num_items=args.num_items, rng_seed=args.item_rng_seed)
    )

    controller_name = f"branch_{args.base_controller}"
    if args.prompt_profile == "upstream":
        controller_name += "_up"
    if args.reset_mode == "append":
        controller_name += "_append"

    run_id = f"{out.name}_{controller_name}_{datetime.now():%Y%m%d_%H%M%S}"
    run_config = {
        "agent_model_id": args.agent_model,
        "seeds": list(args.seeds),
        "item_ids": [i.item_id for i in items],
        "base_controller": args.base_controller,
        "controller": controller_name,
        "reset_mode": args.reset_mode,
        "prompt_profile": args.prompt_profile,
        "agent_max_new_tokens": args.agent_max_new_tokens,
        "output_dir": str(out),
        "arm": "EK-A counterfactual branch",
    }
    configure_run_logger(run_id, run_config)
    (out / "run_config.json").write_text(json.dumps(run_config, indent=2))

    done = _load_done(pair_path) & _load_done(base_path)
    total = len(items) * len(args.seeds)
    logger.info("EK-A branch arm: {} items x {} seeds = {} trajectories; {} already done",
                len(items), len(args.seeds), total, len(done))

    config = ErgoMathTrajectoryConfig(
        agent_gen=GenerationConfig(max_new_tokens=args.agent_max_new_tokens),
        reset_mode=args.reset_mode,
        prompt_profile=args.prompt_profile,
    )
    agent, judge = load_agent_and_judge(
        ChatModel, args.agent_model, args.agent_model, args.device, False, needed=len(done) < total
    )

    n_pairs = 0
    start = time.monotonic()
    with base_path.open("a") as bh, pair_path.open("a") as ph:
        for i, entry in enumerate(items):
            for seed in args.seeds:
                tid = f"{entry.item_id}__seed{seed}"
                if tid in done:
                    logger.info("skipping already-completed {}", tid)
                    continue
                t0 = time.monotonic()
                controller = BASE_CONTROLLERS[args.base_controller]()
                controller.name = controller_name
                base_rows, cf_rows = run_ergo_math_branch_trajectory(
                    agent=agent, judge=judge, entry=entry, seed=seed,
                    trajectory_id=tid, config=config, run_id=run_id, controller=controller,
                )
                for r in base_rows:
                    bh.write(json.dumps(r) + "\n")
                for r in cf_rows:
                    ph.write(json.dumps(r) + "\n")
                bh.flush()
                ph.flush()
                n_pairs += len(cf_rows)
                logger.info("{} done in {:.0f}s ({} pairs, {} total, +{:.0f}s elapsed)",
                            tid, time.monotonic() - t0, len(cf_rows), n_pairs, time.monotonic() - start)

    print(f"base trajectories -> {base_path}")
    print(f"counterfactual pairs -> {pair_path}  ({n_pairs} pairs this run)")


if __name__ == "__main__":
    main()
