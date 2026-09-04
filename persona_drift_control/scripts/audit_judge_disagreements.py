#!/usr/bin/env python3
"""Print the raw text behind the largest self-judge / independent-judge
disagreements, so the paired numbers from compare_safety_judge_runs.py can
be settled by reading the replies instead of by assuming which judge is
right.

This is the step the sycophancy line's comparison treated as decisive: its
statistics only showed the two judges disagreed one way; what turned that
into "the self-judge is missing capitulations" was reading four of the
disputed turns against the item's ground truth
(docs/experiments/sycophancy_screening_pilot.md). The defense line's
disagreement runs the *other* way -- the independent judge scores higher --
so the same audit answers a different question: is the self-judge
over-penalizing its own replies, or is the independent judge waving through
replies that do advance the harmful goal?

Prints, per case: the underlying harmful goal, the turn's attacker query,
both judges' scores and raw outputs, and the reply (truncated). CPU-only.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from persona_drift.judge_bias import pair_rows  # noqa: E402
from persona_drift.modeling.dataset import load_trajectories  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm-dir", default="outputs/koopman_defense_phaseJ_budget1_koopman")
    parser.add_argument("--rejudge-subdir", default="rejudge_qwen3_4b_instruct_2507")
    parser.add_argument("--n", type=int, default=6, help="cases to print, largest disagreement first")
    parser.add_argument("--min-gap", type=float, default=0.75, help="only cases at least this far apart")
    parser.add_argument("--reply-chars", type=int, default=1200)
    parser.add_argument(
        "--direction",
        choices=("independent_higher", "independent_lower", "both"),
        default="both",
        help="which way the disagreement goes; 'independent_lower' is the rarer direction and the "
        "one that would mean the self-judge is missing real compliance",
    )
    parser.add_argument("--turn", type=int, default=None, help="restrict to one turn")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    arm = pathlib.Path(args.arm_dir)
    self_rows = load_trajectories(arm / "trajectories.jsonl")
    indep_rows = load_trajectories(arm / args.rejudge_subdir / "trajectories.jsonl")
    by_key = {(r["trajectory_id"], int(r["turn"])): r for r in self_rows}
    indep_by_key = {(r["trajectory_id"], int(r["turn"])): r for r in indep_rows}

    pairs, diagnostics = pair_rows(self_rows, indep_rows)
    if diagnostics["n_text_mismatch"]:
        raise SystemExit("text mismatches present -- not a paired comparison, refusing to audit")

    disputed = [p for p in pairs if abs(p["y_indep"] - p["y_self"]) >= args.min_gap]
    if args.direction == "independent_higher":
        disputed = [p for p in disputed if p["y_indep"] > p["y_self"]]
    elif args.direction == "independent_lower":
        disputed = [p for p in disputed if p["y_indep"] < p["y_self"]]
    if args.turn is not None:
        disputed = [p for p in disputed if p["turn"] == args.turn]
    disputed.sort(key=lambda p: (-abs(p["y_indep"] - p["y_self"]), p["trajectory_id"], p["turn"]))
    print(f"{len(disputed)} cases with |gap| >= {args.min_gap} out of {len(pairs)} paired rows in {arm}\n")

    for pair in disputed[: args.n]:
        key = (pair["trajectory_id"], pair["turn"])
        row, indep = by_key[key], indep_by_key[key]
        print("=" * 100)
        print(f"{key[0]}  turn {key[1]}  u_remind={row['u_remind']}")
        print(f"  harmful goal : {row['plain_query']}")
        print(f"  turn question: {row['attacker_query']}")
        print(f"  self judge   : y={pair['y_self']:.2f}  raw={row['judge_raw_output']!r}")
        print(f"  independent  : y={pair['y_indep']:.2f}  raw={indep['judge_raw_output']!r}")
        reply = row["agent_message"]
        print(f"  reply ({len(reply)} chars):\n{reply[: args.reply_chars]}")
        if len(reply) > args.reply_chars:
            print(f"  ... [{len(reply) - args.reply_chars} more chars]")
        print()


if __name__ == "__main__":
    main()
