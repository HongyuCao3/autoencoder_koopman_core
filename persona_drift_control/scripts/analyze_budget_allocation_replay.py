#!/usr/bin/env python3
"""Offline go/no-go check for the budget-constrained arm (Phase J,
docs/experiments/budget_constrained_defense_plan.md) BEFORE spending GPU on
a closed-loop run: under a k-reminder-per-trajectory budget, does the
v-aligned interaction Koopman MPC actually place its reminders at different
turns for different trajectories?

If it always spends on the same turn, it has degenerated into
`FixedScheduleController` with extra steps and the closed-loop comparison
against the fixed-schedule sweep cannot possibly be informative -- that is
exactly the kind of negative result that Phase I's `pad_short_history` check
established offline (turn2: 0/16) instead of paying for a GPU run to learn
it. Adaptivity here is a *necessary* condition for the closed-loop test to
be worth running, not a sufficient one for it to succeed.

Replays the policy over trajectories that were really collected under some
other controller: the `y_safety` readings are real, but the reminder history
feeding the lag slots is not what this policy would have produced, so
`--use-policy-actions` (default) substitutes the policy's own decisions into
the lag window and budget accounting, and `--use-real-actions` reproduces
the strict "real history" convention scripts/analyze_mu2_interaction_replay.py
used. Neither is a substitute for the closed-loop run -- a different reminder
pattern changes the agent's real downstream responses, which no replay can
show.

CPU-only, pure numpy -- no GPU needed.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from persona_drift.controller_cli import load_koopman_mpc_interaction_controller  # noqa: E402
from persona_drift.modeling.dataset import group_by_trajectory, load_trajectories  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--model-path",
        type=pathlib.Path,
        default=pathlib.Path("outputs/koopman_case_study/interaction_model_report_valigned.json"),
        help="the v-aligned interaction model Phase I used and Phase J's koopman arm will use",
    )
    parser.add_argument(
        "--replay-paths",
        type=pathlib.Path,
        nargs="+",
        default=[
            pathlib.Path("outputs/koopman_defense_phaseI_koopman_mpc_valigned/trajectories.jsonl"),
            pathlib.Path("outputs/koopman_defense_phaseG_periodic/trajectories.jsonl"),
        ],
        help="trajectory files to replay over; several are better, since each one's y_safety readings "
        "come from a different controller's state distribution",
    )
    parser.add_argument("--budget", type=int, default=1)
    parser.add_argument("--episode-length", type=int, default=5)
    parser.add_argument("--nu", type=int, default=1)
    parser.add_argument("--mu", type=int, default=2)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--repeat-penalty", type=float, default=0.0)
    parser.add_argument("--no-pad-short-history", dest="pad_short_history", action="store_false")
    parser.add_argument(
        "--use-real-actions",
        dest="use_policy_actions",
        action="store_false",
        help="feed the replayed trajectory's real u_remind into the lag window instead of the policy's own",
    )
    parser.add_argument(
        "--out-path",
        type=pathlib.Path,
        default=pathlib.Path("outputs/koopman_case_study/budget_allocation_replay_report.json"),
    )
    return parser.parse_args()


def _replay_trajectory(controller, rows: list[dict], use_policy_actions: bool) -> dict:
    """Walks one trajectory turn by turn, asking the controller what it would
    do given the history so far. Returns the turn its budget was spent on
    (None if never), plus the per-turn decisions."""

    history: list[dict] = []
    decisions = []
    spent_turns = []
    for row in sorted(rows, key=lambda r: r["turn"]):
        action = controller.next_u_remind(row["turn"], history)
        decisions.append({"turn": row["turn"], "y_probe_prev": history[-1]["y_probe"] if history else None, "action": action})
        if action:
            spent_turns.append(row["turn"])
        history.append({**row, "u_remind": action if use_policy_actions else row["u_remind"]})
    return {"spent_turns": spent_turns, "decisions": decisions}


def main() -> None:
    args = parse_args()
    controller = load_koopman_mpc_interaction_controller(
        args.model_path,
        args.nu,
        args.mu,
        args.horizon,
        args.repeat_penalty,
        contemporaneous_v=True,
        pad_short_history=args.pad_short_history,
        remind_budget=args.budget,
        episode_length=args.episode_length,
        name="koopman_mpc_interaction",
    )
    print(f"controller.name={controller.name} budget={args.budget} horizon={args.horizon} "
          f"episode_length={args.episode_length} pad_short_history={args.pad_short_history} "
          f"use_policy_actions={args.use_policy_actions}")

    report: dict = {"config": {k: str(v) for k, v in vars(args).items()}, "replays": {}}
    for replay_path in args.replay_paths:
        if not replay_path.exists():
            print(f"\n!! {replay_path} missing, skipped")
            continue
        by_tid = group_by_trajectory(load_trajectories(replay_path))
        results = {tid: _replay_trajectory(controller, rows, args.use_policy_actions) for tid, rows in by_tid.items()}

        spend_turn_counts = collections.Counter()
        n_unspent = 0
        for res in results.values():
            if res["spent_turns"]:
                spend_turn_counts[res["spent_turns"][0]] += 1
            else:
                n_unspent += 1
        n_traj = len(results)
        distinct = len(spend_turn_counts)

        print(f"\n=== {replay_path} (n_trajectories={n_traj}) ===")
        for turn in sorted(spend_turn_counts):
            print(f"  spends its budget on turn {turn}: {spend_turn_counts[turn]}/{n_traj}")
        if n_unspent:
            print(f"  never spends it at all: {n_unspent}/{n_traj}")
        print(f"  distinct spend turns = {distinct} -> "
              f"{'ADAPTIVE (not reducible to one fixed schedule)' if distinct > 1 or n_unspent else 'DEGENERATE (equivalent to fixed_schedule at that turn)'}")
        for tid in sorted(results):
            spent = results[tid]["spent_turns"]
            print(f"    {tid}: spends on turn {spent[0] if spent else '-'} "
                  f"(decisions: {[(d['turn'], d['action']) for d in results[tid]['decisions']]})")
        # Over-spend guard: the budget is enforced inside the controller, so a
        # violation here would be a bug in _remaining_budget, not a finding.
        assert all(len(res["spent_turns"]) <= args.budget for res in results.values()), "budget violated"

        report["replays"][str(replay_path)] = {
            "n_trajectories": n_traj,
            "spend_turn_counts": {str(k): v for k, v in sorted(spend_turn_counts.items())},
            "n_never_spends": n_unspent,
            "n_distinct_spend_turns": distinct,
            "per_trajectory": results,
        }

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nreport written to {args.out_path}")


if __name__ == "__main__":
    main()
