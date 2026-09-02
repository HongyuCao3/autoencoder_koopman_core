#!/usr/bin/env python3
"""Controller decision-latency benchmark: separate from token/insertion-count
cost (already exactly matched by Phase G's `period=2` choice, see
docs/experiments/koopman_defense_pilot.md), this measures the wall-clock cost
of the controller's OWN decision computation itself -- `koopman_mpc`'s
horizon-2 brute-force MPC forward simulation vs `periodic`'s trivial modulo
check vs `threshold`'s single comparison -- to make explicit that "periodic
is simpler" also holds on this axis, and to confirm (rather than assume) that
none of these decision costs are large enough to matter next to the LLM
inference calls that dominate a real screening job's wall-clock (Phase E/G
jobs ran ~11-16 minutes for 16 trajectories x 5 turns, i.e. seconds per turn
for generation+judging).

Replays real decision points from Phase E/G's already-recorded
`trajectories.jsonl` (offline, no GPU, no new LLM calls) so the history
lengths/content fed to each controller are realistic, not synthetic.

CPU-only. Run directly (no sbatch).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from persona_drift.control import KoopmanMPCController, PeriodicController, ThresholdController  # noqa: E402
from persona_drift.modeling.dataset import ReducedStateConfig, group_by_trajectory, load_trajectories  # noqa: E402
from persona_drift.modeling.koopman import abs_sign_extra_features, surrogate_from_arrays  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fit-report", type=pathlib.Path, default=pathlib.Path("outputs/koopman_defense_phaseB_random_excite/koopman_fit_report.json")
    )
    parser.add_argument(
        "--trajectories", type=pathlib.Path, default=pathlib.Path("outputs/koopman_defense_phaseE_koopman_mpc/trajectories.jsonl")
    )
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=500)
    parser.add_argument("--out-path", type=pathlib.Path, default=pathlib.Path("outputs/koopman_lstm_baseline/controller_decision_cost.json"))
    return parser.parse_args()


def build_decision_points(rows_path: pathlib.Path) -> list[tuple[int, list[dict]]]:
    arm_rows = group_by_trajectory(load_trajectories(rows_path))
    points = []
    for traj_rows in arm_rows.values():
        for i in range(len(traj_rows)):
            points.append((i + 1, traj_rows[:i]))
    return points


def time_controller(controller, decision_points: list[tuple[int, list[dict]]], repeats: int) -> dict:
    for turn, history in decision_points:
        controller.next_u_remind(turn, history)  # warm-up, not timed

    t0 = time.perf_counter()
    for _ in range(repeats):
        for turn, history in decision_points:
            controller.next_u_remind(turn, history)
    elapsed = time.perf_counter() - t0

    n_calls = repeats * len(decision_points)
    return {
        "n_decision_points": len(decision_points),
        "repeats": repeats,
        "total_seconds": elapsed,
        "microseconds_per_call": elapsed / n_calls * 1e6,
    }


def main() -> None:
    args = parse_args()
    decision_points = build_decision_points(args.trajectories)

    fit_report = json.loads(args.fit_report.read_text())
    cfg, model_report = fit_report["config"], fit_report["richer_abs_sign"]
    state_config = ReducedStateConfig(nu=cfg["nu"], mu=cfg["mu"])
    surrogate = surrogate_from_arrays(
        A=model_report["A"], B=model_report["B"], b=model_report["b"], C=model_report["C"],
        state_dim=state_config.state_dim, extra_features_fn=abs_sign_extra_features, ridge=cfg["ridge"],
    )

    controllers = {
        "koopman_mpc": KoopmanMPCController(surrogate=surrogate, state_config=state_config, horizon=args.horizon),
        "periodic": PeriodicController(period=2),
        "threshold": ThresholdController(y_min=0.7),
    }

    results = {name: time_controller(controller, decision_points, args.repeats) for name, controller in controllers.items()}

    for name, result in results.items():
        print(f"{name:>12}: {result['microseconds_per_call']:.2f} us/call over {result['n_decision_points']} decision points x {result['repeats']} repeats")

    baseline_us = results["periodic"]["microseconds_per_call"]
    for name, result in results.items():
        result["ratio_vs_periodic"] = result["microseconds_per_call"] / baseline_us
        print(f"{name:>12}: {result['ratio_vs_periodic']:.2f}x periodic's decision cost")

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_path.write_text(json.dumps({"config": vars(args) | {"fit_report": str(args.fit_report), "trajectories": str(args.trajectories), "out_path": str(args.out_path)}, "results": results}, indent=2, default=str))
    print(f"report written to {args.out_path}")


if __name__ == "__main__":
    main()
