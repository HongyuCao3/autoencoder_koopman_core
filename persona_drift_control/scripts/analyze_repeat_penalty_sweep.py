#!/usr/bin/env python3
"""Path 1 of "how to prove Koopman's motivation"
(docs/experiments/koopman_case_study_design.md's "对下一步方向的启示"):
tests whether simply raising `KoopmanMPCController.repeat_penalty` above 0
can make reminder decisions state-dependent, or whether it can only ever
produce a uniform flip (all-remind <-> never-remind) -- as the case study's
algebra already predicts, since `repeat_penalty` is a flat per-action cost
with no dependence on `z` or on how recently a reminder was actually
inserted.

Reuses the exact same fitted `richer_abs_sign` model and the exact same real
recorded states from Phase E's `koopman_mpc` arm as
`analyze_koopman_mpc_cases.py` (same replay method: `_current_state` +
`_simulate` for both actions at every turn with a real state), generalized
to a grid of `repeat_penalty` values instead of a single one. Not a new
experiment -- offline, no GPU/LLM calls.

CPU-only. Run directly (no sbatch).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from persona_drift.control import KoopmanMPCController  # noqa: E402
from persona_drift.modeling.dataset import ReducedStateConfig, group_by_trajectory, load_trajectories  # noqa: E402
from persona_drift.modeling.koopman import abs_sign_extra_features, surrogate_from_arrays  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fit-report", type=pathlib.Path, default=pathlib.Path("outputs/koopman_defense_phaseB_random_excite/koopman_fit_report.json")
    )
    parser.add_argument(
        "--koopman-mpc-dir", type=pathlib.Path, default=pathlib.Path("outputs/koopman_defense_phaseE_koopman_mpc/trajectories.jsonl")
    )
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument(
        "--repeat-penalties", type=float, nargs="*", default=[0.0, 0.1, 0.2, 0.3, 0.35, 0.3672, 0.37, 0.4, 0.5, 1.0]
    )
    parser.add_argument("--out-path", type=pathlib.Path, default=pathlib.Path("outputs/koopman_case_study/repeat_penalty_sweep.json"))
    return parser.parse_args()


def load_surrogate(fit_report_path: pathlib.Path):
    report = json.loads(fit_report_path.read_text())
    cfg, model_report = report["config"], report["richer_abs_sign"]
    config = ReducedStateConfig(nu=cfg["nu"], mu=cfg["mu"])
    surrogate = surrogate_from_arrays(
        A=model_report["A"], B=model_report["B"], b=model_report["b"], C=model_report["C"],
        state_dim=config.state_dim, extra_features_fn=abs_sign_extra_features, ridge=cfg["ridge"],
    )
    return surrogate, config


def real_decision_states(controller: KoopmanMPCController, arm_rows: dict[str, list[dict]]) -> list[np.ndarray]:
    states = []
    for rows in arm_rows.values():
        for i in range(len(rows)):
            z = controller._current_state(rows[:i])
            if z is not None:
                states.append(z)
    return states


def sweep(surrogate, config: ReducedStateConfig, horizon: int, states: list[np.ndarray], repeat_penalty: float) -> dict:
    controller = KoopmanMPCController(surrogate=surrogate, state_config=config, horizon=horizon, repeat_penalty=repeat_penalty)
    margins = []
    actions = []
    for z in states:
        value_0 = controller._simulate(z, 0, horizon - 1)
        value_1 = controller._simulate(z, 1, horizon - 1)
        margins.append(value_1 - value_0)
        actions.append(int(value_1 > value_0))
    margins = np.array(margins)
    actions = np.array(actions)
    return {
        "repeat_penalty": repeat_penalty,
        "n_states": len(states),
        "n_remind": int(actions.sum()),
        "fraction_remind": float(actions.mean()),
        "is_genuine_mix": bool(0 < actions.sum() < len(actions)),
        "margin_mean": float(margins.mean()),
        "margin_std": float(margins.std()),
        "margin_min": float(margins.min()),
        "margin_max": float(margins.max()),
    }


def main() -> None:
    args = parse_args()
    surrogate, config = load_surrogate(args.fit_report)
    zero_penalty_controller = KoopmanMPCController(surrogate=surrogate, state_config=config, horizon=args.horizon)
    arm_rows = group_by_trajectory(load_trajectories(args.koopman_mpc_dir))
    states = real_decision_states(zero_penalty_controller, arm_rows)

    results = [sweep(surrogate, config, args.horizon, states, rp) for rp in args.repeat_penalties]

    print(f"{'repeat_penalty':>14} {'n_remind':>9} {'fraction':>9} {'genuine_mix':>12} {'margin_std':>11}")
    for r in results:
        print(
            f"{r['repeat_penalty']:>14.4f} {r['n_remind']:>9} {r['fraction_remind']:>9.3f} "
            f"{str(r['is_genuine_mix']):>12} {r['margin_std']:>11.2e}"
        )

    any_mix = any(r["is_genuine_mix"] for r in results)
    print(f"\nany repeat_penalty value in this grid produced a genuine 0/1 mix: {any_mix}")

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_path.write_text(json.dumps({"n_states": len(states), "results": results, "any_genuine_mix": any_mix}, indent=2))
    print(f"report written to {args.out_path}")


if __name__ == "__main__":
    main()
