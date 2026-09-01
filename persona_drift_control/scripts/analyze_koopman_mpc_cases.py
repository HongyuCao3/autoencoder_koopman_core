#!/usr/bin/env python3
"""Case-study analysis for `KoopmanMPCController`'s decisions
(docs/experiments/koopman_case_study_design.md): replays the controller's
exact decision logic (same fitted richer_abs_sign model, same state
reconstruction) over Phase E/G's already-recorded trajectories to surface
five candidate phenomena about *how* koopman_mpc behaves, in response to
Phase G's open question (koopman_mpc ties periodic at matched cost -- does
it actually adapt to attack-specific dynamics, or does it degenerate to a
fixed pattern?).

Not a new experiment: loads the existing richer_abs_sign (nu=1, mu=2) model
from outputs/koopman_defense_phaseB_random_excite/koopman_fit_report.json
and Phase E/G's trajectories.jsonl. Reuses KoopmanMPCController itself
(not a reimplementation) so the reconstructed decisions are exactly what the
real controller computed, just with the previously-discarded intermediate
values (value_immediate/value_full per action) kept instead of thrown away.

CPU-only, pure numpy/pandas -- no GPU needed. Run directly (no sbatch).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from persona_drift.control import KoopmanMPCController  # noqa: E402
from persona_drift.modeling.dataset import (  # noqa: E402
    ReducedStateConfig,
    group_by_trajectory,
    load_trajectories,
)
from persona_drift.modeling.koopman import (  # noqa: E402
    abs_sign_extra_features,
    surrogate_from_arrays,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fit-report",
        type=pathlib.Path,
        default=pathlib.Path("outputs/koopman_defense_phaseB_random_excite/koopman_fit_report.json"),
    )
    parser.add_argument("--model-key", default="richer_abs_sign")
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument("--repeat-penalty", type=float, default=0.0)
    parser.add_argument("--threshold-y-min", type=float, default=0.7)
    parser.add_argument(
        "--koopman-mpc-dir",
        type=pathlib.Path,
        default=pathlib.Path("outputs/koopman_defense_phaseE_koopman_mpc/trajectories.jsonl"),
    )
    parser.add_argument(
        "--threshold-dir",
        type=pathlib.Path,
        default=pathlib.Path("outputs/koopman_defense_phaseE_threshold/trajectories.jsonl"),
    )
    parser.add_argument(
        "--constant-remind-dir",
        type=pathlib.Path,
        default=pathlib.Path("outputs/koopman_defense_phaseE_constant_remind/trajectories.jsonl"),
    )
    parser.add_argument(
        "--periodic-dir",
        type=pathlib.Path,
        default=pathlib.Path("outputs/koopman_defense_phaseG_periodic/trajectories.jsonl"),
    )
    parser.add_argument("--out-dir", type=pathlib.Path, default=pathlib.Path("outputs/koopman_case_study"))
    return parser.parse_args()


def load_controller(fit_report_path: pathlib.Path, model_key: str, horizon: int, repeat_penalty: float) -> KoopmanMPCController:
    report = json.loads(fit_report_path.read_text())
    cfg = report["config"]
    model_report = report[model_key]
    if model_key != "richer_abs_sign":
        raise ValueError(f"unsupported model_key {model_key!r} for this script")
    config = ReducedStateConfig(nu=cfg["nu"], mu=cfg["mu"])
    surrogate = surrogate_from_arrays(
        A=model_report["A"],
        B=model_report["B"],
        b=model_report["b"],
        C=model_report["C"],
        state_dim=config.state_dim,
        extra_features_fn=abs_sign_extra_features,
        ridge=cfg["ridge"],
    )
    return KoopmanMPCController(
        surrogate=surrogate, state_config=config, horizon=horizon, repeat_penalty=repeat_penalty
    )


def load_arm(path: pathlib.Path) -> dict[str, list[dict]]:
    return group_by_trajectory(load_trajectories(path))


def excerpt(row: dict, n: int = 160) -> str:
    text = (row.get("agent_message") or "").replace("\n", " ")
    return text[:n] + ("..." if len(text) > n else "")


def build_case_rows(controller: KoopmanMPCController, arm_rows: dict[str, list[dict]]) -> list[dict]:
    """Replays controller.next_u_remind turn-by-turn, but also keeps
    value_immediate(a) = controller._simulate(z, a, 0) (horizon=1 greedy)
    and value_full(a) = controller._simulate(z, a, horizon-1) (the actual
    MPC value the real run used), for every turn where a real state exists."""

    records = []
    for traj_id, rows in arm_rows.items():
        for i, row in enumerate(rows):
            history = rows[:i]
            z = controller._current_state(history)
            record = {
                "trajectory_id": traj_id,
                "attack_id": row["attack_id"],
                "seed": row["seed"],
                "turn": row["turn"],
                "y_probe": row["y_probe"],
                "u_remind_actual": row["u_remind"],
                "has_state": z is not None,
            }
            if z is not None:
                for action in (0, 1):
                    record[f"value_immediate_{action}"] = controller._simulate(z, action, 0)
                    record[f"value_full_{action}"] = controller._simulate(z, action, controller.horizon - 1)
                record["argmax_immediate"] = int(
                    record["value_immediate_1"] > record["value_immediate_0"]
                )
                record["argmax_full"] = int(record["value_full_1"] > record["value_full_0"])
                record["horizon_changed_decision"] = record["argmax_immediate"] != record["argmax_full"]
            records.append(record)
    return records


def phenomenon_1_pattern_vs_periodic(koopman_rows: dict, periodic_rows: dict) -> pd.DataFrame:
    out = []
    for traj_id in sorted(set(koopman_rows) & set(periodic_rows)):
        k_pattern = tuple(r["u_remind"] for r in koopman_rows[traj_id])
        p_pattern = tuple(r["u_remind"] for r in periodic_rows[traj_id])
        out.append(
            {
                "trajectory_id": traj_id,
                "koopman_pattern": k_pattern,
                "periodic_pattern": p_pattern,
                "identical": k_pattern == p_pattern,
            }
        )
    return pd.DataFrame(out)


def phenomenon_2_anticipatory_vs_threshold(
    koopman_rows: dict, threshold_rows: dict, y_min: float
) -> pd.DataFrame:
    out = []
    for traj_id in sorted(set(koopman_rows) & set(threshold_rows)):
        k_rows, t_rows = koopman_rows[traj_id], threshold_rows[traj_id]
        for k_row, t_row in zip(k_rows, t_rows):
            if k_row["u_remind"] == 1 and t_row["u_remind"] == 0 and k_row["y_probe"] >= y_min:
                out.append(
                    {
                        "trajectory_id": traj_id,
                        "turn": k_row["turn"],
                        "koopman_y_probe": k_row["y_probe"],
                        "threshold_y_min": y_min,
                        "koopman_agent_excerpt": excerpt(k_row),
                    }
                )
                break  # earliest divergence only
    return pd.DataFrame(out)


def phenomenon_3_economy_vs_constant(koopman_rows: dict, constant_rows: dict) -> pd.DataFrame:
    out = []
    for traj_id in sorted(set(koopman_rows) & set(constant_rows)):
        k_rows = koopman_rows[traj_id]
        for i, k_row in enumerate(k_rows):
            if k_row["u_remind"] == 0:
                next_y = k_rows[i + 1]["y_probe"] if i + 1 < len(k_rows) else None
                out.append(
                    {
                        "trajectory_id": traj_id,
                        "turn": k_row["turn"],
                        "koopman_y_probe": k_row["y_probe"],
                        "koopman_next_y_probe": next_y,
                        "had_real_state": k_row["turn"] > 3,  # nu=1,mu=2 fallback cutoff, see docstring
                    }
                )
    return pd.DataFrame(out)


def phenomenon_4_horizon_matters(case_rows: pd.DataFrame) -> pd.DataFrame:
    real = case_rows[case_rows["has_state"]]
    return real[
        [
            "trajectory_id",
            "turn",
            "y_probe",
            "value_immediate_0",
            "value_immediate_1",
            "value_full_0",
            "value_full_1",
            "argmax_immediate",
            "argmax_full",
            "horizon_changed_decision",
            "u_remind_actual",
        ]
    ].reset_index(drop=True)


def phenomenon_5_mean_reversion(
    koopman_rows: dict, threshold_rows: dict, y_min: float, controller: KoopmanMPCController
) -> pd.DataFrame:
    out = []
    for traj_id in sorted(set(koopman_rows) & set(threshold_rows)):
        k_rows, t_rows = koopman_rows[traj_id], threshold_rows[traj_id]
        for i, (k_row, t_row) in enumerate(zip(k_rows, t_rows)):
            if t_row["u_remind"] == 0 and t_row["y_probe"] < y_min:
                continue  # threshold's own decision only fires off ITS y_probe; not comparable here
            if k_row["u_remind"] == 0 and k_row["y_probe"] < y_min:
                z = controller._current_state(k_rows[:i])
                out.append(
                    {
                        "trajectory_id": traj_id,
                        "turn": k_row["turn"],
                        "koopman_y_probe": k_row["y_probe"],
                        "had_real_state": z is not None,
                        "z_t": z.tolist() if z is not None else None,
                    }
                )
    return pd.DataFrame(out)


def main() -> None:
    args = parse_args()
    controller = load_controller(args.fit_report, args.model_key, args.horizon, args.repeat_penalty)

    koopman_rows = load_arm(args.koopman_mpc_dir)
    threshold_rows = load_arm(args.threshold_dir)
    constant_rows = load_arm(args.constant_remind_dir)
    periodic_rows = load_arm(args.periodic_dir)

    case_records = build_case_rows(controller, koopman_rows)
    case_df = pd.DataFrame(case_records)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    case_df.to_csv(args.out_dir / "koopman_mpc_decisions.csv", index=False)

    p1 = phenomenon_1_pattern_vs_periodic(koopman_rows, periodic_rows)
    p2 = phenomenon_2_anticipatory_vs_threshold(koopman_rows, threshold_rows, args.threshold_y_min)
    p3 = phenomenon_3_economy_vs_constant(koopman_rows, constant_rows)
    p4 = phenomenon_4_horizon_matters(case_df)
    p5 = phenomenon_5_mean_reversion(koopman_rows, threshold_rows, args.threshold_y_min, controller)

    p1.to_csv(args.out_dir / "phenomenon1_pattern_vs_periodic.csv", index=False)
    p2.to_csv(args.out_dir / "phenomenon2_anticipatory_vs_threshold.csv", index=False)
    p3.to_csv(args.out_dir / "phenomenon3_economy_vs_constant.csv", index=False)
    p4.to_csv(args.out_dir / "phenomenon4_horizon_matters.csv", index=False)
    p5.to_csv(args.out_dir / "phenomenon5_mean_reversion.csv", index=False)

    summary = {
        "n_trajectories": len(koopman_rows),
        "phenomenon_1_n_identical_to_periodic": int(p1["identical"].sum()) if len(p1) else None,
        "phenomenon_1_n_total": len(p1),
        "phenomenon_2_n_anticipatory_cases": len(p2),
        "phenomenon_3_n_zero_remind_turns": len(p3),
        "phenomenon_3_n_zero_remind_turns_with_real_state": int(p3["had_real_state"].sum()) if len(p3) else 0,
        "phenomenon_4_n_real_decisions": len(p4),
        "phenomenon_4_n_horizon_changed_decision": int(p4["horizon_changed_decision"].sum()) if len(p4) else 0,
        "phenomenon_5_n_cases": len(p5),
    }
    (args.out_dir / "case_study_summary.json").write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))
    print(f"tables written to {args.out_dir}/")


if __name__ == "__main__":
    main()
