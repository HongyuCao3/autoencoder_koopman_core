#!/usr/bin/env python3
"""F2 (docs/experiments/signal_resolution_plan.md section three): refits
Phase B with `closeness` (F1's de-thresholded state readout) instead of
`y_task_success`. Copies scripts/fit_koopman_ergo_model.py's structure --
same split protocol, same state config, same two models (ARX/richer) --
with these differences:

1. `Y_COL = "closeness"` (computed from the same extractor + gold answer
   scripts/analyze_ergo_closeness_readout.py's `_closeness` uses -- section
   2.2's definition, duplicated here verbatim rather than importing across
   scripts, matching this codebase's existing per-domain-script convention).
2. Output path is a NEW file, `koopman_fit_report_closeness.json` -- does
   NOT overwrite `koopman_fit_report.json` (the y_task_success fit).
3. `richer_abs_sign` is marked `"vacuous_for_binary_y": false` -- unlike
   `y_task_success` (binary, so |y|==y and sign(y)==y make richer's extra
   features exactly collinear with y), `closeness` is continuous, so this
   comparison is no longer degenerate (still not RC-1's judge -- RC-1's
   opponent is always the three trivial nulls, never richer).
4. Rollout MSE is reported BOTH naive (`modeling.evaluate.rollout_output_error`,
   unmodified, shared with the defense line) and aux(shard_frac)-truth-
   overridden (a local variant, same one
   scripts/analyze_ergo_readout_state.py's `rollout_output_error_aux_truth`
   uses), plus the three trivial nulls on the same held-out split -- the
   fit's own `held_out_rollout_mse` field keeps the naive number (matching
   `koopman_fit_report.json`'s existing schema so the two files stay
   comparable), and the additional fields carry the rest.

CPU-only, pure numpy -- no GPU needed. Run directly (no sbatch).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from persona_drift.ergo_math_judge import _normalize, extract_answer_by_regex  # noqa: E402
from persona_drift.modeling.dataset import (  # noqa: E402
    ReducedStateConfig,
    build_identification_dataset,
    build_reduced_state_pairs,
    group_by_trajectory,
    load_trajectories,
    split_by_system_prompt_id,
)
from persona_drift.modeling.evaluate import one_step_error, rollout_output_error  # noqa: E402
from persona_drift.modeling.koopman import (  # noqa: E402
    KoopmanSurrogate,
    abs_sign_extra_features,
    no_extra_features,
)

import numpy as np  # noqa: E402

Y_COL = "closeness"
U_COL = "u_reset"
AUX_COL = "shard_frac"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--rows-path",
        type=pathlib.Path,
        default=pathlib.Path("outputs/ergo_math_phaseB_random_excite/trajectories.jsonl"),
    )
    parser.add_argument("--nu", type=int, default=1)
    parser.add_argument("--mu", type=int, default=1)
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument("--controllability-horizon", type=int, default=5)
    parser.add_argument("--held-out-frac", type=float, default=0.25)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument(
        "--out-path",
        type=pathlib.Path,
        default=pathlib.Path("outputs/ergo_math_phaseB_random_excite/koopman_fit_report_closeness.json"),
    )
    return parser.parse_args()


def _to_number(text: str | None) -> float | None:
    if text is None:
        return None
    try:
        return float(_normalize(text))
    except (ValueError, TypeError):
        return None


def _add_derived_columns(rows: list[dict]) -> list[dict]:
    for row in rows:
        row[AUX_COL] = row["turn"] / row["num_shards"]
        extracted = extract_answer_by_regex(row["agent_message"])
        a = _to_number(extracted) if extracted is not None else None
        g = _to_number(row["gold_answer"])
        row[Y_COL] = 0.0 if a is None or g is None else 1.0 / (1.0 + abs(a - g) / max(abs(g), 1.0))
    return rows


def rollout_output_error_aux_truth(predictor, rows: list[dict], config: ReducedStateConfig) -> float:
    squared_errors: list[float] = []
    for traj_rows in group_by_trajectory(rows, id_col="trajectory_id").values():
        pairs = build_reduced_state_pairs(traj_rows, config, y_col=Y_COL, u_col=U_COL)
        if not pairs:
            continue
        z = pairs[0]["z"]
        squared_errors.append((predictor.readout(z) - pairs[0]["y"]) ** 2)
        for pair in pairs:
            z = predictor.step(z, pair["v"])
            z[-1] = pair["z_next"][-1]
            true_y_next = pair["z_next"][config.nu - 1]
            squared_errors.append((predictor.readout(z) - true_y_next) ** 2)
    return float(np.mean(squared_errors)) if squared_errors else float("nan")


def _ols_beta(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def _null_mses(train_rows: list[dict], held_out_rows: list[dict]) -> dict:
    train_y = np.array([r[Y_COL] for r in train_rows])
    held_y = np.array([r[Y_COL] for r in held_out_rows])

    global_mean = float(train_y.mean())
    mse_constant = float(np.mean((held_y - global_mean) ** 2))

    turn_mean: dict[int, float] = {}
    for t in sorted({r["turn"] for r in train_rows}):
        turn_mean[t] = float(np.mean([r[Y_COL] for r in train_rows if r["turn"] == t]))
    pred_turn = np.array([turn_mean.get(r["turn"], global_mean) for r in held_out_rows])
    mse_turn_mean = float(np.mean((held_y - pred_turn) ** 2))

    X_train = np.array([[1.0, r[AUX_COL], float(r[U_COL])] for r in train_rows])
    beta = _ols_beta(X_train, train_y)
    X_held = np.array([[1.0, r[AUX_COL], float(r[U_COL])] for r in held_out_rows])
    mse_stateless_ols = float(np.mean((held_y - X_held @ beta) ** 2))

    return {"constant": mse_constant, "turn_mean": mse_turn_mean, "stateless_ols": mse_stateless_ols}


def _fit_and_evaluate(name, extra_features_fn, train_rows, held_out_rows, config, ridge):
    train_dataset = build_identification_dataset(train_rows, config, y_col=Y_COL, u_col=U_COL)
    model = KoopmanSurrogate(extra_features_fn=extra_features_fn, ridge=ridge).fit(train_dataset)
    return {
        "name": name,
        "A": model.A.tolist(),
        "B": model.B.tolist(),
        "b": model.b.tolist(),
        "C": model.C.tolist(),
        "train_one_step_mse": one_step_error(model, train_dataset),
        "held_out_rollout_mse": rollout_output_error(model, held_out_rows, config, y_col=Y_COL, u_col=U_COL),
    }, model


def main() -> None:
    args = parse_args()
    rows = _add_derived_columns(load_trajectories(args.rows_path))

    split = split_by_system_prompt_id(
        rows,
        train_frac=1.0 - args.held_out_frac,
        val_frac=0.0,
        seed=args.split_seed,
        split_col="item_id",
    )
    train_rows = split["train"]
    held_out_rows = split["test"]
    n_train_items = len({r["item_id"] for r in train_rows})
    held_out_item_ids = sorted({r["item_id"] for r in held_out_rows})
    n_held_out_items = len(held_out_item_ids)

    config = ReducedStateConfig(nu=args.nu, mu=args.mu, aux_cols=(AUX_COL,), contemporaneous_v=True)

    arx_report, arx_model = _fit_and_evaluate("arx", no_extra_features, train_rows, held_out_rows, config, args.ridge)
    richer_report, richer_model = _fit_and_evaluate(
        "richer_abs_sign", abs_sign_extra_features, train_rows, held_out_rows, config, args.ridge
    )
    richer_report["vacuous_for_binary_y"] = False

    aux_truth_rollout_mse = rollout_output_error_aux_truth(arx_model, held_out_rows, config)
    nulls = _null_mses(train_rows, held_out_rows)

    controllability = arx_model.controllability(args.controllability_horizon)

    report = {
        "config": {
            "nu": args.nu,
            "mu": args.mu,
            "ridge": args.ridge,
            "rows_path": str(args.rows_path),
            "aux_cols": [AUX_COL],
            "contemporaneous_v": True,
            "y_col": Y_COL,
            "u_col": U_COL,
        },
        "n_train_items": n_train_items,
        "n_held_out_items": n_held_out_items,
        "held_out_item_ids": held_out_item_ids,
        "arx": arx_report,
        "richer_abs_sign": richer_report,
        "arx_held_out_rollout_mse_aux_truth": aux_truth_rollout_mse,
        "null_held_out_mses": nulls,
        "controllability_arx": controllability,
    }
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.out_path.exists():
        raise SystemExit(f"refusing to overwrite existing {args.out_path}")
    args.out_path.write_text(json.dumps(report, indent=2))

    print(f"n_train_items={n_train_items} n_held_out_items={n_held_out_items}")
    print(f"held_out_item_ids={held_out_item_ids}")
    print(f"ARX: A={arx_model.A.tolist()} B={arx_model.B.tolist()} b={arx_model.b.tolist()}")
    print(f"ARX held_out_rollout_mse (naive) ={arx_report['held_out_rollout_mse']:.6f}")
    print(f"ARX held_out_rollout_mse (aux=truth) ={aux_truth_rollout_mse:.6f}")
    print(f"null MSEs: {nulls}")
    print(f"richer held_out_rollout_mse={richer_report['held_out_rollout_mse']:.6f} (vacuous_for_binary_y=False)")
    print(f"controllability_rank={controllability['controllability_rank']} (state_dim={arx_model.state_dim})")
    print(f"gramian_condition={controllability['gramian_condition']:.4e}")
    print(f"A_spectral_radius={controllability['spectral_radius']:.4f}")
    print(f"report written to {args.out_path}")


if __name__ == "__main__":
    main()
