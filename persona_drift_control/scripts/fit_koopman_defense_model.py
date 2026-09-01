#!/usr/bin/env python3
"""Phase C of docs/experiments/koopman_defense_pilot.md: fit a Koopman
surrogate on Phase B's open-loop random-excitation trajectories
(y_safety/u_remind instead of persona-drift's y_probe/u_remind), evaluate
one-step/rollout error on a held-out attack_id split, and run
controllability diagnostics on B -- the go/no-go gate before designing a
KoopmanMPCController on top of it.

CPU-only, pure numpy/pandas -- no GPU needed. Run directly (no sbatch).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from persona_drift.modeling.dataset import (  # noqa: E402
    ReducedStateConfig,
    build_identification_dataset,
    load_trajectories,
    split_by_system_prompt_id,
)
from persona_drift.modeling.evaluate import one_step_error, rollout_output_error  # noqa: E402
from persona_drift.modeling.koopman import (  # noqa: E402
    KoopmanSurrogate,
    abs_sign_extra_features,
    no_extra_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rows-path", type=pathlib.Path, default=pathlib.Path("outputs/koopman_defense_phaseB_random_excite/trajectories.jsonl")
    )
    parser.add_argument("--nu", type=int, default=1)
    parser.add_argument("--mu", type=int, default=1)
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument("--controllability-horizon", type=int, default=5)
    parser.add_argument("--held-out-frac", type=float, default=0.25)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument(
        "--out-path", type=pathlib.Path, default=pathlib.Path("outputs/koopman_defense_phaseB_random_excite/koopman_fit_report.json")
    )
    return parser.parse_args()


def _fit_and_evaluate(name, extra_features_fn, train_rows, held_out_rows, config, ridge):
    train_dataset = build_identification_dataset(train_rows, config, y_col="y_safety")
    model = KoopmanSurrogate(extra_features_fn=extra_features_fn, ridge=ridge).fit(train_dataset)
    return {
        "name": name,
        "A": model.A.tolist(),
        "B": model.B.tolist(),
        "b": model.b.tolist(),
        "C": model.C.tolist(),
        "train_one_step_mse": one_step_error(model, train_dataset),
        "held_out_rollout_mse": rollout_output_error(model, held_out_rows, config, y_col="y_safety"),
    }, model


def main() -> None:
    args = parse_args()
    rows = load_trajectories(args.rows_path)
    config = ReducedStateConfig(nu=args.nu, mu=args.mu)

    split = split_by_system_prompt_id(
        rows,
        train_frac=1.0 - args.held_out_frac,
        val_frac=0.0,
        seed=args.split_seed,
        split_col="attack_id",
    )
    train_rows = split["train"]
    held_out_rows = split["test"]
    n_train_attacks = len({r["attack_id"] for r in train_rows})
    n_held_out_attacks = len({r["attack_id"] for r in held_out_rows})

    arx_report, arx_model = _fit_and_evaluate(
        "arx", no_extra_features, train_rows, held_out_rows, config, args.ridge
    )
    richer_report, richer_model = _fit_and_evaluate(
        "richer_abs_sign", abs_sign_extra_features, train_rows, held_out_rows, config, args.ridge
    )

    controllability = arx_model.controllability(args.controllability_horizon)

    report = {
        "config": {"nu": args.nu, "mu": args.mu, "ridge": args.ridge, "rows_path": str(args.rows_path)},
        "n_train_attacks": n_train_attacks,
        "n_held_out_attacks": n_held_out_attacks,
        "arx": arx_report,
        "richer_abs_sign": richer_report,
        "controllability_arx": controllability,
    }
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_path.write_text(json.dumps(report, indent=2))

    print(f"n_train_attacks={n_train_attacks} n_held_out_attacks={n_held_out_attacks}")
    print(f"ARX: A={arx_model.A.tolist()} B={arx_model.B.tolist()} b={arx_model.b.tolist()}")
    print(f"ARX held_out_rollout_mse={arx_report['held_out_rollout_mse']:.6f}")
    print(f"richer held_out_rollout_mse={richer_report['held_out_rollout_mse']:.6f}")
    print(f"controllability_rank={controllability['controllability_rank']} (state_dim={arx_model.state_dim})")
    print(f"gramian_condition={controllability['gramian_condition']:.4e}")
    print(f"A_spectral_radius={controllability['spectral_radius']:.4f}")
    print(f"report written to {args.out_path}")


if __name__ == "__main__":
    main()
