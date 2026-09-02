#!/usr/bin/env python3
"""Path 2 of "how to prove Koopman's motivation"
(docs/experiments/koopman_case_study_design.md's "对下一步方向的启示"): fits
a Koopman surrogate with an explicit state-action interaction control input
(`modeling.interaction_lift.augment_with_interaction`) on the exact same
Phase B data and `attack_id` split `fit_koopman_defense_model.py` used for
`richer_abs_sign`, reports one-step/rollout MSE for a head-to-head
comparison, then replays `KoopmanMPCController` decisions (wrapping the
fitted model in `InteractionLiftedSurrogate`) over Phase E `koopman_mpc`'s
real recorded states -- exactly `analyze_koopman_mpc_cases.py`'s replay
method -- to check whether the remind-vs-not marginal value is now genuinely
state-dependent (path 1's `analyze_repeat_penalty_sweep.py` proved a flat
`repeat_penalty` cannot do this).

Uses `no_extra_features` (plain ARX) as the state lifting, not
`abs_sign_extra_features`: `koopman_detection_design.md`'s dataset shows
`y_safety` never goes negative (min 0.0, `abs(y)==y` in every one of 300
rows, `sign(y)` is 1 in 290/300 and 0 only at the 10 exact-zero rows) --
`abs_sign_extra_features` is near-degenerate on this signal, so stacking it
with the new interaction term would muddy which change is responsible for
any observed effect. The interaction term is the one thing under test here.

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
from persona_drift.modeling.dataset import (  # noqa: E402
    ReducedStateConfig,
    build_identification_dataset,
    group_by_trajectory,
    load_trajectories,
    split_by_system_prompt_id,
)
from persona_drift.modeling.evaluate import one_step_error, rollout_output_error  # noqa: E402
from persona_drift.modeling.interaction_lift import InteractionLiftedSurrogate, augment_with_interaction  # noqa: E402
from persona_drift.modeling.koopman import KoopmanSurrogate, no_extra_features  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rows-path", type=pathlib.Path, default=pathlib.Path("outputs/koopman_defense_phaseB_random_excite/trajectories.jsonl")
    )
    parser.add_argument(
        "--koopman-fit-report", type=pathlib.Path, default=pathlib.Path("outputs/koopman_defense_phaseB_random_excite/koopman_fit_report.json")
    )
    parser.add_argument(
        "--phase-e-koopman-mpc-dir", type=pathlib.Path, default=pathlib.Path("outputs/koopman_defense_phaseE_koopman_mpc/trajectories.jsonl")
    )
    parser.add_argument("--nu", type=int, default=1)
    parser.add_argument("--mu", type=int, default=2)
    parser.add_argument(
        "--contemporaneous-v",
        action="store_true",
        help="fit under the corrected v-alignment (ReducedStateConfig.contemporaneous_v -- v is the SAME turn's "
        "action as z_next's y, matching attack_trajectory.py's real same-turn causal timing) instead of the "
        "original 'v contemporaneous with z_t' convention this script originally shipped with, which measures "
        "only the reminder's residual/carryover effect one turn later -- see docs/next step.md (2026-09-02).",
    )
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument("--held-out-frac", type=float, default=0.25)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument(
        "--repeat-penalty",
        type=float,
        default=0.0,
        help="now meaningful (unlike the flat repeat_penalty in path 1's sweep): a value inside the interaction model's observed margin range should produce a genuine action mix.",
    )
    parser.add_argument("--out-path", type=pathlib.Path, default=pathlib.Path("outputs/koopman_case_study/interaction_model_report.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_trajectories(args.rows_path)

    split = split_by_system_prompt_id(
        rows, train_frac=1.0 - args.held_out_frac, val_frac=0.0, seed=args.split_seed, split_col="attack_id"
    )
    train_rows, held_out_rows = split["train"], split["test"]
    config = ReducedStateConfig(nu=args.nu, mu=args.mu, contemporaneous_v=args.contemporaneous_v)

    train_dataset = build_identification_dataset(train_rows, config, y_col="y_safety")
    V_aug = augment_with_interaction(train_dataset["V"], train_dataset["Z"], state_index=0)
    model = KoopmanSurrogate(extra_features_fn=no_extra_features, ridge=args.ridge).fit({**train_dataset, "V": V_aug})
    wrapped = InteractionLiftedSurrogate(surrogate=model, state_index=0)

    train_one_step_mse = one_step_error(wrapped, train_dataset)  # raw V (d_v=1): wrapper lifts internally
    held_out_rollout_mse = rollout_output_error(wrapped, held_out_rows, config, y_col="y_safety")

    baseline = json.loads(args.koopman_fit_report.read_text())
    arx_mse = baseline["arx"]["held_out_rollout_mse"]
    richer_mse = baseline["richer_abs_sign"]["held_out_rollout_mse"]

    print(f"interaction model: A={model.A.tolist()} B={model.B.tolist()} b={model.b.tolist()}")
    print(f"interaction train_one_step_mse={train_one_step_mse:.4f}")
    print(f"interaction held_out_rollout_mse={held_out_rollout_mse:.4f}")
    print(f"(reference) arx held_out_rollout_mse={arx_mse:.4f}, richer_abs_sign held_out_rollout_mse={richer_mse:.4f}")

    # Path 2's real question: replay decisions over Phase E koopman_mpc's
    # real recorded states and check whether the remind-vs-not marginal
    # value now depends on z, unlike path 1's repeat_penalty sweep.
    controller = KoopmanMPCController(surrogate=wrapped, state_config=config, horizon=args.horizon, repeat_penalty=args.repeat_penalty)
    arm_rows = group_by_trajectory(load_trajectories(args.phase_e_koopman_mpc_dir))

    records = []
    for rows_ in arm_rows.values():
        for i in range(len(rows_)):
            history = rows_[:i]
            z = controller._current_state(history)
            if z is None:
                continue
            value_0 = controller._simulate(z, 0, args.horizon - 1)
            value_1 = controller._simulate(z, 1, args.horizon - 1)
            records.append({"y_probe": float(z[0]), "z": z.tolist(), "margin": value_1 - value_0, "action": int(value_1 > value_0)})

    margins = np.array([r["margin"] for r in records])
    actions = np.array([r["action"] for r in records])
    y_probes = np.array([r["y_probe"] for r in records])
    correlation = float(np.corrcoef(margins, y_probes)[0, 1]) if len(records) > 1 and margins.std() > 0 else None

    print(f"\nreal decisions replayed: n={len(records)}")
    print(f"margin std={margins.std():.6f} (path 1's repeat_penalty sweep found ~1e-16 at every setting)")
    print(f"margin min={margins.min():.4f} max={margins.max():.4f}")
    print(f"n_remind={int(actions.sum())}/{len(actions)} (genuine mix: {0 < actions.sum() < len(actions)})")
    print(f"corr(margin, y_probe)={correlation}")

    report = {
        "config": {k: str(v) for k, v in vars(args).items()},
        "model": {"A": model.A.tolist(), "B": model.B.tolist(), "b": model.b.tolist(), "C": model.C.tolist()},
        "train_one_step_mse": train_one_step_mse,
        "held_out_rollout_mse": held_out_rollout_mse,
        "reference_arx_held_out_rollout_mse": arx_mse,
        "reference_richer_abs_sign_held_out_rollout_mse": richer_mse,
        "decision_replay": {
            "n_real_decisions": len(records),
            "margin_std": float(margins.std()),
            "margin_min": float(margins.min()),
            "margin_max": float(margins.max()),
            "n_remind": int(actions.sum()),
            "is_genuine_mix": bool(0 < actions.sum() < len(actions)),
            "corr_margin_y_probe": correlation,
            "records": records,
        },
    }
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nreport written to {args.out_path}")


if __name__ == "__main__":
    main()
