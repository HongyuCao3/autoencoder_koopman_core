#!/usr/bin/env python3
"""Follow-up to scripts/analyze_v_alignment_fix.py and
scripts/analyze_compounding_hypothesis.py (2026-09-02): the mu sweep showed
`nu=1, mu=2` -- the memory length Phase C-H already used throughout -- with
the corrected v-alignment (`contemporaneous_v=True`) explains ~107% of
Phase A's observed cumulative remind-vs-no-remind gap by the terminal turn
(vs -1.6% for the old mu=1 single-step check). So the fix doesn't need a
bigger/richer state at all: just re-fit the SAME `nu=1,mu=2` architecture
Phase D-H already used, with the alignment bug fixed.

This script re-runs the state-action-interaction offline-replay check
(same method as scripts/analyze_state_action_interaction.py /
analyze_v_alignment_fix.py's step 3) at `mu=2` instead of `mu=1`, and adds
one more diagnostic the earlier checks didn't: Phase H's real closed-loop
failure was concrete and named -- `safemtdata_0074__seed0` and
`safemtdata_0476__seed0` (fast-eroding attacks) had their turn4/5 reminders
REMOVED by the old-alignment interaction model + repeat_penalty=0.2, while
`safemtdata_0074__seed1`/`safemtdata_0169`/`safemtdata_0530` (resilient
attacks/seeds) kept theirs -- exactly backwards. This script prints the
new-alignment model's margin at those exact (trajectory, turn) points so
the direction fix can be checked concretely, not just via the aggregate
correlation.

CPU-only, pure numpy -- no GPU needed.
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
from persona_drift.modeling.evaluate import rollout_output_error  # noqa: E402
from persona_drift.modeling.interaction_lift import InteractionLiftedSurrogate, augment_with_interaction  # noqa: E402
from persona_drift.modeling.koopman import KoopmanSurrogate, no_extra_features  # noqa: E402

_PHASE_H_FAILURE_TIDS = {"safemtdata_0074__seed0", "safemtdata_0476__seed0"}
_PHASE_H_CONTROL_TIDS = {"safemtdata_0074__seed1", "safemtdata_0169__seed0", "safemtdata_0169__seed1", "safemtdata_0530__seed0", "safemtdata_0530__seed1"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--rows-path", type=pathlib.Path, default=pathlib.Path("outputs/koopman_defense_phaseB_random_excite/trajectories.jsonl")
    )
    parser.add_argument(
        "--phase-e-koopman-mpc-path",
        type=pathlib.Path,
        default=pathlib.Path("outputs/koopman_defense_phaseE_koopman_mpc/trajectories.jsonl"),
    )
    parser.add_argument("--mu", type=int, default=2)
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument("--held-out-frac", type=float, default=0.25)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument("--repeat-penalty", type=float, default=0.0)
    parser.add_argument(
        "--pad-short-history",
        action="store_true",
        help="KoopmanMPCController.pad_short_history -- zero-pad the mu lag window instead of defaulting to 0 "
        "when there's real y but not yet mu real actions (see control.py docstring, docs/next_step_diagnosis.md 2026-09-02).",
    )
    parser.add_argument(
        "--replay-path",
        type=pathlib.Path,
        default=None,
        help="defaults to --phase-e-koopman-mpc-path; pass Phase I's own real trajectories.jsonl to replay against "
        "the states this exact v-aligned policy actually produced instead of the original koopman_mpc's.",
    )
    parser.add_argument("--out-path", type=pathlib.Path, default=pathlib.Path("outputs/koopman_case_study/mu2_interaction_replay_report.json"))
    return parser.parse_args()


def _fit_arx(rows, config, ridge):
    dataset = build_identification_dataset(rows, config, y_col="y_safety")
    return KoopmanSurrogate(extra_features_fn=no_extra_features, ridge=ridge).fit(dataset)


def _fit_interaction(rows, config, ridge):
    dataset = build_identification_dataset(rows, config, y_col="y_safety")
    v_aug = augment_with_interaction(dataset["V"], dataset["Z"], state_index=0)
    model = KoopmanSurrogate(extra_features_fn=no_extra_features, ridge=ridge).fit({**dataset, "V": v_aug})
    return InteractionLiftedSurrogate(surrogate=model, state_index=0), model


def _replay_phase_e(surrogate, config, horizon, repeat_penalty, phase_e_path, pad_short_history=False):
    controller = KoopmanMPCController(
        surrogate=surrogate, state_config=config, horizon=horizon, repeat_penalty=repeat_penalty, pad_short_history=pad_short_history
    )
    arm_rows = group_by_trajectory(load_trajectories(phase_e_path))
    records = []
    for tid, rows_ in arm_rows.items():
        for i in range(len(rows_)):
            history = rows_[:i]
            z = controller._current_state(history)
            if z is None:
                continue
            value_0 = controller._simulate(z, 0, horizon - 1)
            value_1 = controller._simulate(z, 1, horizon - 1)
            records.append(
                {
                    "trajectory_id": tid,
                    "turn": rows_[i]["turn"],
                    "y_probe": float(z[0]),
                    "margin": value_1 - value_0,
                    "action": int(value_1 > value_0),
                }
            )
    return records


def main() -> None:
    args = parse_args()
    rows = load_trajectories(args.rows_path)
    split = split_by_system_prompt_id(
        rows, train_frac=1.0 - args.held_out_frac, val_frac=0.0, seed=args.split_seed, split_col="attack_id"
    )
    train_rows, held_out_rows = split["train"], split["test"]

    old_config = ReducedStateConfig(nu=1, mu=args.mu, contemporaneous_v=False)
    new_config = ReducedStateConfig(nu=1, mu=args.mu, contemporaneous_v=True)

    old_model = _fit_arx(train_rows, old_config, args.ridge)
    new_model = _fit_arx(train_rows, new_config, args.ridge)
    old_mse = rollout_output_error(old_model, held_out_rows, old_config, y_col="y_safety")
    new_mse = rollout_output_error(new_model, held_out_rows, new_config, y_col="y_safety")

    print(f"=== plain ARX, nu=1, mu={args.mu}: old vs new alignment ===")
    print(f"old: B={old_model.B.tolist()}  held_out_rollout_mse={old_mse:.6f}")
    print(f"new: B={new_model.B.tolist()}  held_out_rollout_mse={new_mse:.6f}")

    new_interaction, new_interaction_raw = _fit_interaction(train_rows, new_config, args.ridge)
    print(f"\n=== interaction model, nu=1, mu={args.mu}, new alignment ===")
    print(f"B={new_interaction_raw.B.tolist()}  (col 2 = interaction coefficient v*y)")

    replay_path = args.replay_path or args.phase_e_koopman_mpc_path
    records = _replay_phase_e(
        new_interaction, new_config, args.horizon, args.repeat_penalty, replay_path, pad_short_history=args.pad_short_history
    )
    margins = np.array([r["margin"] for r in records])
    y_probes = np.array([r["y_probe"] for r in records])
    actions = np.array([r["action"] for r in records])
    corr = float(np.corrcoef(margins, y_probes)[0, 1]) if len(records) > 1 and margins.std() > 0 else None

    print(f"\nreal-state offline replay ({replay_path}, pad_short_history={args.pad_short_history}): n={len(records)}")
    print(f"margin std={margins.std():.6f} min={margins.min():.4f} max={margins.max():.4f}")
    print(f"n_remind={int(actions.sum())}/{len(actions)} (genuine mix: {0 < actions.sum() < len(actions)})")
    print(f"corr(margin, y_probe)={corr}")

    by_turn: dict[int, list[dict]] = {}
    for r in records:
        by_turn.setdefault(r["turn"], []).append(r)
    print("\ndecisions by turn (does pad_short_history actually unlock earlier action?):")
    for turn in sorted(by_turn):
        recs = by_turn[turn]
        n_remind = sum(r["action"] for r in recs)
        print(f"  turn={turn}: n={len(recs)} n_remind={n_remind} mean_y={np.mean([r['y_probe'] for r in recs]):.3f}")

    print("\n=== Phase H's named failure/control trajectories, this model's decision at every replayed turn ===")
    by_tid: dict[str, list[dict]] = {}
    for r in records:
        by_tid.setdefault(r["trajectory_id"], []).append(r)
    for tid in sorted(_PHASE_H_FAILURE_TIDS | _PHASE_H_CONTROL_TIDS):
        label = "FAILURE (Phase H wrongly skipped turn4/5 here)" if tid in _PHASE_H_FAILURE_TIDS else "control (Phase H correctly reminded here)"
        recs = by_tid.get(tid, [])
        if not recs:
            print(f"  {tid} [{label}]: no replayed decisions (not in this arm's file or too short)")
            continue
        detail = ", ".join(f"turn{r['turn']}: y={r['y_probe']:.2f} margin={r['margin']:+.4f} action={r['action']}" for r in recs)
        print(f"  {tid} [{label}]: {detail}")

    report = {
        "config": {**{k: str(v) for k, v in vars(args).items()}},
        "arx_old": {"B": old_model.B.tolist(), "held_out_rollout_mse": old_mse},
        "arx_new": {"B": new_model.B.tolist(), "held_out_rollout_mse": new_mse},
        "interaction_new": {"B": new_interaction_raw.B.tolist()},
        "phase_e_replay": {
            "n_records": len(records),
            "margin_std": float(margins.std()) if len(records) else None,
            "n_remind": int(actions.sum()) if len(records) else None,
            "corr_margin_y_probe": corr,
            "records": records,
        },
    }
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nreport written to {args.out_path}")


if __name__ == "__main__":
    main()
