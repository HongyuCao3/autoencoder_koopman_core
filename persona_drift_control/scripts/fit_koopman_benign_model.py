#!/usr/bin/env python3
"""Detection-design option 3, step 1 (docs/experiments/koopman_detection_design.md):
fit a "benign regime" Koopman surrogate on y_safety dynamics, to pair with the
existing "attack regime" model (Phase C,
outputs/koopman_defense_phaseB_random_excite/koopman_fit_report.json) for a
two-regime residual-comparison detector.

No new data collection needed: docs/experiments/koopman_defense_pilot.md's
Phase F already ran benign MT-Bench conversations through the same
safety_judge used for attack trajectories (to measure the defense
controllers' helpfulness cost), so its trajectories.jsonl files already carry
y_safety/u_remind in the exact schema modeling.dataset expects.

Combines all four Phase F controller arms (zero_control/constant_remind/
threshold/koopman_mpc) into one identification dataset for input diversity
(u_remind varies differently per arm). Phase F reuses the SAME trajectory_id
across arms (identical benign session content, only u_remind differs) --
trajectory_id is arm-qualified here before combining, otherwise
group_by_trajectory would silently merge four arms' rows into one corrupted
per-session sequence.

Also supports detection-design option 4
(docs/experiments/koopman_detection_design.md): pass
`--aux-cols attack_similarity` to lift each benign turn's `question` text
into the SAME reference corpus scripts/fit_koopman_defense_model.py built for
the attack-regime model (read from `--content-reference-report`'s
`content_reference_texts`, not re-derived) -- so both regimes' models see the
"does this look like a known attack" feature computed the identical way.

CPU-only, pure numpy/pandas -- no GPU needed. Run directly (no sbatch).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from persona_drift.modeling.content_similarity import annotate_similarity, fit_tfidf_corpus  # noqa: E402
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

ARMS = ["zero_control", "constant_remind", "threshold", "koopman_mpc"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase-f-dir-template",
        default="outputs/koopman_defense_phaseF_{arm}/trajectories.jsonl",
    )
    parser.add_argument("--nu", type=int, default=1)
    parser.add_argument("--mu", type=int, default=2)
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument("--controllability-horizon", type=int, default=5)
    parser.add_argument("--held-out-frac", type=float, default=0.25)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument(
        "--contemporaneous-v",
        action="store_true",
        help=(
            "fit with ReducedStateConfig.contemporaneous_v=True -- must match the attack-regime "
            "report's setting for evaluate_koopman_detector.py's comparison to be meaningful."
        ),
    )
    parser.add_argument(
        "--aux-cols",
        nargs="*",
        default=[],
        choices=["attack_similarity"],
        help="detection-design option 4: lift extra content-derived features into z",
    )
    parser.add_argument(
        "--content-reference-report",
        type=pathlib.Path,
        default=pathlib.Path("outputs/koopman_detection_content_feature/attack_fit_report.json"),
        help="attack-regime fit report to read content_reference_texts from (only used if --aux-cols attack_similarity)",
    )
    parser.add_argument(
        "--out-path",
        type=pathlib.Path,
        default=pathlib.Path("outputs/koopman_detection_benign_baseline/koopman_fit_report.json"),
    )
    return parser.parse_args()


def load_all_arms(template: str) -> list[dict]:
    """Load all four Phase F arms, arm-qualifying trajectory_id so rows from
    different controllers (which share the underlying benign session content
    and its original trajectory_id) never collide in group_by_trajectory."""

    rows: list[dict] = []
    for arm in ARMS:
        for row in load_trajectories(pathlib.Path(template.format(arm=arm))):
            row = dict(row)
            row["trajectory_id"] = f"{arm}__{row['trajectory_id']}"
            rows.append(row)
    return rows


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
    rows = load_all_arms(args.phase_f_dir_template)

    split = split_by_system_prompt_id(
        rows,
        train_frac=1.0 - args.held_out_frac,
        val_frac=0.0,
        seed=args.split_seed,
        split_col="benign_id",
    )
    train_rows = split["train"]
    held_out_rows = split["test"]
    n_train_benign = len({r["benign_id"] for r in train_rows})
    held_out_benign_ids = sorted({r["benign_id"] for r in held_out_rows})

    if "attack_similarity" in args.aux_cols:
        # Same frozen reference corpus the attack-regime model used -- NOT
        # refit from benign text, so both regimes' models measure "resembles
        # a known attack" against the identical reference.
        reference_texts = json.loads(args.content_reference_report.read_text())["content_reference_texts"]
        corpus = fit_tfidf_corpus(reference_texts)
        train_rows = annotate_similarity(train_rows, "question", corpus, out_col="attack_similarity")
        held_out_rows = annotate_similarity(held_out_rows, "question", corpus, out_col="attack_similarity")

    config = ReducedStateConfig(
        nu=args.nu, mu=args.mu, aux_cols=tuple(args.aux_cols), contemporaneous_v=args.contemporaneous_v
    )

    arx_report, arx_model = _fit_and_evaluate(
        "arx", no_extra_features, train_rows, held_out_rows, config, args.ridge
    )
    richer_report, richer_model = _fit_and_evaluate(
        "richer_abs_sign", abs_sign_extra_features, train_rows, held_out_rows, config, args.ridge
    )

    controllability = arx_model.controllability(args.controllability_horizon)

    report = {
        "config": {
            "nu": args.nu,
            "mu": args.mu,
            "ridge": args.ridge,
            "source": "phaseF_all_arms_combined",
            "aux_cols": list(args.aux_cols),
            "contemporaneous_v": args.contemporaneous_v,
        },
        "n_train_benign_categories": n_train_benign,
        "n_held_out_benign_categories": len(held_out_benign_ids),
        "held_out_benign_ids": held_out_benign_ids,
        "arx": arx_report,
        "richer_abs_sign": richer_report,
        "controllability_arx": controllability,
    }
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_path.write_text(json.dumps(report, indent=2))

    print(f"n_train_benign_categories={n_train_benign} n_held_out_benign_categories={len(held_out_benign_ids)}")
    print(f"held_out_benign_ids={held_out_benign_ids}")
    print(f"ARX A={arx_model.A.tolist()} B={arx_model.B.tolist()} b={arx_model.b.tolist()}")
    print(f"ARX held_out_rollout_mse={arx_report['held_out_rollout_mse']:.6f}")
    print(f"richer held_out_rollout_mse={richer_report['held_out_rollout_mse']:.6f}")
    print(f"controllability_rank={controllability['controllability_rank']} (state_dim={arx_model.state_dim})")
    print(f"report written to {args.out_path}")


if __name__ == "__main__":
    main()
