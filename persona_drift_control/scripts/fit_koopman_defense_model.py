#!/usr/bin/env python3
"""Phase C of docs/experiments/koopman_defense_pilot.md: fit a Koopman
surrogate on Phase B's open-loop random-excitation trajectories
(y_safety/u_remind instead of persona-drift's y_probe/u_remind), evaluate
one-step/rollout error on a held-out attack_id split, and run
controllability diagnostics on B -- the go/no-go gate before designing a
KoopmanMPCController on top of it.

Also supports detection-design option 4
(docs/experiments/koopman_detection_design.md): pass
`--aux-cols attack_similarity` to lift `attacker_query` text into an extra
state dimension via modeling.content_similarity, measuring resemblance to a
reference corpus built ONLY from this run's own train-split attacks (never
the held-out ones, so a held-out attack's near-identical replayed text can't
leak into its own reference corpus). The reference corpus's texts and the
`aux_cols` list are written into the report so
scripts/fit_koopman_benign_model.py and scripts/evaluate_koopman_detector.py
can reconstruct an identical corpus without re-deriving the split.

CPU-only, pure numpy/pandas -- no GPU needed. Run directly (no sbatch).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from persona_drift.modeling.content_similarity import (  # noqa: E402
    annotate_similarity,
    fit_tfidf_corpus,
    reference_texts_excluding_ids,
)
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
        "--contemporaneous-v",
        action="store_true",
        help=(
            "fit with ReducedStateConfig.contemporaneous_v=True (pairs v with the SAME turn's "
            "y it acts on -- see docs/experiments/koopman_case_study_design.md's Phase I). "
            "Default False preserves the original Phase C report exactly."
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
        "--out-path", type=pathlib.Path, default=pathlib.Path("outputs/koopman_defense_phaseB_random_excite/koopman_fit_report.json")
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=0,
        help=(
            "0 (default) = original single held-out split only, report unchanged. >0 adds a "
            "`fold_evaluation` block: that many attack-disjoint folds, each reporting the "
            "surrogate's SCALAR y one-step MSE against three trivial nulls "
            "(docs/experiments/koopman_fast_track_plan.md gate G-K2-1). The RC-1 lesson is that "
            "an MSE without a null is unreadable, and the ERGO lesson is that the null must get "
            "every deterministic exogenous quantity the model gets -- here that is `turn`."
        ),
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


# ---------------------------------------------------------------------------
# Null baselines for gate G-K2-1 (docs/experiments/koopman_fast_track_plan.md).
#
# Two things here are deliberate and must not be "simplified":
#
# 1. The comparison target is the SCALAR y_(t+1), not the full z_(t+1) that
#    modeling.evaluate.one_step_error scores. z_next's v-block is just this
#    trajectory's own already-realized inputs shifted in by one slot -- the
#    surrogate reproduces it exactly and for free, so a full-z MSE flatters the
#    surrogate against any null that only ever predicts y. y_(t+1) sits at index
#    `nu - 1` of z_next (the same slot modeling.evaluate.rollout_output_error
#    reads), so this stays consistent with the existing harness without touching
#    it.
#
# 2. The nulls get `turn`. On the d1 pool y_safety erodes 0.90 -> 0.43 across
#    the five turns, so a turn-indexed mean is the strongest trivial predictor
#    and it is exactly the dominant Koopman mode -- beating it is the whole
#    question. Denying the null a deterministic exogenous quantity the model has
#    is the mirror image of the ERGO RC-1 defect (there the null was HANDED one
#    the model had to extrapolate), and it would manufacture a pass.
# ---------------------------------------------------------------------------


def _aligned_transitions(rows, config, y_col="y_safety"):
    """(y_now, v, y_next, turn_next) per transition, aligned 1:1 and in the same
    order as `build_identification_dataset(rows, config, y_col=y_col)`.

    `turn` is carried by appending it to `aux_cols`, not by re-deriving the
    index arithmetic of `build_reduced_state_pairs` here -- a hand-rolled copy
    of that loop would silently drift from it (and from its NaN filter) the
    first time either changes. `turn` is never NaN, so the aux-extended config
    yields exactly the same pair set; turn_(t+1) is then z_next's last element.
    """
    aux_config = ReducedStateConfig(
        nu=config.nu,
        mu=config.mu,
        aux_cols=tuple(config.aux_cols) + ("turn",),
        contemporaneous_v=config.contemporaneous_v,
    )
    y_now, v, y_next, turn_next = [], [], [], []
    for traj_rows in group_by_trajectory(rows, id_col="trajectory_id").values():
        for pair in build_reduced_state_pairs(traj_rows, aux_config, y_col=y_col, u_col="u_remind"):
            y_now.append(float(pair["z"][config.nu - 1]))
            v.append(float(pair["v"][0]))
            y_next.append(float(pair["z_next"][config.nu - 1]))
            turn_next.append(float(pair["z_next"][-1]))
    return (
        np.array(y_now),
        np.array(v),
        np.array(y_next),
        np.array(turn_next),
    )


def _null_predictions(train, test):
    """The three trivial predictors of y_(t+1), fit on `train`, applied to
    `test`. Each is a dict of the arrays returned by `_aligned_transitions`."""
    const = float(np.mean(train["y_next"]))

    turn_means = {
        t: float(np.mean(train["y_next"][train["turn_next"] == t]))
        for t in np.unique(train["turn_next"])
    }
    turn_mean = np.array([turn_means.get(t, const) for t in test["turn_next"]])

    # stateless: y_(t+1) ~ 1 + turn + u, i.e. everything except the state.
    X_train = np.column_stack([np.ones(len(train["y_next"])), train["turn_next"], train["v"]])
    beta, *_ = np.linalg.lstsq(X_train, train["y_next"], rcond=None)
    X_test = np.column_stack([np.ones(len(test["y_next"])), test["turn_next"], test["v"]])
    stateless = X_test @ beta

    return {
        "const": np.full(len(test["y_next"]), const),
        "turn_mean": turn_mean,
        "stateless": stateless,
    }


def _fold_report(train_rows, test_rows, config, ridge):
    """One attack-disjoint fold: scalar-y one-step MSE for both surrogates and
    for the three nulls, all on the identical transition set."""
    train_dataset = build_identification_dataset(train_rows, config, y_col="y_safety")
    test_dataset = build_identification_dataset(test_rows, config, y_col="y_safety")
    if train_dataset["Z"].shape[0] == 0 or test_dataset["Z"].shape[0] == 0:
        return None

    keys = ("y_now", "v", "y_next", "turn_next")
    train_aligned = dict(zip(keys, _aligned_transitions(train_rows, config)))
    test_aligned = dict(zip(keys, _aligned_transitions(test_rows, config)))
    assert len(test_aligned["y_next"]) == test_dataset["Z"].shape[0], "aux-turn alignment drifted"

    out = {"n_train_transitions": train_dataset["Z"].shape[0], "n_test_transitions": len(test_aligned["y_next"])}
    for name, extra_features_fn in (("arx", no_extra_features), ("richer_abs_sign", abs_sign_extra_features)):
        model = KoopmanSurrogate(extra_features_fn=extra_features_fn, ridge=ridge).fit(train_dataset)
        pred = np.array(
            [model.step(z, v)[config.nu - 1] for z, v in zip(test_dataset["Z"], test_dataset["V"])]
        )
        out[f"{name}_y_one_step_mse"] = float(np.mean((pred - test_aligned["y_next"]) ** 2))

    for name, pred in _null_predictions(train_aligned, test_aligned).items():
        out[f"null_{name}_y_one_step_mse"] = float(np.mean((pred - test_aligned["y_next"]) ** 2))

    out["best_null_y_one_step_mse"] = min(
        out[f"null_{name}_y_one_step_mse"] for name in ("const", "turn_mean", "stateless")
    )
    for name in ("arx", "richer_abs_sign"):
        out[f"{name}_beats_best_null"] = bool(out[f"{name}_y_one_step_mse"] < out["best_null_y_one_step_mse"])
    return out


def _fold_evaluation(rows, config, ridge, n_folds, split_seed):
    """`n_folds` attack-disjoint folds. Splitting by attack_id (not by turn and
    not by trajectory) is the same anti-leak rule the single split already
    uses -- turns inside one dialogue are dependent, and two seeds of one attack
    replay the same attacker text."""
    attack_ids = sorted({row["attack_id"] for row in rows})
    rng = np.random.default_rng(split_seed)
    shuffled = list(rng.permutation(attack_ids))
    groups = [shuffled[i::n_folds] for i in range(n_folds)]

    folds = []
    for held_out in groups:
        if not held_out:
            continue
        held_out_set = set(held_out)
        fold = _fold_report(
            [r for r in rows if r["attack_id"] not in held_out_set],
            [r for r in rows if r["attack_id"] in held_out_set],
            config,
            ridge,
        )
        if fold is None:
            continue
        fold["held_out_attack_ids"] = sorted(held_out_set)
        folds.append(fold)

    summary = {"n_folds": len(folds), "folds": folds}
    for name in ("arx", "richer_abs_sign"):
        summary[f"{name}_n_folds_beating_best_null"] = sum(
            1 for f in folds if f[f"{name}_beats_best_null"]
        )
        summary[f"{name}_mean_y_one_step_mse"] = float(
            np.mean([f[f"{name}_y_one_step_mse"] for f in folds])
        )
    for name in ("const", "turn_mean", "stateless"):
        summary[f"null_{name}_mean_y_one_step_mse"] = float(
            np.mean([f[f"null_{name}_y_one_step_mse"] for f in folds])
        )
    return summary


def main() -> None:
    args = parse_args()
    rows = load_trajectories(args.rows_path)

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
    held_out_attack_ids = sorted({r["attack_id"] for r in held_out_rows})
    n_held_out_attacks = len(held_out_attack_ids)

    reference_texts = None
    if "attack_similarity" in args.aux_cols:
        # Corpus built ONLY from train_rows -- held_out_attack_ids' own text
        # never enters it, so a held-out attack can't trivially match itself.
        reference_texts = reference_texts_excluding_ids(
            train_rows, exclude_ids=set(held_out_attack_ids), text_col="attacker_query", id_col="attack_id"
        )
        corpus = fit_tfidf_corpus(reference_texts)
        train_rows = annotate_similarity(train_rows, "attacker_query", corpus, out_col="attack_similarity")
        held_out_rows = annotate_similarity(held_out_rows, "attacker_query", corpus, out_col="attack_similarity")

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
            "rows_path": str(args.rows_path),
            "aux_cols": list(args.aux_cols),
            "contemporaneous_v": args.contemporaneous_v,
        },
        "n_train_attacks": n_train_attacks,
        "n_held_out_attacks": n_held_out_attacks,
        "held_out_attack_ids": held_out_attack_ids,
        "content_reference_texts": reference_texts,
        "arx": arx_report,
        "richer_abs_sign": richer_report,
        "controllability_arx": controllability,
    }
    if args.n_folds > 0:
        report["fold_evaluation"] = _fold_evaluation(rows, config, args.ridge, args.n_folds, args.split_seed)
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_path.write_text(json.dumps(report, indent=2))

    print(f"n_train_attacks={n_train_attacks} n_held_out_attacks={n_held_out_attacks}")
    print(f"held_out_attack_ids={held_out_attack_ids}")
    print(f"ARX: A={arx_model.A.tolist()} B={arx_model.B.tolist()} b={arx_model.b.tolist()}")
    print(f"ARX held_out_rollout_mse={arx_report['held_out_rollout_mse']:.6f}")
    print(f"richer held_out_rollout_mse={richer_report['held_out_rollout_mse']:.6f}")
    print(f"controllability_rank={controllability['controllability_rank']} (state_dim={arx_model.state_dim})")
    print(f"gramian_condition={controllability['gramian_condition']:.4e}")
    print(f"A_spectral_radius={controllability['spectral_radius']:.4f}")
    if args.n_folds > 0:
        fold = report["fold_evaluation"]
        print(f"\nfold_evaluation ({fold['n_folds']} attack-disjoint folds), SCALAR y one-step MSE:")
        for name in ("const", "turn_mean", "stateless"):
            print(f"  null {name:<10} mean={fold[f'null_{name}_mean_y_one_step_mse']:.6f}")
        for name in ("arx", "richer_abs_sign"):
            print(
                f"  {name:<15} mean={fold[f'{name}_mean_y_one_step_mse']:.6f}  "
                f"beats best null in {fold[f'{name}_n_folds_beating_best_null']}/{fold['n_folds']} folds"
            )
    print(f"report written to {args.out_path}")


if __name__ == "__main__":
    main()
