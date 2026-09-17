#!/usr/bin/env python3
"""A1 / M1 / M2: the paper's side experiments on the eight core tasks.

Spec: docs/paper_side_experiments_plan_2026-09-16.md. Three probes, one file,
because all three need the same setup (load, delay-embed, fit the affine
Koopman, roll out from a shared observed prefix, score against the same
trivial null) and only diverge in what they vary.

    A1  memory depth sweep      -- lag 0..lag_max at a FIXED seed turn.
    M1  spectrum + response     -- eigenvalues of A, response of B, plus the
                                   rank check that says whether B is a
                                   measurement or an extrapolation.
    M2  item-intercept swap     -- give the Markov row the per-trajectory
                                   level, estimated outside the scored window,
                                   and see whether the memory gain survives.

Additive: imports `scripts/eval_surrogate_rows.py` rather than editing it, so
Table 1's code path is untouched and every cell here comes out of the same
scorer (`src/surrogate_eval`). Writes to three new directories under
`results/`; existing artifacts are never overwritten.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib

import numpy as np
import pandas as pd

from koopman_ae import (
    AugmentedKoopmanModel,
    AugmentedStateConfig,
    build_augmented_state_dataset,
    rollout_augmented_from_trajectories,
)
from surrogate_eval import best_null

PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[1]

_SPEC = importlib.util.spec_from_file_location(
    "_eval_surrogate_rows", PACKAGE_ROOT / "scripts" / "eval_surrogate_rows.py")
ev = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ev)

OUT_DIRS = {
    "a1": "ablation_memory_depth",
    "a1_matched": "ablation_memory_depth_matched",
    "m1": "mechanism_spectrum",
    "m2": "mechanism_item_effect",
}
RIDGE_ALPHA = 1e-6
# A control channel is identified only if the regressor is not a function of
# the state. Candidate A (2026-09-15) died on exactly this: `core`'s control is
# the tracking error, an exact affine function of the state, so its cross terms
# were undefined rather than insignificant. Anything above this threshold on
# the per-dimension R^2 of R against [Z|1] means the same thing here.
COLLINEAR_R2 = 1.0 - 1e-8
NEAR_COLLINEAR_R2 = 0.99


class LeakageError(AssertionError):
    """The item level was estimated on turns that are also scored."""


def deep_lag(n_turns: int) -> int:
    """Deepest lag that still leaves MIN_HORIZON rollout steps.

    `ev.choose_lag` only ever answers 3 or 1, because Table 1 pins the lag to
    the repository default where the published numbers live. The sweep needs
    to know how deep the data would actually allow.
    """
    lag = n_turns - 1 - ev.MIN_HORIZON
    if lag < 0:
        raise ValueError(f"no lag leaves >= {ev.MIN_HORIZON} rollout steps at T={n_turns}")
    return lag


def _state_config(lag: int, output_columns, target_columns) -> AugmentedStateConfig:
    return AugmentedStateConfig(
        output_memory=lag + 1, input_memory=0, control_mode="error",
        output_columns=tuple(output_columns), target_columns=tuple(target_columns),
    )


def _fit(train_frame: pd.DataFrame, cfg: AugmentedStateConfig, *, min_turn: int | None = None):
    """Affine Koopman on one delay-embedded training set.

    `min_turn` drops transitions whose TARGET turn is inside the observed
    prefix. M2 needs that: its extra regressor is the mean of y over that
    prefix, so a transition landing inside the prefix would let the regressor
    predict itself and the fitted weight on the item level would come out
    larger than its real predictive value -- which biases the comparison
    toward the answer this project wants, not away from it.
    """
    dataset = build_augmented_state_dataset(train_frame, cfg)
    Z_t, R, Z_next = dataset.Z_t, dataset.R, dataset.Z_next
    if min_turn is not None:
        keep = dataset.meta["next_turn"].to_numpy(dtype=int) > min_turn
        Z_t, R, Z_next = Z_t[keep], R[keep], Z_next[keep]
    model = AugmentedKoopmanModel(output_dim=dataset.output_dim, alpha=RIDGE_ALPHA)
    model.fit(Z_t, R, Z_next)
    return model, dataset, (Z_t, R, Z_next)


def _panel(frame: pd.DataFrame, output_columns, target_columns, seed_turn: int, deepest_cfg):
    """The row set, the truth, and the trivial null every probe scores against.

    Identical to Table 1's panel by construction: same scorer, same null
    family, same by-trajectory resampling unit. Only the seed turn is a
    parameter here, and A1 is the reason it has to be.
    """
    train_frame, _, test_frame = (ev._split(frame, s) for s in ("train", "validation", "test"))
    seed_model, _, _ = _fit(train_frame, deepest_cfg)
    reference = rollout_augmented_from_trajectories(
        seed_model, test_frame, deepest_cfg, observed_seed_turns=seed_turn)
    reference = reference[~reference["uses_observed_seed"]].reset_index(drop=True)
    y_true = ev._stacked(reference, "y_true")
    groups = np.repeat(reference["trajectory_id"].to_numpy(), y_true.shape[1])
    nulls = ev._null_panels(train_frame, reference, output_columns, target_columns, seed_turn)
    null_name, y_null = best_null(
        {k: v.reshape(-1) for k, v in nulls.items()}, y_true.reshape(-1))
    y_null = y_null.reshape(y_true.shape)
    return {
        "train_frame": train_frame,
        "test_frame": test_frame,
        "reference": reference,
        "keys": list(zip(reference["trajectory_id"].astype(str), reference["turn"].astype(int))),
        "y_true": y_true,
        "groups": groups,
        "se_null": ((y_null - y_true) ** 2).reshape(-1),
        "null_name": null_name,
        "scored_turns": sorted(set(reference["turn"].astype(int))),
    }


def _score(model, cfg, panel, seed_turn: int, horizon: int, test_frame=None) -> tuple[dict, np.ndarray]:
    pred = rollout_augmented_from_trajectories(
        model, test_frame if test_frame is not None else panel["test_frame"],
        cfg, observed_seed_turns=seed_turn)
    pred = pred[~pred["uses_observed_seed"]].reset_index(drop=True)
    keys = list(zip(pred["trajectory_id"].astype(str), pred["turn"].astype(int)))
    if keys != panel["keys"]:
        raise AssertionError("scored on different rows than the reference rollout")
    y_pred = ev._stacked(pred, "y_pred")[:, : panel["y_true"].shape[1]]
    se = ((y_pred - panel["y_true"]) ** 2).reshape(-1)
    return ev._cell(se, panel["se_null"], panel["groups"], horizon, 0), se


# --------------------------------------------------------------------------- A1

def run_a1_frame(frame, output_columns, target_columns, task: str, anchor: str,
                 *, matched_window: bool = False) -> dict:
    """lag 0..lag_max, all rolled out from the SAME observed prefix.

    The fixed seed turn is the whole design. `eval_surrogate_rows` computes
    `horizon = n_turns - (lag + 1)`, so sweeping lag there would move the
    horizon at the same time and `Skill_H` would not be comparable between two
    points on the curve -- a depth reading that is really a depth-and-horizon
    reading. Here lag_max pins the seed turn and every shallower model simply
    uses less of the same observed prefix.

    Two anchors, because two signed requirements pull in different directions:
    `primary` keeps Table 1's lag and horizon, so the deepest grid point has to
    reproduce the landed `ours` cell to the last digit (that is the drift
    guard); `deep` spends the horizon on depth instead, and answers whether
    anything beyond Table 1's lag was ever on the table. Cells from the two
    anchors are never compared -- different H.
    """
    n_turns = int(frame["turn"].max())
    lag_max = ev.choose_lag(n_turns) if anchor == "primary" else deep_lag(n_turns)
    seed_turn = lag_max + 1
    horizon = n_turns - seed_turn
    deepest_cfg = _state_config(lag_max, output_columns, target_columns)
    panel = _panel(frame, output_columns, target_columns, seed_turn, deepest_cfg)

    # MATCHED WINDOW, added 2026-09-16 after the first run. A shallow state is
    # usable earlier in a trajectory, so by default the lag-0 row is fit on
    # transitions the lag-3 row never sees -- on `sentence_length_t10`, three
    # extra early transitions per trajectory. Dropping them moves that column's
    # Markov skill from +0.166 to +0.660, which is most of what Table 1's row 2
    # reads as "memory is needed". Both variants are run and reported: the
    # default is Table 1's convention (each class gets all the data its state
    # permits), the matched one isolates depth from the training window.
    min_turn = seed_turn if matched_window else None
    rows, squared_error = {}, {}
    for lag in range(lag_max + 1):
        cfg = _state_config(lag, output_columns, target_columns)
        model, dataset, (Z_fit, _, _) = _fit(panel["train_frame"], cfg, min_turn=min_turn)
        cell, se = _score(model, cfg, panel, seed_turn, horizon)
        cell["lag"] = lag
        cell["state_dim"] = int(model.A.shape[0])
        cell["n_train_transitions"] = int(len(Z_fit))
        rows[f"lag_{lag}"] = cell
        squared_error[f"lag_{lag}"] = se

    contrasts = {}
    for lag in range(1, lag_max + 1):
        if rows["lag_0"].get("skill_h") is None or rows[f"lag_{lag}"].get("skill_h") is None:
            continue
        # Positive = the deeper state is better. Against lag 0 this is Table 1
        # row 6 minus row 2; against lag-1 it is the increment, which is what
        # says where the curve stops buying anything.
        contrasts[f"lag_{lag}_minus_lag_0"] = ev._contrast(
            squared_error[f"lag_{lag}"], squared_error["lag_0"], panel["se_null"], panel["groups"], 0)
        contrasts[f"lag_{lag}_minus_lag_{lag - 1}"] = ev._contrast(
            squared_error[f"lag_{lag}"], squared_error[f"lag_{lag - 1}"],
            panel["se_null"], panel["groups"], 0)

    return {
        "probe": "a1",
        "task": task,
        "anchor": anchor,
        "matched_window": bool(matched_window),
        "provenance": ev.provenance(),
        "n_turns": n_turns,
        "lag_max": lag_max,
        "observed_seed_turns": seed_turn,
        "horizon": horizon,
        "scan_space": "none" if lag_max < 2 else "yes",
        "n_test_trajectories": int(panel["reference"]["trajectory_id"].nunique()),
        "n_scored_rows": int(panel["y_true"].size),
        "best_null_name": panel["null_name"],
        "rows": rows,
        "contrasts": contrasts,
    }


# --------------------------------------------------------------------------- M1

def _identifiability(Z_t: np.ndarray, R: np.ndarray) -> dict:
    """Two rows of rank check, run before any B-derived number is emitted.

    P1-a, signed 2026-09-16: rank-deficient or R^2 = 1 means the question is
    undefined, not that the answer is small. Candidate A reported a headroom of
    zero on `core` before this check existed, and the Limitations sentence
    written from it had to be withdrawn.
    """
    X = np.column_stack([Z_t, R, np.ones(len(Z_t))])
    rank = int(np.linalg.matrix_rank(X))
    singular = np.linalg.svd(X, compute_uv=False)
    ZO = np.column_stack([Z_t, np.ones(len(Z_t))])
    coef, *_ = np.linalg.lstsq(ZO, R, rcond=None)
    resid = R - ZO @ coef
    ss_tot = ((R - R.mean(axis=0)) ** 2).sum(axis=0)
    r2 = np.where(ss_tot > 0, 1.0 - (resid ** 2).sum(axis=0) / np.where(ss_tot > 0, ss_tot, 1.0), 1.0)
    identified = bool(rank == X.shape[1] and np.all(r2 < COLLINEAR_R2))
    out = {
        "n_columns": int(X.shape[1]),
        "rank": rank,
        "rank_deficient": bool(rank < X.shape[1]),
        "singular_values": [float(v) for v in singular],
        "r_on_state_r2": [float(v) for v in np.atleast_1d(r2)],
        "identified": identified,
        # Graded, because the binary rule is exact and real data never is. A
        # column that clears the rule at R^2 = 0.999 has an identified B and a
        # badly conditioned one, and the reader has to be able to tell that
        # apart from R^2 = 0.2 without refitting anything.
        "near_collinear": bool(np.any(r2 > NEAR_COLLINEAR_R2)),
        "condition_number": float(singular[0] / singular[-1]) if singular[-1] > 0 else None,
    }
    if not identified:
        out["undefined_reason"] = (
            "rank-deficient design matrix" if rank < X.shape[1]
            else "R is an exact affine function of [Z|1]")
    return out


def run_m1_frame(frame, output_columns, target_columns, task: str, *, action_channel: str) -> dict:
    """Eigenvalues of A and the response of B, on Table 1's own fit.

    The claim this probe exists to explain is that delay embedding is needed
    and re-timing the same actions is not. Both are properties of one operator:
    a slowly decaying state (long half-life -- memory pays) driven by an input
    whose effect lands and stops (short response -- planning does not). Putting
    the two on one axis is the mechanism; each alone is a diagnostic.

    Fit diagnostics only. Spectral radius, half-life and response length are
    fit quantities, and `.claude/global.md` does not let them stand as
    cross-arm headline numbers; they are read beside the landed `Skill_H`.
    """
    n_turns = int(frame["turn"].max())
    lag = ev.choose_lag(n_turns)
    seed_turn = lag + 1
    horizon = n_turns - seed_turn
    cfg = _state_config(lag, output_columns, target_columns)
    train_frame = ev._split(frame, "train")
    model, dataset, (Z_t, R, _) = _fit(train_frame, cfg)

    A, B = np.asarray(model.A), np.asarray(model.B)
    eigenvalues = np.linalg.eigvals(A)
    modulus = np.sort(np.abs(eigenvalues))[::-1]
    rho = float(modulus[0])
    half_life = float(np.log(0.5) / np.log(rho)) if 0.0 < rho < 1.0 else None

    # HOW MUCH OF y_(t+1) IS CARRIED BY TAPS OLDER THAN y_t.
    #
    # Added 2026-09-16 after the first run: the half-life alone does not say
    # whether memory pays. A slow AR(1) has a long half-life and is perfectly
    # Markov -- delay embedding buys nothing on it -- and the eight columns
    # contain that case (`formality_t5`, half-life 11.5, `ours - Markov` not
    # separable) next to its opposite (`character_length_t5`, half-life 0.8,
    # `ours - Markov` +0.235*). The quantity that matches the claim is the
    # weight the readout row puts on the older taps; the half-life is a
    # different property of the same operator and is reported beside it, not
    # in place of it.
    output_dim_ = int(dataset.output_dim)
    readout_row = A[:output_dim_, :]
    taps = [float(np.linalg.norm(readout_row[:, j * output_dim_:(j + 1) * output_dim_]))
            for j in range(lag + 1)]
    total = float(np.sqrt(sum(v ** 2 for v in taps)))
    older = float(np.sqrt(sum(v ** 2 for v in taps[1:])))

    ident = _identifiability(Z_t, R)
    out = {
        "probe": "m1",
        "task": task,
        "provenance": ev.provenance(),
        "lag": lag,
        "observed_seed_turns": seed_turn,
        "horizon": horizon,
        "state_dim": int(A.shape[0]),
        "n_train_transitions": int(len(Z_t)),
        "spectrum": {
            "eigenvalue_modulus": [float(v) for v in modulus],
            "spectral_radius": rho,
            "half_life_steps": half_life,
            "identified": True,
        },
        "lag_loading": {
            "tap_norms": taps,
            "non_markov_share": (older / total) if total > 0 else None,
            "rule": "||A[:d, tap j]||_F over taps j=0..lag; share is taps j>=1 over all taps",
        },
        "action_channel": action_channel,
        "identifiability": ident,
    }

    if not ident["identified"]:
        out["response"] = {
            "identified": False,
            "undefined_reason": ident["undefined_reason"],
            "impulse_response_norm": None,
            "step_response_norm": None,
            "response_length": None,
        }
        return out

    output_dim = output_dim_
    impulse, step = [], []
    power = np.eye(A.shape[0])
    for _ in range(horizon):
        impulse.append(float(np.linalg.norm((power @ B)[:output_dim, :])))
        power = A @ power
    cumulative = np.zeros_like(B)
    power = np.eye(A.shape[0])
    for _ in range(horizon):
        cumulative = cumulative + power @ B
        step.append(float(np.linalg.norm(cumulative[:output_dim, :])))
        power = A @ power
    peak = max(impulse) if impulse else 0.0
    # How many steps the input keeps arriving for. A one-step channel is the
    # structural reason re-timing a fixed budget buys nothing: by the time the
    # controller could have moved the action, its effect is already spent.
    length = int(sum(1 for v in impulse if peak > 0 and v >= 0.1 * peak))
    out["response"] = {
        "identified": True,
        "impulse_response_norm": impulse,
        "step_response_norm": step,
        "response_length": length,
        "response_length_rule": "steps with ||C A^(k-1) B|| >= 0.1 * peak",
    }
    if action_channel != "randomized_per_turn":
        out["response"]["caveat"] = (
            "the reference is constant within a trajectory, so a one-step impulse is never "
            "observed in this data; the impulse curve is a counterfactual read off the fit, "
            "and only the step response corresponds to a contrast the data contains")
    return out


# --------------------------------------------------------------------------- M2

def _with_item_level(frame: pd.DataFrame, output_columns, seed_turn: int, scored_turns) -> tuple[pd.DataFrame, list[str]]:
    """Per-trajectory mean of y over the observed prefix, as extra columns.

    The item level goes in through the target channel, which is the one place
    the state builder already treats as constant along a trajectory -- which is
    what a persistent per-item effect is. Raises rather than returns if the
    estimation window touches a scored turn: a leak here would not show up as a
    failure, it would show up as a stronger baseline and a quieter paper.
    """
    estimation_turns = sorted({t for t in frame["turn"].astype(int) if t <= seed_turn})
    overlap = set(estimation_turns) & set(int(t) for t in scored_turns)
    if overlap:
        raise LeakageError(
            f"item level estimated on turns {sorted(overlap)} which are also scored")
    prefix = frame[frame["turn"].astype(int) <= seed_turn]
    level = prefix.groupby("trajectory_id")[list(output_columns)].mean()
    level.columns = [f"item_level_{i}" for i in range(len(output_columns))]
    merged = frame.merge(level, on="trajectory_id", how="left")
    if merged[list(level.columns)].isna().any().any():
        raise ValueError("a trajectory has no observed prefix to estimate its item level from")
    return merged, list(level.columns)


def run_m2_frame(frame, output_columns, target_columns, task: str) -> dict:
    """Does the delay-embedding gain survive handing the Markov row item identity?

    Competing mechanism: the window may be estimating "which trajectory is
    this" rather than carrying dynamics across turns. Both sides of this are
    publishable -- if the gain survives, memory carries dynamics; if it
    collapses, the paper's one positive claim is about persistent item
    heterogeneity and the abstract's reading of it has to say so.

    All three rows are refit on the same restricted training set, so the panel
    is internally fair; it is therefore not Table 1's fit and its cells are not
    interchangeable with Table 1's.
    """
    n_turns = int(frame["turn"].max())
    lag = ev.choose_lag(n_turns)
    seed_turn = lag + 1
    horizon = n_turns - seed_turn
    deepest_cfg = _state_config(lag, output_columns, target_columns)
    panel = _panel(frame, output_columns, target_columns, seed_turn, deepest_cfg)

    item_frame, item_columns = _with_item_level(
        frame, output_columns, seed_turn, panel["scored_turns"])
    item_targets = tuple(target_columns) + tuple(item_columns)
    item_cfg = _state_config(0, output_columns, item_targets)
    item_train = ev._split(item_frame, "train")
    item_test = ev._split(item_frame, "test")

    specs = {
        "ours": (_state_config(lag, output_columns, target_columns), panel["train_frame"], None),
        "markov": (_state_config(0, output_columns, target_columns), panel["train_frame"], None),
        "markov_plus_item": (item_cfg, item_train, item_test),
    }
    rows, squared_error = {}, {}
    for name, (cfg, train_frame, test_frame) in specs.items():
        model, _, _ = _fit(train_frame, cfg, min_turn=seed_turn)
        cell, se = _score(model, cfg, panel, seed_turn, horizon, test_frame=test_frame)
        cell["state_dim"] = int(model.A.shape[0])
        cell["target_dim"] = int(model.B.shape[1])
        rows[name] = cell
        squared_error[name] = se

    contrasts = {}
    if all(rows[n].get("skill_h") is not None for n in specs):
        for other in ("markov_plus_item", "markov"):
            contrasts[f"ours_minus_{other}"] = ev._contrast(
                squared_error["ours"], squared_error[other], panel["se_null"], panel["groups"], 0)
        contrasts["markov_plus_item_minus_markov"] = ev._contrast(
            squared_error["markov_plus_item"], squared_error["markov"],
            panel["se_null"], panel["groups"], 0)

    return {
        "probe": "m2",
        "task": task,
        "provenance": ev.provenance(),
        "lag": lag,
        "observed_seed_turns": seed_turn,
        "horizon": horizon,
        "fit_note": "all rows refit on transitions whose target turn is past the observed prefix",
        "item_level_columns": item_columns,
        "estimation_turns": [t for t in sorted(set(frame["turn"].astype(int))) if t <= seed_turn],
        "scored_turns": panel["scored_turns"],
        "n_test_trajectories": int(panel["reference"]["trajectory_id"].nunique()),
        "n_scored_rows": int(panel["y_true"].size),
        "best_null_name": panel["null_name"],
        "rows": rows,
        "contrasts": contrasts,
    }


# --------------------------------------------------------------------------- CLI

def run_task(probe: str, task: str) -> dict:
    frame, cfg = ev._load(task)
    output_columns, target_columns = tuple(cfg["output_columns"]), tuple(cfg["target_columns"])
    if probe in ("a1", "a1_matched"):
        matched = probe == "a1_matched"
        return {
            "probe": probe, "task": task,
            "anchors": {
                anchor: run_a1_frame(frame, output_columns, target_columns, task, anchor,
                                     matched_window=matched)
                for anchor in ("primary", "deep")
            },
        }
    if probe == "m1":
        return run_m1_frame(frame, output_columns, target_columns, task,
                            action_channel="constant_reference_per_trajectory")
    return run_m2_frame(frame, output_columns, target_columns, task)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", choices=sorted(OUT_DIRS), required=True)
    parser.add_argument("--tasks", nargs="*", default=ev.TASKS)
    parser.add_argument("--out-dir", type=pathlib.Path, default=None)
    args = parser.parse_args()

    out_dir = args.out_dir or PACKAGE_ROOT / "results" / OUT_DIRS[args.probe]
    out_dir.mkdir(parents=True, exist_ok=True)
    for task in args.tasks:
        out_path = out_dir / f"{task}.json"
        if out_path.exists():
            raise FileExistsError(f"{out_path} exists; outputs are never overwritten")
        result = run_task(args.probe, task)
        out_path.write_text(json.dumps(result, indent=2, default=float) + "\n")
        print(f"{args.probe} {task} -> {out_path}")


if __name__ == "__main__":
    main()
