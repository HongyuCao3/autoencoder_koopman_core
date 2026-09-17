#!/usr/bin/env python3
"""A1 / M1 / M2 on the three multi-turn behavioural lines.

Spec: ../docs/paper_side_experiments_plan_2026-09-16.md. The core side's
caller is `scripts/paper_side_experiments.py` in the root package; this is the
behavioural side's, and the split is the same one Table 1 already lives with
(the two code bases do not import each other, so each line keeps its native
model classes).

WHY THIS SIDE MATTERS MORE FOR M1. On `core` the control channel is the
reference, constant within a trajectory, so an impulse in it is never
observed and the response read off the fit is a counterfactual. Here the
action is a real per-turn intervention, randomised at collection time
(`constraint` is Bernoulli/antithetic), so the response length is a
measurement -- and it is the quantity the paper's negative result is about.

Additive: imports `scripts/eval_surrogate_rows_behavioral.py` rather than
editing it, so Table 1's behavioural code path is untouched. Writes to three
new directories under `outputs/`.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib

import numpy as np

from persona_drift.modeling.dataset import (
    ReducedStateConfig,
    build_reduced_state_pairs,
    group_by_trajectory,
)
from persona_drift.modeling.koopman import KoopmanSurrogate, no_extra_features
from surrogate_eval import best_null, bootstrap_ci, make_folds, skill_from_squared_errors, trivial_nulls
from surrogate_eval.skill import DegenerateNullError

REPO = pathlib.Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "_eval_surrogate_rows_behavioral", REPO / "scripts" / "eval_surrogate_rows_behavioral.py")
beh = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(beh)

OUT_DIRS = {
    "a1": "ablation_memory_depth_behavioral",
    "a1_matched": "ablation_memory_depth_matched_behavioral",
    "m1": "mechanism_operator_profile_behavioral",
    "m2": "mechanism_item_effect_behavioral",
}
COLLINEAR_R2 = 1.0 - 1e-8
NEAR_COLLINEAR_R2 = 0.99


class LeakageError(AssertionError):
    """The item level was estimated on turns that are also scored."""


class _WithItemLevel:
    """The Markov row, handed the per-item level through the action channel.

    That channel is the one the fit already treats as an exogenous input, and
    a persistent item effect is exactly an exogenous constant. The wrapper is
    rebuilt per trajectory because the level is a property of the item, not of
    the model.
    """

    def __init__(self, inner: KoopmanSurrogate, level: float):
        self.inner, self.level = inner, float(level)

    def step(self, z, v):
        return self.inner.step(z, np.append(np.asarray(v, dtype=float).reshape(-1), self.level))

    def readout(self, z):
        return self.inner.readout(z)


def _window_start(cfg: ReducedStateConfig) -> int:
    return max(cfg.nu - 1, cfg.mu - (1 if cfg.contemporaneous_v else 0))


def _pairs(traj_rows, cfg, y_col, u_col, *, min_turn: int | None = None):
    """{turn of z_t: pair}, optionally dropping transitions that land inside
    the observed prefix (the matched-window variant; see the core side)."""
    by_turn = beh._pairs_by_turn(traj_rows, cfg, y_col, u_col)
    if by_turn is None:
        return None
    if min_turn is not None:
        by_turn = {t: p for t, p in by_turn.items() if t + 1 > min_turn}
    return by_turn


def _dataset(train_trajs, cfg, y_col, u_col, *, min_turn=None, levels=None) -> dict:
    Z, V, Z_next, Y = [], [], [], []
    for tid, traj_rows in train_trajs.items():
        by_turn = _pairs(traj_rows, cfg, y_col, u_col, min_turn=min_turn)
        if not by_turn:
            continue
        for _, pair in sorted(by_turn.items()):
            v = np.asarray(pair["v"], dtype=float).reshape(-1)
            if levels is not None:
                v = np.append(v, float(levels[tid]))
            Z.append(pair["z"])
            V.append(v)
            Z_next.append(pair["z_next"])
            Y.append(pair["y"])
    if not Z:
        raise ValueError("empty identification dataset")
    return {"Z": np.stack(Z), "V": np.stack(V), "Z_next": np.stack(Z_next), "Y": np.asarray(Y, dtype=float)}


def _item_levels(by_traj, y_col: str, prefix_turn: int, scored_min_turn: int) -> dict:
    """Mean readout over the observed prefix, per trajectory.

    Raises rather than returns if the estimation window reaches a scored turn.
    A leak here does not fail loudly; it makes the baseline stronger and the
    paper's one positive claim quietly weaker for the wrong reason.
    """
    if prefix_turn >= scored_min_turn:
        raise LeakageError(
            f"item level would use turns <= {prefix_turn} but scoring starts at {scored_min_turn}")
    levels = {}
    for tid, traj_rows in by_traj.items():
        vals = [float(r[y_col]) for r in traj_rows if int(r["turn"]) <= prefix_turn]
        if not vals:
            raise ValueError(f"trajectory {tid} has no observed prefix")
        levels[tid] = float(np.mean(vals))
    return levels


def _identifiability(Z: np.ndarray, V: np.ndarray) -> dict:
    X = np.column_stack([Z, V, np.ones(len(Z))])
    rank = int(np.linalg.matrix_rank(X))
    singular = np.linalg.svd(X, compute_uv=False)
    ZO = np.column_stack([Z, np.ones(len(Z))])
    coef, *_ = np.linalg.lstsq(ZO, V, rcond=None)
    resid = V - ZO @ coef
    ss_tot = ((V - V.mean(axis=0)) ** 2).sum(axis=0)
    r2 = np.where(ss_tot > 0, 1.0 - (resid ** 2).sum(axis=0) / np.where(ss_tot > 0, ss_tot, 1.0), 1.0)
    identified = bool(rank == X.shape[1] and np.all(r2 < COLLINEAR_R2))
    out = {
        "n_columns": int(X.shape[1]), "rank": rank,
        "rank_deficient": bool(rank < X.shape[1]),
        "v_on_state_r2": [float(v) for v in np.atleast_1d(r2)],
        "identified": identified,
        "near_collinear": bool(np.any(r2 > NEAR_COLLINEAR_R2)),
        "condition_number": float(singular[0] / singular[-1]) if singular[-1] > 0 else None,
    }
    if not identified:
        out["undefined_reason"] = ("rank-deficient design matrix" if rank < X.shape[1]
                                   else "V is an exact affine function of [Z|1]")
    return out


def _readout_taps(A: np.ndarray, nu: int) -> list[float]:
    """How much of the next newest y each tap of the current state carries.

    Read off A, not C. In this code base C is a selection vector -- the y-block
    holds y verbatim, so the fitted C comes out as e_(nu-1) on every line and a
    loading computed from it is [1, 0, ...] by construction, which is what the
    first version of this function reported on all three lines. The memory
    lives in how the state EVOLVES: row nu-1 of A is the map from z_t to the
    newest y of z_(t+1). The y-block is oldest-first, so tap j (newest-first)
    is column nu-1-j.
    """
    row = np.asarray(A)[nu - 1]
    return [float(abs(row[nu - 1 - j])) for j in range(nu)]


def _setup(name: str):
    spec = beh.LINES[name]
    rows = beh.load_rows(spec)
    y_col = spec["y_col"]
    by_traj = group_by_trajectory(rows)
    refused = [t for t, rs in by_traj.items() if any(float(r[y_col]) != float(r[y_col]) for r in rs)]
    if refused:
        by_traj = {t: rs for t, rs in by_traj.items() if t not in set(refused)}
        rows = [r for rs in by_traj.values() for r in rs]
    n_turns = max(len(r) for r in by_traj.values())
    nu, horizon = beh.choose_nu(n_turns)
    return spec, rows, by_traj, n_turns, nu, horizon, len(refused)


# --------------------------------------------------------------------------- scored probes

def _scored(name: str, variants, *, bootstrap_seed: int = 0) -> dict:
    """One fold-averaged panel: every variant scored on identical rows.

    `variants` maps a row name to (state config, a builder that returns a
    per-trajectory predictor). The window comes from the deepest config, so a
    shallower state is seeded at the same absolute turn rather than starting
    earlier -- the same rule Table 1 runs under.
    """
    spec, rows, by_traj, n_turns, nu, horizon, refused = _setup(name)
    y_col, u_col, item_col = spec["y_col"], spec["u_col"], spec["item_col"]
    window_cfg = ReducedStateConfig(nu=nu, mu=1, contemporaneous_v=True)
    traj_items = {tid: rs[0][item_col] for tid, rs in by_traj.items()}
    items = sorted(set(traj_items.values()))
    folds = make_folds(items, beh.N_FOLDS, purpose="report", seed=bootstrap_seed)

    collected: dict[str, list[dict]] = {}
    null_pred: dict[str, list[np.ndarray]] = {}
    meta_rows: list[dict] = []
    for fold in folds:
        test_items = set(fold["test_groups"].tolist())
        train_trajs = {t: rs for t, rs in by_traj.items() if traj_items[t] not in test_items}
        test_trajs = {t: rs for t, rs in by_traj.items() if traj_items[t] in test_items}
        built = {n: build(train_trajs) for n, (cfg, build) in variants.items()}

        fold_meta = None
        for row_name, (cfg, _) in variants.items():
            predictor_for = built[row_name]
            emitted = []
            for tid, traj_rows in test_trajs.items():
                predictor = predictor_for(tid)
                for rec in beh._rollout(predictor, traj_rows, window_cfg, cfg, y_col, u_col, horizon):
                    emitted.append(dict(rec, trajectory_id=tid, item=traj_items[tid]))
            collected.setdefault(row_name, []).extend(emitted)
            if fold_meta is None:
                fold_meta = emitted
            else:
                assert [(r["trajectory_id"], r["turn"], r["step"]) for r in emitted] == \
                       [(r["trajectory_id"], r["turn"], r["step"]) for r in fold_meta], \
                       f"{row_name} scored on different rows"
        meta_rows.extend(fold_meta)

        train_panel = {"y_next": [], "turn_next": [], "v": []}
        for traj_rows in train_trajs.values():
            by_turn = _pairs(traj_rows, window_cfg, y_col, u_col)
            if not by_turn:
                continue
            for turn, pair in sorted(by_turn.items()):
                train_panel["y_next"].append(float(pair["z_next"][window_cfg.nu - 1]))
                train_panel["v"].append(float(np.asarray(pair["v"]).reshape(-1)[0]))
                train_panel["turn_next"].append(float(turn + 1))
        train_panel = {k: np.asarray(v, dtype=float) for k, v in train_panel.items()}
        test_panel = {
            "y_next": np.asarray([r["y_true"] for r in fold_meta], dtype=float),
            "turn_next": np.asarray([r["turn"] for r in fold_meta], dtype=float),
            "v": np.asarray([r["v"] for r in fold_meta], dtype=float),
        }
        for k, v in trivial_nulls(train_panel, test_panel, exogenous=("turn_next", "v")).items():
            null_pred.setdefault(k, []).append(v)

    y_true = np.asarray([r["y_true"] for r in meta_rows], dtype=float)
    groups = np.asarray([r["item"] for r in meta_rows])
    null_name, y_null = best_null({k: np.concatenate(v) for k, v in null_pred.items()}, y_true)
    se_null = (y_null - y_true) ** 2
    se = {k: (np.asarray([r["y_pred"] for r in v], dtype=float) - y_true) ** 2 for k, v in collected.items()}

    def cell(s):
        out = {"rollout_mse": float(np.mean(s)), "null_mse": float(np.mean(se_null)), "n_rows": int(s.size)}
        try:
            out["skill_h"] = skill_from_squared_errors(s, se_null, horizon=horizon)
        except DegenerateNullError as exc:
            out["skill_h"] = None
            out["degenerate"] = str(exc)
            return out
        out["bootstrap"] = bootstrap_ci(
            np.column_stack([s, se_null]), groups, seed=bootstrap_seed,
            statistic=lambda v: 1.0 - float(np.mean(v[:, 0])) / float(np.mean(v[:, 1])))
        return out

    table = {k: cell(v) for k, v in se.items()}
    table["best_null"] = {"which": null_name, "rollout_mse": float(np.mean(se_null)),
                          "skill_h": 0.0, "n_rows": int(y_true.size)}

    def contrast(a, b):
        return bootstrap_ci(
            np.column_stack([se[a], se[b], se_null]), groups, seed=bootstrap_seed,
            statistic=lambda v: (float(np.mean(v[:, 1])) - float(np.mean(v[:, 0]))) / float(np.mean(v[:, 2])))

    return {
        "line": name, "provenance": beh.provenance(), "judge": spec["judge"],
        "n_turns": n_turns, "nu": nu, "horizon": horizon,
        "n_items": len(items), "n_trajectories": len(by_traj),
        "n_scored_rows": int(y_true.size), "best_null_name": null_name,
        "n_trajectories_refused_unparsed_readout": refused,
        "scored_turn_min": int(min(r["turn"] for r in meta_rows)),
        "rows": table, "_contrast": contrast, "_se": se,
    }


def run_a1(name: str, *, matched_window: bool = False) -> dict:
    spec, rows, by_traj, n_turns, nu_max, horizon, _ = _setup(name)
    y_col, u_col = spec["y_col"], spec["u_col"]
    window_cfg = ReducedStateConfig(nu=nu_max, mu=1, contemporaneous_v=True)
    min_turn = _window_start(window_cfg) if matched_window else None

    def make(nu):
        cfg = ReducedStateConfig(nu=nu, mu=1, contemporaneous_v=True)

        def build(train_trajs):
            model = KoopmanSurrogate(extra_features_fn=no_extra_features).fit(
                _dataset(train_trajs, cfg, y_col, u_col, min_turn=min_turn))
            return lambda tid: model
        return cfg, build

    variants = {f"nu_{nu}": make(nu) for nu in range(1, nu_max + 1)}
    out = _scored(name, variants)
    contrast, se = out.pop("_contrast"), out.pop("_se")
    contrasts = {}
    for nu in range(2, nu_max + 1):
        if out["rows"]["nu_1"].get("skill_h") is None or out["rows"][f"nu_{nu}"].get("skill_h") is None:
            continue
        contrasts[f"nu_{nu}_minus_nu_1"] = contrast(f"nu_{nu}", "nu_1")
        contrasts[f"nu_{nu}_minus_nu_{nu - 1}"] = contrast(f"nu_{nu}", f"nu_{nu - 1}")
    out.update({"probe": "a1", "matched_window": bool(matched_window),
                "nu_max": nu_max, "contrasts": contrasts})
    return out


def run_m2(name: str) -> dict:
    spec, rows, by_traj, n_turns, nu_max, horizon, _ = _setup(name)
    y_col, u_col = spec["y_col"], spec["u_col"]
    window_cfg = ReducedStateConfig(nu=nu_max, mu=1, contemporaneous_v=True)
    markov_cfg = ReducedStateConfig(nu=1, mu=1, contemporaneous_v=True)
    prefix_turn = int(min(int(r["turn"]) for rs in by_traj.values() for r in rs)) + _window_start(window_cfg)
    scored_min_turn = prefix_turn + 1
    levels = _item_levels(by_traj, y_col, prefix_turn, scored_min_turn)
    # Matched training window for every row, so the only thing that differs is
    # the state: a shallow state is usable earlier, and letting it train on
    # transitions the deep state never sees turns a depth comparison into a
    # training-window comparison (core side, `sentence_length_t10`: Markov
    # skill +0.166 unmatched vs +0.660 matched).
    min_turn = _window_start(window_cfg)

    def plain(cfg):
        def build(train_trajs):
            model = KoopmanSurrogate(extra_features_fn=no_extra_features).fit(
                _dataset(train_trajs, cfg, y_col, u_col, min_turn=min_turn))
            return lambda tid: model
        return cfg, build

    def with_item(cfg):
        def build(train_trajs):
            model = KoopmanSurrogate(extra_features_fn=no_extra_features).fit(
                _dataset(train_trajs, cfg, y_col, u_col, min_turn=min_turn, levels=levels))
            return lambda tid: _WithItemLevel(model, levels[tid])
        return cfg, build

    variants = {"ours": plain(window_cfg), "markov": plain(markov_cfg),
                "markov_plus_item": with_item(markov_cfg)}
    out = _scored(name, variants)
    contrast, se = out.pop("_contrast"), out.pop("_se")
    if out["scored_turn_min"] <= prefix_turn:
        raise LeakageError(
            f"scoring starts at turn {out['scored_turn_min']} but the item level used turns <= {prefix_turn}")
    contrasts = {}
    if all(out["rows"][n].get("skill_h") is not None for n in variants):
        contrasts["ours_minus_markov_plus_item"] = contrast("ours", "markov_plus_item")
        contrasts["ours_minus_markov"] = contrast("ours", "markov")
        contrasts["markov_plus_item_minus_markov"] = contrast("markov_plus_item", "markov")
    out.update({"probe": "m2", "item_level_prefix_turn": prefix_turn,
                "fit_note": "all rows fit on transitions landing past the observed prefix",
                "contrasts": contrasts})
    return out


# --------------------------------------------------------------------------- M1

def run_m1(name: str) -> dict:
    """Spectrum, lag loading and action response, on one fit over all rows.

    A descriptive read of the operator, not a scored comparison, so it does
    not use the report folds; `.claude/global.md` does not let any of these
    numbers stand as a cross-arm headline either way. What it buys is the half
    of the mechanism `core` cannot give: here the action is randomised per
    turn, so the response length is measured rather than extrapolated.
    """
    spec, rows, by_traj, n_turns, nu, horizon, refused = _setup(name)
    y_col, u_col = spec["y_col"], spec["u_col"]
    cfg = ReducedStateConfig(nu=nu, mu=1, contemporaneous_v=True)
    data = _dataset(by_traj, cfg, y_col, u_col)
    model = KoopmanSurrogate(extra_features_fn=no_extra_features).fit(data)

    A, B, C = np.asarray(model.A), np.asarray(model.B), np.asarray(model.C)
    modulus = np.sort(np.abs(np.linalg.eigvals(A)))[::-1]
    rho = float(modulus[0])
    half_life = float(np.log(0.5) / np.log(rho)) if 0.0 < rho < 1.0 else None
    taps = _readout_taps(A, nu)
    total = float(np.sqrt(sum(v ** 2 for v in taps)))
    older = float(np.sqrt(sum(v ** 2 for v in taps[1:])))

    ident = _identifiability(data["Z"], data["V"])
    out = {
        "probe": "m1", "line": name, "provenance": beh.provenance(), "judge": spec["judge"],
        "nu": nu, "horizon": horizon, "n_transitions": int(len(data["Z"])),
        "spectrum": {"eigenvalue_modulus": [float(v) for v in modulus],
                     "spectral_radius": rho, "half_life_steps": half_life},
        "readout_loading": {"tap_abs": taps,
                            "non_markov_share": (older / total) if total > 0 else None,
                            "rule": "|C[nu-1-j]| for taps j=0..nu-1, newest first"},
        "action_channel": "randomized_per_turn",
        "identifiability": ident,
    }
    if not ident["identified"]:
        out["response"] = {"identified": False, "undefined_reason": ident["undefined_reason"],
                           "impulse_response": None, "step_response": None, "response_length": None}
        return out

    impulse, step, power, cumulative = [], [], np.eye(A.shape[0]), np.zeros_like(B)
    for _ in range(horizon):
        impulse.append(float(np.linalg.norm(C @ (power @ B))))
        cumulative = cumulative + power @ B
        step.append(float(np.linalg.norm(C @ cumulative)))
        power = A @ power
    peak = max(impulse) if impulse else 0.0
    out["response"] = {
        "identified": True,
        "impulse_response": impulse,
        "step_response": step,
        "response_length": int(sum(1 for v in impulse if peak > 0 and v >= 0.1 * peak)),
        "response_length_rule": "steps with ||C A^(k-1) B|| >= 0.1 * peak",
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", choices=sorted(OUT_DIRS), required=True)
    parser.add_argument("--lines", nargs="*", default=list(beh.LINES))
    parser.add_argument("--out-dir", type=pathlib.Path, default=None)
    args = parser.parse_args()

    out_dir = args.out_dir or REPO / "outputs" / OUT_DIRS[args.probe]
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in args.lines:
        out_path = out_dir / f"{name}.json"
        if out_path.exists():
            raise FileExistsError(f"{out_path} exists; outputs are never overwritten")
        if args.probe == "m1":
            result = run_m1(name)
        elif args.probe == "m2":
            result = run_m2(name)
        else:
            result = run_a1(name, matched_window=args.probe == "a1_matched")
        out_path.write_text(json.dumps(result, indent=2, default=float) + "\n")
        print(f"[done] {args.probe} {name} -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
