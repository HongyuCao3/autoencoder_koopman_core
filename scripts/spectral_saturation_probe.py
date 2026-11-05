#!/usr/bin/env python3
"""#1: can the spectrum of K explain "two readings and the state saturates"?

Spec: docs/experiments/spectral_saturation_probe.md. Not an M1 rerun: M1 asked
whether an action impulse lands in one step; this asks whether the modes K
gains when one more reading enters the state are dead (|lambda| < tau) or
copies of modes it already had -- and whether the multi-step score agrees when
K is cut back to its leading modes.

    eigvals          spectrum of K at each depth L, modulus CI from a
                     trajectory-level bootstrap (2000, default_rng(0)).
    new_mode_test    L-1 -> L: Hungarian-match the two spectra; the d modes
                     left unmatched are the new ones (d = readout dimension;
                     user ruling 2026-10-25 for the vector tasks). (a) every
                     new modulus has CI upper < tau; (b) every new mode lies
                     within tau of an eigenvalue K already had at L-1.
    truncated_skill  K_k = Re(V_k Lambda_k V_k^+), top-k modes by modulus, B and
                     the offset untouched, scored on the same held-out panel
                     as A1 matched. At k = full it must reproduce the A1 cell.
    tap_loading      ||K[readout, tap j]||, newest tap first.

Four tasks: three `core` T=10 columns (this package's fit, `paper_side_
experiments._fit`) and `constraint` (the behavioural line's fit, loaded from
`persona_drift_control/scripts/paper_side_experiments_behavioral.py`). Both
use the matched training window, depths L = 0..3, and the deepest state fixes
the scored rows. `constraint`'s K is the full fitted A, including the v_t slot
its state carries (user ruling 2026-10-25).

Fit diagnostics only: eigenvalues and tap loadings are not cross-arm headline
numbers (`.claude/global.md`). Writes `results/spectral_saturation/`, refuses
to overwrite.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

from koopman_ae import AugmentedKoopmanModel, build_augmented_state_dataset

PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[1]
PDC_ROOT = PACKAGE_ROOT / "persona_drift_control"


def _load(module_name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pse = _load("_paper_side_experiments", PACKAGE_ROOT / "scripts" / "paper_side_experiments.py")
ev = pse.ev

TAU = 0.1
N_RESAMPLES = 2000
BOOTSTRAP_SEED = 0
DEPTHS = (0, 1, 2, 3)
CORE_TASKS = ("sentence_length_t10", "vector_count_stage2_t10", "vector_count_stage1_t10")
CONTROL_TASK = "constraint"
TASKS = (*CORE_TASKS, CONTROL_TASK)
OUT_DIR = PACKAGE_ROOT / "results" / "spectral_saturation"


def _behavioral():
    sys.path.insert(0, str(PDC_ROOT / "src"))
    return _load("_paper_side_experiments_behavioral",
                 PDC_ROOT / "scripts" / "paper_side_experiments_behavioral.py")


# --------------------------------------------------------------------------- spectrum

def sorted_eigvals(K: np.ndarray) -> np.ndarray:
    lam = np.linalg.eigvals(K)
    return lam[np.argsort(-np.abs(lam), kind="stable")]


def new_modes(prev: np.ndarray, cur: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Modes of `cur` left over after a minimum-distance matching to `prev`,
    and each one's distance to the nearest eigenvalue `prev` already had."""
    cost = np.abs(cur[:, None] - prev[None, :])
    rows, _ = linear_sum_assignment(cost)
    unmatched = np.setdiff1d(np.arange(len(cur)), rows)
    fresh = cur[unmatched]
    dist = cost[unmatched].min(axis=1) if len(prev) else np.full(len(fresh), np.inf)
    return fresh, dist


def truncate(K: np.ndarray, k: int) -> np.ndarray:
    """K_k = Re(V_k Lambda_k V_k^+), the spec's projection (4.2-3)."""
    lam, V = np.linalg.eig(K)
    order = np.argsort(-np.abs(lam), kind="stable")[:k]
    Vk = V[:, order]
    return np.real(Vk @ np.diag(lam[order]) @ np.linalg.pinv(Vk))


def spectrum_block(K_full: dict, K_draws: dict, d: int) -> dict:
    """Per depth: point spectrum, modulus CIs, and the new-mode test.

    `K_full[L]` is the full-data K; `K_draws[L]` the bootstrap refits, index-
    aligned across L (draw i of every depth comes from the same resample)."""
    out = {}
    for L in sorted(K_full):
        lam = sorted_eigvals(K_full[L])
        mods = np.stack([np.abs(sorted_eigvals(K)) for K in K_draws[L]])
        lo, med, hi = np.quantile(mods, [0.025, 0.5, 0.975], axis=0)
        block = {
            "state_dim": int(K_full[L].shape[0]),
            "eigvals_real": lam.real.tolist(), "eigvals_imag": lam.imag.tolist(),
            "modulus": np.abs(lam).tolist(),
            "modulus_median": med.tolist(), "modulus_ci_low": lo.tolist(), "modulus_ci_high": hi.tolist(),
        }
        if L - 1 in K_full:
            fresh, dist = new_modes(sorted_eigvals(K_full[L - 1]), lam)
            draw_mod, draw_dist = [], []
            for K_prev, K_cur in zip(K_draws[L - 1], K_draws[L]):
                f, dd = new_modes(sorted_eigvals(K_prev), sorted_eigvals(K_cur))
                draw_mod.append(np.abs(f).max())
                draw_dist.append(dd.max())
            mod_hi = float(np.quantile(draw_mod, 0.975))
            dist_hi = float(np.quantile(draw_dist, 0.975))
            a = mod_hi < TAU
            b = bool(np.all(dist < TAU))
            block["new_mode_test"] = {
                "n_new": int(len(fresh)), "expected_n_new": d,
                "new_real": fresh.real.tolist(), "new_imag": fresh.imag.tolist(),
                "new_modulus": np.abs(fresh).tolist(),
                "max_new_modulus_ci_high": mod_hi,
                "distance_to_existing": dist.tolist(),
                "max_distance_ci_high": dist_hi,
                "a_near_zero": bool(a), "b_duplicate": b, "satisfied": bool(a or b),
                "b_holds_under_ci": bool(dist_hi < TAU),
                "rule": f"(a) CI-high of max new |lambda| < {TAU}; (b) every new mode within "
                        f"{TAU} of an L-1 eigenvalue (point estimate, CI-high reported beside it)",
            }
        out[f"L_{L}"] = block
    return out


def saturation(contrasts: dict, k_from: int) -> bool:
    """No increment past `k_from` modes has an interval above zero."""
    later = [c for key, c in contrasts.items() if int(key.split("_")[1]) > k_from]
    return not any(c["ci_low"] > 0 for c in later)


# --------------------------------------------------------------------------- core tasks

def run_core(task: str, *, n_resamples: int) -> dict:
    frame, cfg = ev._load(task)
    return run_core_frame(frame, tuple(cfg["output_columns"]), tuple(cfg["target_columns"]), task,
                          n_resamples=n_resamples)


def run_core_frame(frame, out_cols, tgt_cols, task: str, *, n_resamples: int) -> dict:
    n_turns = int(frame["turn"].max())
    lag_max = max(DEPTHS)
    if ev.choose_lag(n_turns) != lag_max:
        raise ValueError(f"{task}: Table 1 lag is {ev.choose_lag(n_turns)}, spec depths stop at {lag_max}")
    seed_turn = lag_max + 1
    horizon = n_turns - seed_turn
    panel = pse._panel(frame, out_cols, tgt_cols, seed_turn, pse._state_config(lag_max, out_cols, tgt_cols))
    d = len(out_cols)

    K_full, models, fits = {}, {}, {}
    for L in DEPTHS:
        state_cfg = pse._state_config(L, out_cols, tgt_cols)
        model, dataset, (Z, R, Zn) = pse._fit(panel["train_frame"], state_cfg, min_turn=seed_turn)
        keep = dataset.meta["next_turn"].to_numpy(dtype=int) > seed_turn
        tids = dataset.meta["trajectory_id"].astype(str).to_numpy()[keep]
        K_full[L], models[L], fits[L] = np.asarray(model.A), (model, state_cfg), (Z, R, Zn, tids)

    groups = sorted(set(fits[0][3]))
    index_of = {L: {g: np.flatnonzero(fits[L][3] == g) for g in groups} for L in DEPTHS}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    K_draws = {L: [] for L in DEPTHS}
    for _ in range(n_resamples):
        picked = rng.choice(len(groups), size=len(groups), replace=True)
        for L in DEPTHS:
            idx = np.concatenate([index_of[L][groups[i]] for i in picked])
            Z, R, Zn, _ = fits[L]
            m = AugmentedKoopmanModel(output_dim=d, alpha=pse.RIDGE_ALPHA).fit(Z[idx], R[idx], Zn[idx])
            K_draws[L].append(np.asarray(m.A))

    spectra = spectrum_block(K_full, K_draws, d)
    for L in DEPTHS:
        model, state_cfg = models[L]
        dim = K_full[L].shape[0]
        cells, se = {}, {}
        for k in range(1, dim + 1):
            trunc = AugmentedKoopmanModel(output_dim=d, alpha=pse.RIDGE_ALPHA)
            trunc.A, trunc.B, trunc.c = truncate(K_full[L], k), model.B, model.c
            cells[f"k_{k}"], se[k] = pse._score(trunc, state_cfg, panel, seed_turn, horizon)
        _, se_full = pse._score(model, state_cfg, panel, seed_turn, horizon)
        if not np.allclose(se[dim], se_full, rtol=1e-6, atol=1e-10):
            raise AssertionError(f"{task} L={L}: full-rank truncation does not reproduce the fit")
        contrasts = {f"k_{k}_minus_k_{k - 1}": ev._contrast(se[k], se[k - 1], panel["se_null"], panel["groups"], 0)
                     for k in range(2, dim + 1)}
        readout = K_full[L][:d, :]
        taps = [float(np.linalg.norm(readout[:, j * d:(j + 1) * d])) for j in range(L + 1)]
        spectra[f"L_{L}"].update({
            "truncated_skill": cells, "truncated_contrasts": contrasts,
            "saturated_at_k_d": saturation(contrasts, d), "saturated_at_k_2d": saturation(contrasts, 2 * d),
            "tap_loading": taps,
            "non_markov_share": float(np.linalg.norm(taps[1:]) / np.linalg.norm(taps)) if L else 0.0,
        })

    return {
        "probe": "spectral_saturation", "task": task, "provenance": ev.provenance(),
        "python": sys.executable, "readout_dim": d, "tau": TAU, "n_resamples": n_resamples,
        "window": "matched (A1 matched primary anchor): every depth fit on transitions landing past the seed turn",
        "observed_seed_turns": seed_turn, "horizon": horizon,
        "n_train_trajectories": len(groups), "n_test_trajectories": int(panel["reference"]["trajectory_id"].nunique()),
        "best_null_name": panel["null_name"], "state_layout": "newest-first taps of the readout vector",
        "depths": spectra,
    }


# --------------------------------------------------------------------------- constraint

def run_constraint(*, n_resamples: int) -> dict:
    sb = _behavioral()
    from persona_drift.modeling.dataset import ReducedStateConfig
    from persona_drift.modeling.koopman import KoopmanSurrogate, no_extra_features, surrogate_from_arrays

    spec, _, by_traj, n_turns, nu_max, horizon, refused = sb._setup(CONTROL_TASK)
    if nu_max != max(DEPTHS) + 1:
        raise ValueError(f"constraint: Table 1 nu is {nu_max}, spec depths stop at {max(DEPTHS) + 1}")
    y_col, u_col = spec["y_col"], spec["u_col"]
    min_turn = sb._window_start(ReducedStateConfig(nu=nu_max, mu=1, contemporaneous_v=True))
    cfgs = {L: ReducedStateConfig(nu=L + 1, mu=1, contemporaneous_v=True) for L in DEPTHS}

    def fit(trajs, L):
        return KoopmanSurrogate(extra_features_fn=no_extra_features).fit(
            sb._dataset(trajs, cfgs[L], y_col, u_col, min_turn=min_turn))

    K_full = {L: np.asarray(fit(by_traj, L).A) for L in DEPTHS}
    per_traj = {L: {tid: sb._dataset({tid: rows}, cfgs[L], y_col, u_col, min_turn=min_turn)
                    for tid, rows in by_traj.items()} for L in DEPTHS}
    groups = sorted(by_traj)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    K_draws = {L: [] for L in DEPTHS}
    for _ in range(n_resamples):
        picked = rng.choice(len(groups), size=len(groups), replace=True)
        for L in DEPTHS:
            parts = [per_traj[L][groups[i]] for i in picked]
            data = {k: np.concatenate([p[k] for p in parts]) for k in ("Z", "V", "Z_next", "Y")}
            K_draws[L].append(np.asarray(KoopmanSurrogate(extra_features_fn=no_extra_features).fit(data).A))

    spectra = spectrum_block(K_full, K_draws, 1)
    for L in DEPTHS:
        dim = K_full[L].shape[0]

        def make(k, L=L):
            def build(train_trajs):
                m = fit(train_trajs, L)
                A = m.A if k is None else truncate(np.asarray(m.A), k)
                trunc = surrogate_from_arrays(A, m.B, m.b, m.C, state_dim=m.state_dim)
                return lambda tid: trunc
            return cfgs[L], build

        variants = {f"k_{k}": make(k) for k in range(1, dim + 1)}
        variants["full"] = make(None)
        scored = sb._scored(CONTROL_TASK, variants)
        contrast = scored.pop("_contrast")
        se = scored.pop("_se")
        if not np.allclose(se[f"k_{dim}"], se["full"], rtol=1e-6, atol=1e-10):
            raise AssertionError(f"constraint L={L}: full-rank truncation does not reproduce the fit")
        cells = {k: v for k, v in scored["rows"].items() if k.startswith("k_")}
        contrasts = {f"k_{k}_minus_k_{k - 1}": contrast(f"k_{k}", f"k_{k - 1}") for k in range(2, dim + 1)}
        taps = sb._readout_taps(K_full[L], L + 1)
        spectra[f"L_{L}"].update({
            "truncated_skill": cells, "truncated_contrasts": contrasts,
            "saturated_at_k_d": saturation(contrasts, 1), "saturated_at_k_2d": saturation(contrasts, 2),
            "tap_loading": taps,
            "non_markov_share": float(np.linalg.norm(taps[1:]) / np.linalg.norm(taps)) if L else 0.0,
            "scored_rows": scored["n_scored_rows"], "best_null_name": scored["best_null_name"],
        })

    return {
        "probe": "spectral_saturation", "task": CONTROL_TASK, "provenance": sb.beh.provenance(),
        "python": sys.executable, "judge": spec["judge"], "readout_dim": 1, "tau": TAU,
        "n_resamples": n_resamples,
        "window": "matched (A1 matched behavioural): every depth fit on transitions past the nu=4 window start",
        "horizon": horizon, "n_trajectories": len(groups),
        "n_trajectories_refused_unparsed_readout": refused,
        "state_layout": "[y oldest-first (L+1 taps), v_t]; K is the full A including the v_t slot",
        "depths": spectra,
    }


# --------------------------------------------------------------------------- verdict

def verdict(results: dict) -> dict:
    """Spec section 2: the death condition, evaluated on new modes at L=2 and L=3."""
    def holds(task):
        return all(results[task]["depths"][f"L_{L}"]["new_mode_test"]["satisfied"] for L in (2, 3))

    saturating = [t for t in CORE_TASKS if t in results]
    n_hold = sum(holds(t) for t in saturating)
    control_holds = holds(CONTROL_TASK) if CONTROL_TASK in results else None
    complete = len(saturating) == len(CORE_TASKS) and control_holds is not None
    return {
        "per_task_ab_at_L2_and_L3": {t: holds(t) for t in results},
        "n_saturating_tasks_satisfying": n_hold,
        "control_satisfies": control_holds,
        "complete": complete,
        "conjecture_supported": bool(complete and n_hold >= 2 and not control_holds),
        "rule": "supported iff >= 2 of the 3 saturating tasks satisfy (a)/(b) at L=2 and L=3 "
                "AND constraint does not",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="*", default=list(TASKS), choices=TASKS)
    parser.add_argument("--n-resamples", type=int, default=N_RESAMPLES)
    parser.add_argument("--out-dir", type=pathlib.Path, default=OUT_DIR)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    paths = {t: args.out_dir / f"{t}.json" for t in args.tasks}
    paths["summary"] = args.out_dir / "summary.json"
    for p in paths.values():
        if p.exists():
            raise FileExistsError(f"{p} exists; outputs are never overwritten")

    results = {}
    for task in args.tasks:
        results[task] = (run_constraint(n_resamples=args.n_resamples) if task == CONTROL_TASK
                         else run_core(task, n_resamples=args.n_resamples))
        paths[task].write_text(json.dumps(results[task], indent=2, default=float) + "\n")
        for L in DEPTHS[1:]:
            t = results[task]["depths"][f"L_{L}"]["new_mode_test"]
            print(f"{task} L={L} new |lambda|={np.round(t['new_modulus'], 3).tolist()} "
                  f"ci_hi={t['max_new_modulus_ci_high']:.3f} dist={np.round(t['distance_to_existing'], 3).tolist()} "
                  f"a={t['a_near_zero']} b={t['b_duplicate']}", flush=True)
    summary = verdict(results)
    paths["summary"].write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
