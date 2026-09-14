#!/usr/bin/env python3
"""tsar_cefr Phase 3 identification (plan
docs/operational_plan_tsar_cefr_line_2026-09-13.md section 5.1). CPU only,
pure numpy -- run directly, no sbatch, no model loaded.

STATE. xi_t is a lag-embedded (ell, s) pair -- ell = level_expected (the
continuous CEFR readout, plan section 3.1), s = meaning_to_source. The plan
names TWO viable lags for this line and defers the pick ("lag=2, H=4 or
lag=1, H=4, annotate per column", section 5.1): this script chooses **lag=1**.
Reason: lag=1 keeps one more one-step transition per trajectory than lag=2
(5 of T=6 steps vs 4), which buys more identification data at this line's
size (100 source_ids after exclusion) than the deeper history buys in return.
lag=2 is the other plan-named option and is NOT implemented here -- a real
choice this script is making on the plan's behalf, not a value the plan
pins down. H=4 (the MPC horizon, plan section 5.4) is fixed either way and
does not depend on this pick.

    xi_t = [ell_t, s_t, ell_(t-1), s_(t-1)]      (newest first, state_dim=4)

Step 1's "t-1" is the source paragraph: ell_0 = source_level_expected (a real
field on every row), s_0 = 1.0 -- a judgment call this script makes because
the row schema carries no meaning_to_source value for the source against
itself; a text's similarity to itself is 1.0 by the same metric used
everywhere else in this line.

ACTION. u_t is the one-hot of the action on the row that PRODUCES xi_(t+1)
(the row with step == t+1), over {step_down, half_step_down, paraphrase};
`copy` is the reference class (all-zero), 3 degrees of freedom, exactly as
plan section 5.1 specifies.

MODEL. xi_(t+1) = K xi_t + B u_t + c, fit as ridge least squares via
persona_drift.modeling.koopman.KoopmanSurrogate with
extra_features_fn=no_extra_features (the ARX/identity lift -- eta_t IS xi_t,
nothing lifted), so K/B/c come from the exact fitting code every other
line's S2 gate already uses (fit_koopman_sequor_model.py,
fit_koopman_defense_model.py) rather than a retyped ridge solve.

xi's trailing two slots (ell_(t-1), s_(t-1)) are a DETERMINISTIC COPY of
xi_t's own leading pair once rolled forward one step -- they carry no new
information. Scoring the full 4-vector would hand the fitted model a free,
uninteresting win on that copied block that no null model could ever match
(none of the three nulls below are handed xi_t at all), inflating G-S2-1 for
a reason that has nothing to do with whether the dynamics are real. So both
G-S2-1's one-step MSE and G-S2-2's B column read off ONLY dims [0, 1] of the
4-vector -- [ell_(t+1), s_(t+1)], the genuinely new part of the state. This
mirrors fit_koopman_sequor_model.py's fold evaluation, which extracts a
single `config.nu - 1` index out of a lagged Z_next rather than scoring the
whole lagged vector.

GATES (plan section 5.1; "S2 same form" as the `constraint` line's
fit_koopman_sequor_model.py and the `defense` line's
fit_koopman_defense_model.py):

    G-S2-1  one-step held-out MSE on [ell_(t+1), s_(t+1)] (summed across the
            two dims per sample, then averaged -- modeling.evaluate's
            `np.sum(...)**2` convention, not a per-dim average) beats the
            best of three trivial nulls (`const` / `turn_mean` / `stateless`
            = OLS on [1, turn_next, target_level_rank]) on >= 14 of 20
            source_id-disjoint folds.
    G-S2-2  B's `step_down` column, source_id-clustered bootstrap (fixed
            seed), 95% CI excludes 0. The plan does not say which of the two
            output dims this applies to; this script's choice is
            `ell_(t+1)` (the primary controlled state -- the same choice
            Phase 2's D-2 check and every other line's K3/D-2 gate made for
            the analogous question), with `s_(t+1)` reported alongside for
            transparency but NOT gated. The plan also states no sign
            requirement here (unlike D-2's "must be RESOLVED in the working
            direction"), so this gate is read literally: CI-excludes-zero,
            no sign check.
    G-S2-3  spectral radius / controllability rank / Gramian condition of
            the operator fit on ALL retained transitions, horizon H=4.
            RECORDED ONLY, no threshold -- reuses
            persona_drift.modeling.koopman.controllability_diagnostics
            rather than a retyped eigendecomposition.

EXCLUSION. text_id "51-a2" is dropped before anything else runs -- the
2026-09-14 named-item ruling already applied by
scripts/analyze_tsar_cefr_gpu1.py for G-S1 (design resolution 1/18 sits
above the 5% cap-pressure line). It is a fixed precondition here, not a
re-derived gate: this script does not re-run cap-pressure or re-justify the
drop, it only applies it.

BOUNDARY (explicit, matching the task this script was written for): only
identification + G-S2. No D-3 degeneracy check (plan section 5.2), no
simulation gate G-T3 (plan section 5.3), no docs/ edits, no GPU submission.
"""

from __future__ import annotations

import itertools
import json
import pathlib
import sys
from collections import defaultdict

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
PDC_ROOT = REPO_ROOT / "persona_drift_control"
sys.path.insert(0, str(PDC_ROOT / "src"))

from persona_drift.run_provenance import provenance                                # noqa: E402
from persona_drift.modeling.koopman import (                                       # noqa: E402
    KoopmanSurrogate,
    controllability_diagnostics,
    no_extra_features,
)

DEFAULT_ROWS_PATH = PDC_ROOT / "outputs" / "tsar_cefr_gpu1" / "trajectories.jsonl"
DEFAULT_OUT_PATH = PDC_ROOT / "outputs" / "tsar_cefr_phase3" / "operator_fit.json"

# 2026-09-14 named exclusion, already applied by analyze_tsar_cefr_gpu1.py for G-S1.
EXCLUDED_TEXT_IDS = ("51-a2",)

# --- module-level gate constants (not CLI args -- analyze_tsar_cefr_gpu1.py's rule) ---
LAG = 1                          # judgment call, see module docstring; alternative = 2
HORIZON_H = 4                    # plan section 5.4, fixed regardless of LAG
N_FOLDS = 20                     # plan section 5.1
MIN_FOLDS_PASS = 14              # plan section 5.1
RIDGE = 1e-6                     # matches every other fit_koopman_*.py in this repo
ACTIONS = ("step_down", "half_step_down", "paraphrase")   # copy = reference (u=0)
REFERENCE_ACTION = "copy"
CEFR_RANK = {"A1": 1, "A2": 2, "B1": 3, "B2": 4, "C1": 5, "C2": 6}   # plan section 3.1
NEW_BLOCK = (0, 1)                # xi's leading [ell_(t+1), s_(t+1)] slots
FOLD_SPLIT_SEED = 20260913        # fixed so the fold assignment reproduces
BOOTSTRAP_DRAWS = 10000           # matches analyze_tsar_cefr_gpu1.py on this line
BOOTSTRAP_SEED = 20260914         # matches analyze_tsar_cefr_gpu1.py on this line


def read_jsonl(path: pathlib.Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def action_one_hot(action: str) -> np.ndarray:
    return np.array([1.0 if action == a else 0.0 for a in ACTIONS])


def build_transitions(rows: list[dict], lag: int = LAG) -> list[dict]:
    """One entry per one-step transition xi_t -> xi_(t+1), tagged with the
    metadata the fold split (source_id), the null models (turn_next,
    target_cefr) and the bootstrap clustering (source_id) below all need.

    `excluded` is applied here and nowhere else in this script -- everything
    downstream sees only retained rows.
    """
    excluded = set(EXCLUDED_TEXT_IDS)
    by_traj: dict[tuple, dict[int, dict]] = defaultdict(dict)
    meta: dict[tuple, dict] = {}
    for row in rows:
        if row["text_id"] in excluded:
            continue
        key = (row["text_id"], row["seed"])
        by_traj[key][row["step"]] = row
        meta[key] = row  # source_id / target_cefr / source_level_expected are constant per key

    transitions = []
    for key, steps in by_traj.items():
        text_id, seed = key
        m = meta[key]
        ell = {0: m["source_level_expected"]}
        s = {0: 1.0}  # judgment call: a text's similarity to itself
        for t, row in steps.items():
            ell[t] = row["level_expected"]
            s[t] = row["meaning_to_source"]
        max_t = max(steps)
        for t in range(lag, max_t):
            if any((t - k) not in ell for k in range(lag + 1)):
                continue
            if (t + 1) not in steps:
                continue
            xi_t = np.array(list(itertools.chain.from_iterable(
                (ell[t - k], s[t - k]) for k in range(lag + 1))))
            xi_next = np.array(list(itertools.chain.from_iterable(
                (ell[t + 1 - k], s[t + 1 - k]) for k in range(lag + 1))))
            action = steps[t + 1]["action"]
            transitions.append({
                "text_id": text_id, "source_id": m["source_id"], "seed": seed,
                "target_cefr": m["target_cefr"], "t": t, "turn_next": t + 1,
                "xi_t": xi_t, "xi_next": xi_next, "u": action_one_hot(action),
                "action": action,
            })
    return transitions


def to_dataset(transitions: list[dict]) -> dict:
    Z = np.stack([tr["xi_t"] for tr in transitions])
    V = np.stack([tr["u"] for tr in transitions])
    Z_next = np.stack([tr["xi_next"] for tr in transitions])
    return {"Z": Z, "V": V, "Z_next": Z_next, "Y": Z[:, 0]}  # Y unused (no readout is gated)


def fit_operator(transitions: list[dict], ridge: float = RIDGE) -> KoopmanSurrogate:
    model = KoopmanSurrogate(extra_features_fn=no_extra_features, ridge=ridge)
    model.fit(to_dataset(transitions))
    return model


def combined_mse(pred: np.ndarray, true: np.ndarray) -> float:
    """Sum of squared error across the new-block dims per sample, then
    averaged over samples -- modeling.evaluate.one_step_error's convention
    (`np.sum(...)**2`), not a per-dim mean."""
    return float(np.mean(np.sum((pred - true) ** 2, axis=1)))


def per_dim_mse(pred: np.ndarray, true: np.ndarray) -> list[float]:
    return [float(v) for v in np.mean((pred - true) ** 2, axis=0)]


def null_predictions(train: list[dict], test: list[dict]) -> dict[str, np.ndarray]:
    """The three trivial predictors of [ell_(t+1), s_(t+1)], fit on `train`,
    applied to `test`. None of the three is handed xi_t -- only `turn_next`
    (deterministic and exogenous) and `target_level_rank` (the trajectory's
    known goal), exactly the plan's `1 + turn + 目标级` for `stateless`."""
    train_target = np.stack([tr["xi_next"][list(NEW_BLOCK)] for tr in train])
    train_turn = np.array([tr["turn_next"] for tr in train])
    test_turn = np.array([tr["turn_next"] for tr in test])

    const = train_target.mean(axis=0)
    const_pred = np.tile(const, (len(test), 1))

    turn_means = {turn: train_target[train_turn == turn].mean(axis=0)
                  for turn in np.unique(train_turn)}
    turn_mean_pred = np.stack([turn_means.get(turn, const) for turn in test_turn])

    train_rank = np.array([CEFR_RANK[tr["target_cefr"].upper()] for tr in train])
    test_rank = np.array([CEFR_RANK[tr["target_cefr"].upper()] for tr in test])
    X_train = np.column_stack([np.ones(len(train)), train_turn, train_rank])
    X_test = np.column_stack([np.ones(len(test)), test_turn, test_rank])
    beta, *_ = np.linalg.lstsq(X_train, train_target, rcond=None)
    stateless_pred = X_test @ beta

    return {"const": const_pred, "turn_mean": turn_mean_pred, "stateless": stateless_pred}


def run_cv_folds(transitions: list[dict], n_folds: int = N_FOLDS,
                  min_pass: int = MIN_FOLDS_PASS, ridge: float = RIDGE,
                  seed: int = FOLD_SPLIT_SEED) -> dict:
    """`n_folds` source_id-disjoint folds. Splitting by source_id (not by
    row, not by text_id) is the anti-leak rule every S2 gate on this line
    and the `constraint`/`defense` lines already use -- steps within one
    trajectory are dependent, and a source's two target cells share the same
    paragraph."""
    source_ids = sorted({tr["source_id"] for tr in transitions})
    rng = np.random.default_rng(seed)
    shuffled = list(rng.permutation(source_ids))
    groups = [shuffled[i::n_folds] for i in range(n_folds)]

    folds = []
    for held_out in groups:
        if len(held_out) == 0:
            continue
        held = {str(s) for s in held_out}
        train = [tr for tr in transitions if tr["source_id"] not in held]
        test = [tr for tr in transitions if tr["source_id"] in held]
        if not train or not test:
            continue

        model = fit_operator(train, ridge=ridge)
        pred = np.stack([model.step(tr["xi_t"], tr["u"])[list(NEW_BLOCK)] for tr in test])
        true = np.stack([tr["xi_next"][list(NEW_BLOCK)] for tr in test])
        model_mse = combined_mse(pred, true)

        nulls = null_predictions(train, test)
        null_mse = {name: combined_mse(p, true) for name, p in nulls.items()}
        best_null_mse = min(null_mse.values())

        folds.append({
            "held_out_source_ids": sorted(held),
            "n_train_transitions": len(train), "n_test_transitions": len(test),
            "model_mse": model_mse, "model_mse_per_dim": per_dim_mse(pred, true),
            "null_mse": null_mse, "best_null_mse": best_null_mse,
            "beats_best_null": bool(model_mse < best_null_mse),
        })

    n_folds_run = len(folds)
    n_passed = sum(1 for f in folds if f["beats_best_null"])
    return {
        "n_folds_requested": n_folds, "n_folds_run": n_folds_run,
        "min_folds_pass": min_pass,
        "n_folds_beating_best_null": n_passed,
        "mean_model_mse": float(np.mean([f["model_mse"] for f in folds])) if folds else None,
        "mean_best_null_mse": float(np.mean([f["best_null_mse"] for f in folds])) if folds else None,
        "folds": folds,
    }


def gate_s2_1(cv: dict) -> dict:
    verdict = (cv["n_folds_run"] == cv["n_folds_requested"]
               and cv["n_folds_beating_best_null"] >= cv["min_folds_pass"])
    return {
        "criterion": f"one-step MSE on [ell_(t+1), s_(t+1)] beats the best of "
                     f"const/turn_mean/stateless on >= {cv['min_folds_pass']} of "
                     f"{cv['n_folds_requested']} source_id-disjoint folds",
        "n_folds_run": cv["n_folds_run"], "n_folds_requested": cv["n_folds_requested"],
        "n_folds_beating_best_null": cv["n_folds_beating_best_null"],
        "min_folds_pass": cv["min_folds_pass"],
        "mean_model_mse": cv["mean_model_mse"], "mean_best_null_mse": cv["mean_best_null_mse"],
        "verdict": "PASS" if verdict else "FAIL",
    }


def bootstrap_b_step_down(transitions: list[dict], ridge: float = RIDGE,
                           draws: int = BOOTSTRAP_DRAWS, seed: int = BOOTSTRAP_SEED) -> dict:
    """Cluster bootstrap over source_id (fixed seed): resample sources with
    replacement, refit via the same normal equations KoopmanSurrogate.fit
    solves, read off B's step_down row for both new-block output dims.
    Per-source Gram/moment blocks are precomputed once, so each of the
    `draws` resamples is a sum plus one linear solve, not a full refit."""
    d_v = len(ACTIONS)
    per_source: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    sources = sorted({tr["source_id"] for tr in transitions})
    for sid in sources:
        rows = [tr for tr in transitions if tr["source_id"] == sid]
        Z = np.stack([tr["xi_t"] for tr in rows])
        V = np.stack([tr["u"] for tr in rows])
        Zn = np.stack([tr["xi_next"] for tr in rows])
        X = np.hstack([Z, V, np.ones((len(rows), 1))])
        per_source[sid] = (X.T @ X, X.T @ Zn)

    d_psi = transitions[0]["xi_t"].shape[0]
    ridge_eye = ridge * np.eye(d_psi + d_v + 1)
    step_down_idx = d_psi + ACTIONS.index("step_down")

    def theta_for(chosen: list[str]) -> np.ndarray:
        gram = sum(per_source[s][0] for s in chosen)
        moment = sum(per_source[s][1] for s in chosen)
        return np.linalg.solve(gram + ridge_eye, moment)

    point_theta = theta_for(sources)
    point_ell = float(point_theta[step_down_idx, NEW_BLOCK[0]])
    point_s = float(point_theta[step_down_idx, NEW_BLOCK[1]])

    rng = np.random.default_rng(seed)
    draws_ell, draws_s = [], []
    for _ in range(draws):
        chosen = [sources[i] for i in rng.integers(0, len(sources), size=len(sources))]
        theta = theta_for(chosen)
        draws_ell.append(float(theta[step_down_idx, NEW_BLOCK[0]]))
        draws_s.append(float(theta[step_down_idx, NEW_BLOCK[1]]))

    def summarize(point: float, arr: list[float]) -> dict:
        lo, hi = np.percentile(arr, [2.5, 97.5])
        return {"point": point, "ci95": [float(lo), float(hi)],
                "bootstrap_sd": float(np.std(arr, ddof=1))}

    return {
        "n_sources": len(sources), "n_transitions": len(transitions),
        "draws": draws, "seed": seed, "bootstrap_over": "source_id",
        "ell_next": summarize(point_ell, draws_ell),
        "s_next": summarize(point_s, draws_s),
    }


def gate_s2_2(boot: dict) -> dict:
    ell = boot["ell_next"]
    excludes_zero = bool(ell["ci95"][0] * ell["ci95"][1] > 0)
    return {
        "criterion": "B's step_down column CI excludes 0 (source_id-clustered bootstrap); "
                     "gated on the ell_(t+1) dim (this script's choice, see module "
                     "docstring); s_(t+1) reported alongside, not gated; no sign "
                     "requirement (unlike Phase 2's D-2)",
        "gated_dim": "ell_next", "point": ell["point"], "ci95": ell["ci95"],
        "bootstrap_sd": ell["bootstrap_sd"], "ci_excludes_zero": excludes_zero,
        "s_next_reported_not_gated": boot["s_next"],
        "verdict": "PASS" if excludes_zero else "FAIL",
    }


def gate_s2_3(model: KoopmanSurrogate, horizon: int = HORIZON_H) -> dict:
    diag = controllability_diagnostics(model.A, model.B, horizon)
    return {
        "criterion": "recorded only, no threshold (plan section 5.1)",
        "horizon": horizon, "state_dim": model.state_dim,
        "spectral_radius": diag["spectral_radius"],
        "controllability_rank": diag["controllability_rank"],
        "gramian_condition": diag["gramian_condition"],
        "full_rank": diag["controllability_rank"] == model.state_dim,
    }


def _assert_bootstrap_matches_fit(model: KoopmanSurrogate, boot: dict) -> None:
    """The bootstrap's own normal-equations solve at the full sample must
    reproduce the model KoopmanSurrogate.fit produced -- if they disagree,
    the CI describes a model nobody fit (the same guard
    fit_koopman_sequor_model.py's `_assert_matches_surrogate` applies)."""
    idx = ACTIONS.index("step_down")
    fitted_ell = float(model.B[NEW_BLOCK[0], idx])
    fitted_s = float(model.B[NEW_BLOCK[1], idx])
    if not (np.isclose(fitted_ell, boot["ell_next"]["point"], rtol=1e-6, atol=1e-9)
            and np.isclose(fitted_s, boot["s_next"]["point"], rtol=1e-6, atol=1e-9)):
        raise SystemExit(
            f"bootstrap point estimate disagrees with KoopmanSurrogate.fit "
            f"(ell: {boot['ell_next']['point']:.8f} vs {fitted_ell:.8f}; "
            f"s: {boot['s_next']['point']:.8f} vs {fitted_s:.8f}): "
            f"the CI would describe a model nobody fit")


def build_report(rows: list[dict]) -> dict:
    n_rows_total = len(rows)
    n_rows_retained = sum(1 for r in rows if r["text_id"] not in EXCLUDED_TEXT_IDS)
    transitions = build_transitions(rows)
    if not transitions:
        raise SystemExit("no transitions built -- check the rows path and the exclusion list")

    model = fit_operator(transitions)
    cv = run_cv_folds(transitions)
    boot = bootstrap_b_step_down(transitions)
    _assert_bootstrap_matches_fit(model, boot)

    xi_layout = list(itertools.chain.from_iterable(
        ((f"ell_(t-{k})" if k else "ell_t"), (f"s_(t-{k})" if k else "s_t"))
        for k in range(LAG + 1)))

    return {
        "gate": "G-S2 identification "
                "(docs/operational_plan_tsar_cefr_line_2026-09-13.md section 5.1)",
        "line": "tsar_cefr", "phase": "phase3_identification",
        "input": {
            "rows_path": str(DEFAULT_ROWS_PATH),
            "n_rows_total": n_rows_total,
            "excluded_text_ids": list(EXCLUDED_TEXT_IDS),
            "n_rows_retained": n_rows_retained,
            "n_text_ids_retained": len({tr["text_id"] for tr in transitions}),
            "n_source_ids_retained": len({tr["source_id"] for tr in transitions}),
            "n_transitions": len(transitions),
        },
        "state_definition": {
            "lag": LAG, "lag_alternative_named_by_plan_not_chosen": 2,
            "horizon_H": HORIZON_H, "state_dim": model.state_dim,
            "xi_layout": xi_layout,
            "s0_convention": "meaning_to_source of the source paragraph against itself = 1.0 "
                             "(not a row field; this script's judgment call)",
        },
        "action_encoding": {"free_actions": list(ACTIONS), "reference_action": REFERENCE_ACTION},
        "final_operator": {
            "K": model.A.tolist(), "B": model.B.tolist(), "c": model.b.tolist(),
            "state_dim": model.state_dim, "u_layout": list(ACTIONS), "ridge": RIDGE,
            "n_transitions_used": len(transitions),
            "fitted_on": "all retained transitions (this script does not hold out a "
                         "separate report split; the 5-fold identification/report split "
                         "in plan section Phase 1 is for the future closed-loop phase)",
        },
        "g_s2_1": gate_s2_1(cv),
        "g_s2_1_fold_detail": cv,
        "g_s2_2": gate_s2_2(boot),
        "g_s2_2_bootstrap_detail": boot,
        "g_s2_3": gate_s2_3(model),
        "choices_made_by_this_script": [
            "lag=1 chosen over the plan's other named option lag=2 (both keep H=4); "
            "lag=1 keeps 5 of T=6 transitions per trajectory vs 4 for lag=2",
            "s_0 (source paragraph's own meaning_to_source) set to 1.0; not a row field",
            "one-step MSE scores only the new [ell_(t+1), s_(t+1)] block, summed not "
            "averaged across the 2 dims, to avoid crediting the model for a trivially "
            "copied lag block no null model has access to",
            "G-S2-2 is gated on the ell_(t+1) dim of B's step_down column; s_(t+1) is "
            "reported but not gated -- the plan names neither dim",
            "G-S2-2 reads the plan literally: CI-excludes-zero only, no sign requirement "
            "(unlike Phase 2's D-2, which required the working direction)",
        ],
        "boundary": "identification + G-S2 only; no D-3 (plan section 5.2), no G-T3 "
                    "simulation (plan section 5.3), no docs/ edits, no GPU submission, "
                    "per the task this script was written for",
        "provenance": provenance({"lag": LAG, "horizon_H": HORIZON_H, "ridge": RIDGE,
                                  "n_folds": N_FOLDS, "min_folds_pass": MIN_FOLDS_PASS,
                                  "bootstrap_draws": BOOTSTRAP_DRAWS,
                                  "bootstrap_seed": BOOTSTRAP_SEED}),
    }


def print_report(report: dict) -> None:
    inp = report["input"]
    print(f"transitions n={inp['n_transitions']} "
          f"(source_ids={inp['n_source_ids_retained']}, text_ids={inp['n_text_ids_retained']}, "
          f"rows_retained={inp['n_rows_retained']}/{inp['n_rows_total']})")
    g1 = report["g_s2_1"]
    print(f"G-S2-1: {g1['n_folds_beating_best_null']}/{g1['n_folds_requested']} folds beat "
          f"best null (run {g1['n_folds_run']})  mean_model_mse={g1['mean_model_mse']:.5f}  "
          f"mean_best_null_mse={g1['mean_best_null_mse']:.5f}  -> {g1['verdict']}")
    g2 = report["g_s2_2"]
    print(f"G-S2-2: step_down -> ell_(t+1) point={g2['point']:+.4f}  "
          f"CI95=[{g2['ci95'][0]:+.4f}, {g2['ci95'][1]:+.4f}]  -> {g2['verdict']}")
    s2 = g2["s_next_reported_not_gated"]
    print(f"        step_down -> s_(t+1)   point={s2['point']:+.4f}  "
          f"CI95=[{s2['ci95'][0]:+.4f}, {s2['ci95'][1]:+.4f}]  (reported, not gated)")
    g3 = report["g_s2_3"]
    print(f"G-S2-3 (recorded only): spectral_radius={g3['spectral_radius']:.4f}  "
          f"controllability_rank={g3['controllability_rank']}/{g3['state_dim']}  "
          f"gramian_condition={g3['gramian_condition']:.4e}")


def main() -> None:
    if DEFAULT_OUT_PATH.exists():
        raise SystemExit(f"refusing to overwrite existing {DEFAULT_OUT_PATH}")
    rows = read_jsonl(DEFAULT_ROWS_PATH)
    report = build_report(rows)
    DEFAULT_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_OUT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print_report(report)
    print(f"-> {DEFAULT_OUT_PATH}")


if __name__ == "__main__":
    main()
