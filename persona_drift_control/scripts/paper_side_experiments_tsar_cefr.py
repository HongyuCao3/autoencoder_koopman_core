#!/usr/bin/env python3
"""A1 / M1 / M2 on `tsar_cefr` -- the eleventh Table 1 column the 09-16 mechanism read
missed. Spec: ../docs/paper_side_experiments_plan_2026-09-16.md section 9.

WHY THIS COLUMN EXISTS AND THE OTHER TWO BEHAVIOURAL ONES SHOULD NOT CARRY
THE VERDICT. M1's 09-16 run read three behavioural lines. Two of them fail
the column-eligibility rule signed in section 9, on grounds that all predate
that run: `defense` is a closed line (09-13, gate R1 FAIL) scored by the
model's own judgement, and `gsm8k_sharded` was suspended on 09-08 with the
finding that its INPUT CHANNEL IS DECOUPLED FROM THE STATE -- which is
precisely the channel whose impulse response M1 claims to measure. The single
fact that drove the "fail" verdict, a response peaking at step 2, came from
that column. `tsar_cefr` is in the main table, is the only active line, is
scored by a fixed CEFR classifier, and its actions were excited open-loop at
collection time, so here the response is a measurement.

WHY A THIRD SCRIPT AND NOT A LINE IN THE BEHAVIOURAL ONE. Same reason
`eval_surrogate_rows_tsar_cefr.py` is separate and says so at length: the
behavioural data path reads ONE scalar action column, and this line's action
is a 3-dof one-hot over {step_down, half_step_down, paraphrase} with `copy`
as reference. Collapsing it to a scalar would shrink the measured action
effect toward zero -- the direction that makes the mechanism story look
better, which is the direction a guard has to block. `_assert_action_dofs`
raises instead.

WHAT IS SHARED. The operator, the state and the action encoding are imported
verbatim from `eval_surrogate_rows_tsar_cefr.py` (which imports them from
`fit_tsar_cefr_operator.py`), so this reads the same fit Table 1's column was
scored on. The spectrum / loading / response rules are copied cell-for-cell
from `paper_side_experiments_behavioral.run_m1` so the `constraint` and
`tsar_cefr` cells are read on one rule.

REPORTING. Spectral radius, half-life, tap loading and response length are
fit diagnostics; `../.claude/global.md` does not let them stand as cross-arm
headline numbers. Per the user's 09-17 ruling, screening only shows a signal
exists -- a side experiment has to report whether it bites ON THE TASK and by
how much -- so the artifact carries this column's landed task-level contrasts
(Table 1 rows 2 and 3, with their intervals) next to every diagnostic. The
09-15 postmortem's "the action cashes out in one step" is the screening, not
this column's reported number.

BOUNDARY: read-only on `outputs/tsar_cefr_gpu1/`, writes one new directory,
no GPU, no edits to any existing script.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys

import numpy as np

PDC_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PDC_ROOT / "src"))
sys.path.insert(0, str(PDC_ROOT.parent / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from persona_drift.modeling.koopman import KoopmanSurrogate, no_extra_features  # noqa: E402
from persona_drift.run_provenance import provenance  # noqa: E402

_SPEC = importlib.util.spec_from_file_location(
    "_eval_surrogate_rows_tsar_cefr", PDC_ROOT / "scripts" / "eval_surrogate_rows_tsar_cefr.py")
tsar = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(tsar)

OUT_DIRS = {
    "m1": "mechanism_operator_profile_tsar_cefr",
    "a1": "ablation_memory_depth_tsar_cefr",
    "a1_matched": "ablation_memory_depth_matched_tsar_cefr",
    "m2": "mechanism_item_effect_tsar_cefr",
}
COLLINEAR_R2 = 1.0 - 1e-8          # paper_side_experiments_behavioral.py
NEAR_COLLINEAR_R2 = 0.99           # paper_side_experiments_behavioral.py
BLOCK_DIM = 2                      # xi stacks (ell, s) per tap
TABLE1_PATH = PDC_ROOT / "outputs" / "surrogate_rows_tsar_cefr" / "tsar_cefr.json"


class ActionEncodingError(AssertionError):
    """The action reached the fit with the wrong number of degrees of freedom."""


def _assert_action_dofs(V: np.ndarray) -> None:
    """The one collapse this column must not survive silently.

    Folding the four actions into a scalar biases the measured action effect
    toward zero, i.e. toward agreeing with the mechanism claim under test. A
    quiet pass here would look like evidence.
    """
    expected = len(tsar.ACTIONS)
    if V.ndim != 2 or V.shape[1] != expected:
        raise ActionEncodingError(
            f"action matrix has shape {V.shape}; expected (n, {expected}) "
            f"for one-hot over {tsar.ACTIONS} with {tsar.REFERENCE_ACTION} as reference")


def _identifiability(Z: np.ndarray, V: np.ndarray) -> dict:
    """Copied from paper_side_experiments_behavioral._identifiability so the
    two eligible columns are screened by one rule."""
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


def _state_taps(A: np.ndarray, lag: int) -> list[float]:
    """How much of the next newest (ell, s) block each tap of z_t carries.

    Read off A, not C -- the lesson the first behavioural M1 run paid for: C
    here is fit against ell alone and a loading computed from it says only
    which slot holds ell. This line's xi is NEWEST-FIRST in blocks of
    (ell, s), so tap j occupies columns [j*2, j*2+2) and the newest block of
    z_(t+1) is rows [0, 2).
    """
    block = np.asarray(A)[:BLOCK_DIM, :]
    return [float(np.linalg.norm(block[:, j * BLOCK_DIM:(j + 1) * BLOCK_DIM]))
            for j in range(lag + 1)]


def _task_level_anchor() -> dict:
    """This column's landed Table 1 contrasts, copied verbatim, not recomputed.

    The mechanism numbers below are diagnostics. These are the reported ones,
    and they sit in the same artifact so no downstream reader can quote a
    half-life without the task-level effect it is supposed to explain.
    """
    landed = json.loads(TABLE1_PATH.read_text())
    contrasts = landed["contrasts"]
    return {
        "source": str(TABLE1_PATH.relative_to(PDC_ROOT)),
        "source_git_sha": landed["provenance"]["git_sha"],
        "readout": landed["readout"],
        "n_items": landed["n_items"],
        "n_scored_rows": landed["n_scored_rows"],
        "table1_row2_ours_minus_markov": contrasts["delay_linear_control_minus_markov_linear_control"],
        "table1_row3_ours_minus_withheld_action": contrasts["delay_linear_control_minus_delay_linear_no_control"],
        "note": ("row 3 is what an action channel is worth to the fit at all; row 2 is what the "
                 "delay window is worth. Both are Skill_H with a by-source bootstrap interval, "
                 "and they are the reportable numbers on this column."),
    }


def run_m1(rows_path: pathlib.Path) -> dict:
    """Spectrum, tap loading and action response on this line's own operator."""
    trajectories = tsar.build_trajectories(tsar.read_jsonl(rows_path))
    refused = sorted(k for k, t in trajectories.items() if t["has_nan"])
    kept = [t for k, t in trajectories.items() if k not in set(refused)]
    data = tsar.to_dataset([tr for traj in kept for tr in tsar.transitions(traj, tsar.LAG)])
    _assert_action_dofs(data["V"])

    model = KoopmanSurrogate(extra_features_fn=no_extra_features).fit(data)
    A, B, C = np.asarray(model.A), np.asarray(model.B), np.asarray(model.C)
    modulus = np.sort(np.abs(np.linalg.eigvals(A)))[::-1]
    rho = float(modulus[0])
    half_life = float(np.log(0.5) / np.log(rho)) if 0.0 < rho < 1.0 else None
    taps = _state_taps(A, tsar.LAG)
    total = float(np.sqrt(sum(v ** 2 for v in taps)))
    older = float(np.sqrt(sum(v ** 2 for v in taps[1:])))

    ident = _identifiability(data["Z"], data["V"])
    out = {
        "probe": "m1",
        "line": "tsar_cefr",
        "provenance": provenance(switches={"rows_path": str(rows_path), "lag": tsar.LAG,
                                           "horizon": tsar.HORIZON}),
        "judge": "fixed CEFR classifier (not the controlled model's self-judgement)",
        "nu": tsar.LAG + 1,
        "horizon": tsar.HORIZON,
        "n_transitions": int(len(data["Z"])),
        "n_trajectories": len(kept),
        "n_trajectories_refused_unparsed_readout": len(refused),
        "actions": list(tsar.ACTIONS),
        "reference_action": tsar.REFERENCE_ACTION,
        "spectrum": {"eigenvalue_modulus": [float(v) for v in modulus],
                     "spectral_radius": rho, "half_life_steps": half_life},
        "readout_loading": {
            "tap_norms": taps,
            "non_markov_share": (older / total) if total > 0 else None,
            "rule": "||A[:2, tap j]||_F over taps j=0..lag, newest-first blocks of (ell, s)",
        },
        "action_channel": "open_loop_excitation",
        "identifiability": ident,
        "task_level_anchor": _task_level_anchor(),
    }

    if not ident["identified"]:
        out["response"] = {"identified": False, "undefined_reason": ident["undefined_reason"],
                           "impulse_response": None, "step_response": None,
                           "response_length": None, "first_step_share": None}
        return out

    impulse, step, power, cumulative = [], [], np.eye(A.shape[0]), np.zeros_like(B)
    for _ in range(tsar.HORIZON):
        impulse.append(float(np.linalg.norm(C @ (power @ B))))
        cumulative = cumulative + power @ B
        step.append(float(np.linalg.norm(C @ cumulative)))
        power = A @ power
    peak = max(impulse) if impulse else 0.0
    total_response = float(sum(impulse))
    out["response"] = {
        "identified": True,
        "impulse_response": impulse,
        "step_response": step,
        "response_length": int(sum(1 for v in impulse if peak > 0 and v >= 0.1 * peak)),
        "response_length_rule": "steps with ||C A^(k-1) B|| >= 0.1 * peak",
        # The 09-16 run left the length threshold unsigned and the verdict
        # turned on it, so both statistics are written down here rather than
        # one of them being chosen after the fact.
        "first_step_share": (impulse[0] / total_response) if total_response > 0 else None,
        "first_step_share_rule": "impulse[0] / sum(impulse)",
        "peak_step": int(np.argmax(impulse)) + 1,
        # Added 2026-09-17, after the synthetic fixture in
        # tests/test_paper_side_experiments_tsar_cefr.py showed what the two
        # statistics above cannot tell apart. ||C A^(k-1) B|| is the effect
        # STILL PRESENT at step k, not the effect ARRIVING at step k, so a
        # one-step action into a state that keeps it -- an integrator, which
        # is what this line was diagnosed as on 09-15 -- reads as a long
        # response. The increments are the arriving part. Reported as a
        # diagnostic; no verdict field is computed from them.
        "impulse_increment": [impulse[0]] + [impulse[k] - impulse[k - 1]
                                             for k in range(1, len(impulse))],
        "impulse_increment_rule": "first difference of impulse_response; what arrives at step k",
    }
    # Signed, in CEFR levels: does the fitted B reproduce what each action is
    # supposed to do? A diagnostic on the fit's semantics, not a reported number.
    out["response"]["per_action_first_step_on_ell"] = {
        action: float((C @ B)[0, i]) for i, action in enumerate(tsar.ACTIONS)}
    return out


# --------------------------------------------------------------------------- A1 / M2

MIN_TURN_NEXT_MATCHED = tsar.LAG + 1


class LeakageError(AssertionError):
    """The item level was estimated on steps that are also scored."""


class _WithItemLevel:
    """The Markov row, handed the item's own level through the action channel.

    Same construction as the behavioural side: that channel is the one the fit
    already treats as exogenous, and a persistent item effect is exactly an
    exogenous constant. On THIS column the level is close to free -- ell_0 is
    `source_level_expected`, the item's identity -- so a large "the intercept
    buys this much" reading is expected here and the informative quantity is
    what survives it.
    """

    def __init__(self, inner, level: float):
        self.inner, self.level = inner, float(level)

    def step(self, z, v):
        return self.inner.step(z, np.append(np.asarray(v, dtype=float).reshape(-1), self.level))

    def readout(self, z):
        return self.inner.readout(z)


def _dataset(trajs, lag: int, *, min_turn_next: int | None = None, levels=None) -> dict:
    """Transitions at `lag`, optionally dropping the ones a deeper state
    cannot see, optionally with the item level appended to the action."""
    Z, V, Z_next, Y = [], [], [], []
    for traj in trajs:
        key = f"{traj['text_id']}|{traj['seed']}"
        for tr in tsar.transitions(traj, lag):
            if min_turn_next is not None and tr["turn_next"] < min_turn_next:
                continue
            v = np.asarray(tr["v"], dtype=float).reshape(-1)
            _assert_action_dofs(v.reshape(1, -1))
            if levels is not None:
                v = np.append(v, float(levels[key]))
            Z.append(tr["z"])
            V.append(v)
            Z_next.append(tr["z_next"])
            Y.append(tr["z"][tsar.ELL_INDEX])
    if not Z:
        raise ValueError("empty identification dataset")
    return {"Z": np.stack(Z), "V": np.stack(V), "Z_next": np.stack(Z_next),
            "Y": np.asarray(Y, dtype=float)}


def _item_levels(trajs, prefix_turn: int, scored_min_turn: int) -> dict:
    """Mean ell over the observed prefix, per trajectory.

    Raises rather than returns on overlap. A leak here does not fail loudly;
    it makes the baseline stronger and the paper's one positive claim quietly
    weaker for the wrong reason.
    """
    if prefix_turn >= scored_min_turn:
        raise LeakageError(
            f"item level would use steps <= {prefix_turn} but scoring starts at {scored_min_turn}")
    out = {}
    for traj in trajs:
        vals = [v for t, v in traj["ell"].items() if t <= prefix_turn]
        if not vals:
            raise ValueError(f"trajectory {traj['text_id']}|{traj['seed']} has no observed prefix")
        out[f"{traj['text_id']}|{traj['seed']}"] = float(np.mean(vals))
    return out


def _null_panel(trajs, recs=None) -> dict:
    """The trivial nulls' design, mirroring eval_surrogate_rows_tsar_cefr.run_on_trajectories.

    Always built at LAG and unmatched, so the denominator every variant is
    scored against is the same one Table 1 used.
    """
    cols = {"y_next": [], "turn_next": [], "target_level_rank": [],
            **{c: [] for c in tsar.U_COLS}}
    if recs is None:
        for traj in trajs:
            for tr in tsar.transitions(traj, tsar.LAG):
                cols["y_next"].append(tr["z_next"][tsar.ELL_INDEX])
                cols["turn_next"].append(float(tr["turn_next"]))
                cols["target_level_rank"].append(traj["target_level_rank"])
                for i, c in enumerate(tsar.U_COLS):
                    cols[c].append(float(tr["v"][i]))
    else:
        # Keyed by trajectory, not by item: one source_id carries both target
        # levels, so a source-keyed lookup hands half the rows the wrong rank.
        rank = {f"{t['text_id']}|{t['seed']}": t["target_level_rank"] for t in trajs}
        u_of = {(f"{t['text_id']}|{t['seed']}", k): t["u"][k] for t in trajs for k in t["u"]}
        for r in recs:
            cols["y_next"].append(r["y_true"])
            cols["turn_next"].append(float(r["turn"]))
            cols["target_level_rank"].append(rank[r["trajectory"]])
            v = u_of[(r["trajectory"], r["turn"])]
            for i, c in enumerate(tsar.U_COLS):
                cols[c].append(float(v[i]))
    return {k: np.asarray(v, dtype=float) for k, v in cols.items()}


def _scored(trajs, variants, *, bootstrap_seed: int = 0) -> dict:
    """One fold-averaged panel; every variant scored on identical rows.

    The window comes from the deepest state (`tsar.rollout` starts at LAG), so
    a shallower state is seeded at the same absolute step rather than starting
    earlier -- the rule Table 1 runs under.
    """
    sources = sorted({t["source_id"] for t in trajs})
    folds = tsar.make_folds(sources, tsar.N_FOLDS, purpose="report", seed=bootstrap_seed)
    collected: dict[str, list[dict]] = {}
    null_pred: dict[str, list[np.ndarray]] = {}
    meta_rows: list[dict] = []

    for fold in folds:
        test_sources = set(fold["test_groups"].tolist())
        train = [t for t in trajs if t["source_id"] not in test_sources]
        test = [t for t in trajs if t["source_id"] in test_sources]
        built = {name: build(train) for name, (_, build) in variants.items()}

        fold_meta = None
        for name, (lag, _) in variants.items():
            emitted = []
            for traj in test:
                key = f"{traj['text_id']}|{traj['seed']}"
                for rec in tsar.rollout(built[name](key), traj, lag, False):
                    emitted.append(dict(rec, trajectory=key, item=traj["source_id"]))
            collected.setdefault(name, []).extend(emitted)
            if fold_meta is None:
                fold_meta = emitted
            else:
                assert [(r["trajectory"], r["turn"], r["step"]) for r in emitted] == \
                       [(r["trajectory"], r["turn"], r["step"]) for r in fold_meta], \
                       f"{name} scored on different rows"
        meta_rows.extend(fold_meta)

        preds = tsar.trivial_nulls(_null_panel(train), _null_panel(test, fold_meta),
                                   exogenous=tsar.EXOGENOUS)
        for k, v in preds.items():
            null_pred.setdefault(k, []).append(v)

    y_true = np.asarray([r["y_true"] for r in meta_rows], dtype=float)
    groups = np.asarray([r["item"] for r in meta_rows])
    null_name, y_null = tsar.best_null({k: np.concatenate(v) for k, v in null_pred.items()}, y_true)
    se_null = (y_null - y_true) ** 2
    se = {k: (np.asarray([r["y_pred"] for r in v], dtype=float) - y_true) ** 2
          for k, v in collected.items()}

    def cell(se_model):
        out = {"rollout_mse": float(np.mean(se_model)), "null_mse": float(np.mean(se_null)),
               "n_rows": int(se_model.size)}
        try:
            out["skill_h"] = tsar.skill_from_squared_errors(se_model, se_null, horizon=tsar.HORIZON)
        except tsar.DegenerateNullError as exc:
            out["skill_h"], out["degenerate"] = None, str(exc)
            return out
        out["bootstrap"] = tsar.bootstrap_ci(
            np.column_stack([se_model, se_null]), groups, seed=bootstrap_seed,
            statistic=lambda v: 1.0 - float(np.mean(v[:, 0])) / float(np.mean(v[:, 1])))
        return out

    def contrast(a: str, b: str) -> dict:
        return tsar.bootstrap_ci(
            np.column_stack([se[a], se[b], se_null]), groups, seed=bootstrap_seed,
            statistic=lambda v: (float(np.mean(v[:, 1])) - float(np.mean(v[:, 0]))) / float(np.mean(v[:, 2])))

    return {
        "line": "tsar_cefr", "provenance": provenance(),
        "readout": "level_expected (fixed CEFR classifier, not the controlled model's self-judgement)",
        "lag_max": tsar.LAG, "horizon": tsar.HORIZON, "n_folds": tsar.N_FOLDS,
        "n_items": len(sources), "n_trajectories": len(trajs),
        "n_scored_rows": int(y_true.size), "best_null_name": null_name,
        "scored_turn_min": int(min(r["turn"] for r in meta_rows)),
        "rows": {k: cell(v) for k, v in se.items()},
        "_contrast": contrast,
    }


def _trajectories(rows_path: pathlib.Path):
    built = tsar.build_trajectories(tsar.read_jsonl(rows_path))
    return [built[k] for k in sorted(built) if not built[k]["has_nan"]]


def run_a1(rows_path: pathlib.Path, *, matched_window: bool) -> dict:
    """Skill at lag 0 vs lag 1 on identical scored rows.

    Two cells is the whole scan space at T=6 with LAG=1, so this column cannot
    speak to memory DEPTH -- it can only say whether the one tap pays, and
    whether the training-window confound the core side found (a shallow state
    is usable earlier, so it trains on transitions the deep state never sees)
    is present here too.
    """
    trajs = _trajectories(rows_path)
    min_turn_next = MIN_TURN_NEXT_MATCHED if matched_window else None

    def make(lag):
        def build(train):
            model = KoopmanSurrogate(extra_features_fn=no_extra_features).fit(
                _dataset(train, lag, min_turn_next=min_turn_next))
            return lambda key: model
        return lag, build

    variants = {f"lag_{lag}": make(lag) for lag in range(tsar.LAG + 1)}
    out = _scored(trajs, variants)
    contrast = out.pop("_contrast")
    out.update({
        "probe": "a1", "matched_window": bool(matched_window),
        "scan_space": "two cells (lag 0 and 1); T=6 with LAG=1 leaves no depth to scan",
        "contrasts": {"lag_1_minus_lag_0": contrast("lag_1", "lag_0")},
    })
    return out


def run_m2(rows_path: pathlib.Path) -> dict:
    """Does the delay-embedding gain survive handing Markov the item's level?

    Every row is fit on the matched training window, so the only thing that
    differs across rows is the state -- the core side showed that letting a
    shallow state train on earlier transitions turns a depth comparison into a
    training-window comparison.
    """
    trajs = _trajectories(rows_path)
    prefix_turn = tsar.LAG                    # the deepest state's first usable step
    scored_min_turn = tsar.LAG + 1
    levels = _item_levels(trajs, prefix_turn, scored_min_turn)

    def plain(lag):
        def build(train):
            model = KoopmanSurrogate(extra_features_fn=no_extra_features).fit(
                _dataset(train, lag, min_turn_next=MIN_TURN_NEXT_MATCHED))
            return lambda key: model
        return lag, build

    def with_item(lag):
        def build(train):
            model = KoopmanSurrogate(extra_features_fn=no_extra_features).fit(
                _dataset(train, lag, min_turn_next=MIN_TURN_NEXT_MATCHED, levels=levels))
            return lambda key: _WithItemLevel(model, levels[key])
        return lag, build

    variants = {"ours": plain(tsar.LAG), "markov": plain(0),
                "markov_plus_item": with_item(0)}
    out = _scored(trajs, variants)
    contrast = out.pop("_contrast")
    if out["scored_turn_min"] <= prefix_turn:
        raise LeakageError(
            f"scoring starts at step {out['scored_turn_min']} but the item level "
            f"used steps <= {prefix_turn}")
    out.update({
        "probe": "m2", "item_level_prefix_turn": prefix_turn,
        "item_level_note": ("ell_0 is source_level_expected, so on this column the item level is "
                            "close to a free readout of item identity; read the residual, not the "
                            "share the intercept buys"),
        "contrasts": {
            "ours_minus_markov_plus_item": contrast("ours", "markov_plus_item"),
            "ours_minus_markov": contrast("ours", "markov"),
            "markov_plus_item_minus_markov": contrast("markov_plus_item", "markov"),
        },
    })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", choices=sorted(OUT_DIRS), required=True)
    parser.add_argument("--rows-path", type=pathlib.Path, default=tsar.DEFAULT_ROWS_PATH)
    parser.add_argument("--out-dir", type=pathlib.Path, default=None)
    args = parser.parse_args()

    out_dir = args.out_dir or PDC_ROOT / "outputs" / OUT_DIRS[args.probe]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "tsar_cefr.json"
    if out_path.exists():
        raise FileExistsError(f"{out_path} exists; outputs are never overwritten")

    if args.probe == "m1":
        result = run_m1(args.rows_path)
    elif args.probe == "m2":
        result = run_m2(args.rows_path)
    else:
        result = run_a1(args.rows_path, matched_window=args.probe == "a1_matched")
    out_path.write_text(json.dumps(result, indent=2, default=float) + "\n")
    print(f"[done] {args.probe} tsar_cefr -> {out_path}", flush=True)

    if args.probe == "m1":
        r = result["response"]
        print(f"       rho={result['spectrum']['spectral_radius']:.4f} "
              f"half_life={result['spectrum']['half_life_steps']} "
              f"non_markov_share={result['readout_loading']['non_markov_share']:.4f}", flush=True)
        if r["identified"]:
            print(f"       impulse={[round(v, 5) for v in r['impulse_response']]} "
                  f"L={r['response_length']} first_step_share={r['first_step_share']:.4f} "
                  f"peak_step={r['peak_step']}", flush=True)
        return
    for name, row in result["rows"].items():
        skill = row.get("skill_h")
        print(f"       {name:20s} skill_h={skill if skill is None else round(skill, 4)}", flush=True)
    for name, c in result["contrasts"].items():
        print(f"       {name:34s} {c['point']:+.4f} [{c['ci_low']:+.4f},{c['ci_high']:+.4f}] "
              f"{'*' if c['excludes_zero'] else ''}", flush=True)


if __name__ == "__main__":
    main()
