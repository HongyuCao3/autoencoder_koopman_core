#!/usr/bin/env python3
"""Table 1's six rows on `tsar_cefr` -- the eleventh column (plan
docs/operational_plan_tsar_cefr_line_2026-09-13.md section 7.1). CPU only.

WHY A SEPARATE SCRIPT AND NOT A `LINES` ENTRY IN
`eval_surrogate_rows_behavioral.py`. That script's whole data path goes
through `modeling.dataset.build_reduced_state_pairs`, which reads ONE scalar
action column (`float(row[u_col])`). This line's action is a four-valued
categorical carried as a 3-dof one-hot over {step_down, half_step_down,
paraphrase} with `copy` as the reference class -- plan section 5.1, and the
encoding `fit_tsar_cefr_operator.py` already signed and fit the Phase 3
operator with. Folding it down to a scalar is the one thing this column must
not do: row 3 measures what is LOST when the action is withheld, so
collapsing four actions into one number would shrink row 3 toward zero --
i.e. toward agreeing with the other ten columns, which is the direction that
flatters the paper's claim.

WHAT IS SHARED AND WHAT IS NOT. `MAIN_TABLE_DESIGN.md` section 4 states the
contract: "跨列可比的是 Skill_H 的定义与区间构造，不是拟合器" -- so the SCORING
runs through the same `surrogate_eval` harness as all ten existing columns
(`skill_from_squared_errors`, `trivial_nulls`, `best_null`, `bootstrap_ci`,
`make_folds(purpose="report")`), while the STATE and ACTION construction is
imported verbatim from this line's own `fit_tsar_cefr_operator.py` rather
than retyped. The row definitions, the fold protocol, the fairness rule
(every row rolled from the SAME window, each model seeded from ITS OWN state
at that same absolute step) and the contrast statistic mirror
`eval_surrogate_rows_behavioral.py` one-for-one.

STATE. xi_t = [ell_t, s_t, ell_(t-1), s_(t-1)] at LAG=1, exactly the operator
the MPC actually ran on -- including its two judgment calls (ell_0 =
source_level_expected, s_0 = 1.0). Row 2 (Markov) is the same pair without
the lag block. Scored readout is ell (index 0), the controlled variable;
G-S2-1 scores [ell, s] jointly, but Table 1's `Skill_H` and `rollout MSE` are
defined on a single readout in its native unit.

HORIZON. H = 4 -- the MPC planning horizon (plan section 5.4), the same
target the other behavioural columns use. At T = 6 with LAG = 1 the deepest
state leaves exactly two H=4 windows per trajectory (t0 = 1, 2).

EXOGENOUS. The `stateless` null gets `turn_next`, `target_level_rank`, and
the three action dummies. `turn_next` is mandatory (`trivial_nulls` raises
without it). The action dummies are the direct analogue of the `v` the three
behavioural columns pass. `target_level_rank` is this line's environment-fixed
quantity -- `MAIN_TABLE_DESIGN.md` section 1 defines exogenous as what the
environment fixes in advance and names core's `r` and `gsm8k_sharded`'s
`shard_frac` as the per-line instances; the target CEFR level is ours, and
`fit_tsar_cefr_operator.py`'s own G-S2-1 null already uses it. Withholding it
would hand every fitted row a free win on a constant it did not have to learn.

PRE-REGISTERED PREDICTION (plan section 7.1): row 3, `ours - withheld u`,
is significantly NON-zero on this column -- the opposite of all ten existing
ones. This script does not know the prediction; it is recorded here so the
verdict is read against what was written down before it ran.

BOUNDARY: this script computes Table 1's column only. No operator re-fit, no
closed-loop replay, no docs/ edits, no GPU.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import defaultdict

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
PDC_ROOT = REPO_ROOT / "persona_drift_control"
sys.path.insert(0, str(PDC_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from persona_drift.modeling.ae_baseline import AEKoopmanConfig, AEKoopmanSurrogate   # noqa: E402
from persona_drift.modeling.koopman import KoopmanSurrogate, no_extra_features       # noqa: E402
from persona_drift.modeling.lstm_baseline import (                                   # noqa: E402
    mse_from_predictions,
    rollout_predictions,
    train_lstm_surrogate,
)
from persona_drift.run_provenance import provenance                                  # noqa: E402
from surrogate_eval import (                                                         # noqa: E402
    best_null,
    bootstrap_ci,
    make_folds,
    skill_from_squared_errors,
    trivial_nulls,
)
from surrogate_eval.skill import DegenerateNullError                                 # noqa: E402

from fit_tsar_cefr_operator import (                                                 # noqa: E402
    ACTIONS,
    CEFR_RANK,
    EXCLUDED_TEXT_IDS,
    LAG,
    REFERENCE_ACTION,
    action_one_hot,
    read_jsonl,
)

DEFAULT_ROWS_PATH = PDC_ROOT / "outputs" / "tsar_cefr_gpu1" / "trajectories.jsonl"
DEFAULT_OUT_DIR = PDC_ROOT / "outputs" / "surrogate_rows_tsar_cefr"

HORIZON = 4                              # plan section 5.4 (the MPC horizon)
N_FOLDS = 5                              # eval_surrogate_rows_behavioral.py
LSTM_HIDDEN_CANDIDATES = (4, 8, 16)      # eval_surrogate_rows_behavioral.py
AE_EPOCHS = 3000                         # eval_surrogate_rows_behavioral.py
BOOTSTRAP_SEED = 0                       # eval_surrogate_rows_behavioral.py
ELL_INDEX = 0                            # xi = [ell_t, s_t, ell_(t-1), s_(t-1)]
U_COLS = tuple(f"u_{a}" for a in ACTIONS)
EXOGENOUS = ("turn_next", "target_level_rank", *U_COLS)
ROW_ORDER = ("markov_linear_control", "delay_linear_no_control", "lstm",
             "ae_koopman", "delay_linear_control")
OURS = "delay_linear_control"


def build_trajectories(rows: list[dict]) -> dict[tuple, dict]:
    """{(text_id, seed): trajectory} with the per-step state this line signed.

    `ell`/`s` are indexed by step with step 0 = the source paragraph, exactly
    as `fit_tsar_cefr_operator.build_transitions` defines them; `u[t]` is the
    one-hot of the action that PRODUCED step `t`, so it is defined for
    t = 1..T and undefined at t = 0.
    """
    excluded = set(EXCLUDED_TEXT_IDS)
    by_traj: dict[tuple, dict[int, dict]] = defaultdict(dict)
    meta: dict[tuple, dict] = {}
    for row in rows:
        if row["text_id"] in excluded:
            continue
        key = (row["text_id"], row["seed"])
        by_traj[key][int(row["step"])] = row
        meta[key] = row
    out = {}
    for key, steps in by_traj.items():
        m = meta[key]
        ell = {0: float(m["source_level_expected"])}
        s = {0: 1.0}
        u = {}
        for t, row in steps.items():
            ell[t] = float(row["level_expected"])
            s[t] = float(row["meaning_to_source"])
            u[t] = action_one_hot(row["action"])
        out[key] = {
            "text_id": key[0], "seed": key[1], "source_id": m["source_id"],
            "target_level_rank": float(CEFR_RANK[m["target_cefr"]]),
            "max_t": max(steps), "ell": ell, "s": s, "u": u,
            "has_nan": any(v != v for v in (*ell.values(), *s.values())),
        }
    return out


def xi(traj: dict, t: int, lag: int) -> np.ndarray:
    return np.array([v for k in range(lag + 1) for v in (traj["ell"][t - k], traj["s"][t - k])])


def transitions(traj: dict, lag: int) -> list[dict]:
    """(xi_t, u_(t+1)) -> xi_(t+1) for every t the state is defined at."""
    return [
        {"z": xi(traj, t, lag), "v": traj["u"][t + 1], "z_next": xi(traj, t + 1, lag), "turn_next": t + 1}
        for t in range(lag, traj["max_t"])
    ]


def to_dataset(trs: list[dict]) -> dict:
    Z = np.stack([tr["z"] for tr in trs])
    return {"Z": Z, "V": np.stack([tr["v"] for tr in trs]),
            "Z_next": np.stack([tr["z_next"] for tr in trs]), "Y": Z[:, ELL_INDEX]}


class _NoControl:
    """Row 3: the same fit with the action withheld at fit AND at rollout."""

    def __init__(self, inner):
        self.inner = inner

    def step(self, z, v):
        return self.inner.step(z, np.zeros_like(np.asarray(v, dtype=float)))

    def readout(self, z):
        return self.inner.readout(z)


class _LSTMPrefix:
    """Row 4 seeded from the same observed window as the rest.

    Alignment is `eval_surrogate_rows_behavioral._LSTMPrefix.warm`'s, pair for
    pair: the cell consumes (ell_i, u_(i+1)) for i = 0..t0, so its hidden
    state has absorbed ell_(t0) -- the last observation its opponents' states
    contain. The cell's input is [y, v] jointly, so absorbing ell_(t0)
    necessarily feeds u_(t0+1) alongside it; that is the architecture's price
    and it is paid identically on the three published columns.
    """

    def __init__(self, model):
        self.model = model

    def warm(self, traj: dict, t0: int) -> np.ndarray:
        ys = [traj["ell"][i] for i in range(t0 + 1)]
        vs = [traj["u"][min(i + 1, traj["max_t"])] for i in range(t0 + 1)]
        return self.model.warm_start(ys, vs)

    def step(self, z, v):
        return self.model.step(z, v)

    def readout(self, z):
        return self.model.readout(z)


def lstm_rows(traj: dict) -> list[dict]:
    """`train_lstm_surrogate` wants flat per-turn dicts; steps 1..T only, so
    the synthetic step-0 state never enters the LSTM's training set."""
    return [
        dict({"trajectory_id": f"{traj['text_id']}|{traj['seed']}", "turn": t, "ell": traj["ell"][t]},
             **{c: float(traj["u"][t][i]) for i, c in enumerate(U_COLS)})
        for t in range(1, traj["max_t"] + 1)
    ]


def fit_rows(train: list[dict], seed: int):
    deep = to_dataset([tr for traj in train for tr in transitions(traj, LAG)])
    flat = to_dataset([tr for traj in train for tr in transitions(traj, 0)])

    ours = KoopmanSurrogate(extra_features_fn=no_extra_features).fit(deep)
    markov = KoopmanSurrogate(extra_features_fn=no_extra_features).fit(flat)
    blind = _NoControl(KoopmanSurrogate(extra_features_fn=no_extra_features)
                       .fit(dict(deep, V=np.zeros_like(deep["V"]))))

    cut = max(1, int(0.8 * len(train)))
    val = train[cut:]
    ae_val = to_dataset([tr for traj in val for tr in transitions(traj, LAG)]) if val else None
    ae = AEKoopmanSurrogate(
        state_dim=deep["Z"].shape[1], y_index=ELL_INDEX,
        config=AEKoopmanConfig(latent_dim=4, num_epochs=AE_EPOCHS, random_state=seed,
                               early_stopping_patience=30),
    ).fit(deep, val_dataset=ae_val)

    by_traj_train = [lstm_rows(t) for t in train[:cut]]
    by_traj_val = [lstm_rows(t) for t in val]
    best, best_loss, best_hidden = None, float("inf"), LSTM_HIDDEN_CANDIDATES[0]
    for hidden in LSTM_HIDDEN_CANDIDATES:
        model, _ = train_lstm_surrogate(
            hidden, by_traj_train, by_traj_val, y_col="ell", u_col=list(U_COLS),
            seed=seed, contemporaneous_v=True)
        loss = mse_from_predictions([
            rollout_predictions(model, rs, y_col="ell", u_col=list(U_COLS), contemporaneous_v=True)
            for rs in by_traj_val])
        if loss < best_loss:
            best, best_loss, best_hidden = model, loss, hidden
    return {
        "markov_linear_control": (markov, 0, False),
        "delay_linear_no_control": (blind, LAG, False),
        "lstm": (_LSTMPrefix(best), LAG, True),
        "ae_koopman": (ae, LAG, False),
        "delay_linear_control": (ours, LAG, False),
    }, {"lstm_hidden": best_hidden, "lstm_val_mse": float(best_loss)}


def rollout(predictor, traj: dict, model_lag: int, prefix: bool) -> list[dict]:
    """Per-row (step, y_true, y_pred) over the windows the DEEPEST state
    defines, so every row is scored on an identical set of steps."""
    starts = [t0 for t0 in range(LAG, traj["max_t"] - HORIZON + 1)]
    out = []
    for t0 in starts:
        z = predictor.warm(traj, t0) if prefix else xi(traj, t0, model_lag)
        for k in range(HORIZON):
            z = predictor.step(z, traj["u"][t0 + k + 1])
            out.append({"t0": t0, "step": k, "turn": t0 + k + 1,
                        "y_true": traj["ell"][t0 + k + 1], "y_pred": float(predictor.readout(z))})
    return out


def run(rows_path: pathlib.Path, bootstrap_seed: int = BOOTSTRAP_SEED) -> dict:
    trajectories = build_trajectories(read_jsonl(rows_path))
    refused = sorted(k for k, t in trajectories.items() if t["has_nan"])
    kept = [trajectories[k] for k in sorted(trajectories) if not trajectories[k]["has_nan"]]
    result = run_on_trajectories(kept, bootstrap_seed=bootstrap_seed)
    result["rows_path"] = str(rows_path.relative_to(REPO_ROOT))
    result["n_trajectories_refused_unparsed_readout"] = len(refused)
    return result


def run_on_trajectories(trajs: list[dict], *, bootstrap_seed: int = BOOTSTRAP_SEED) -> dict:
    """The whole column, given already-built trajectories.

    Split out from `run` so the planted positive/negative controls in
    `tests/test_eval_surrogate_rows_tsar_cefr.py` exercise the identical code
    path without staging a file -- a control that runs a different path from
    the reported number is not a control.
    """
    sources = sorted({t["source_id"] for t in trajs})
    folds = make_folds(sources, N_FOLDS, purpose="report", seed=bootstrap_seed)

    collected: dict[str, list[dict]] = {}
    null_pred: dict[str, list[np.ndarray]] = {}
    meta_rows: list[dict] = []
    extras = []
    for fold in folds:
        test_sources = set(fold["test_groups"].tolist())
        train = [t for t in trajs if t["source_id"] not in test_sources]
        test = [t for t in trajs if t["source_id"] in test_sources]
        fitted, extra = fit_rows(train, bootstrap_seed)
        extras.append(extra)

        fold_meta = None
        for name in ROW_ORDER:
            model, model_lag, prefix = fitted[name]
            emitted = []
            for traj in test:
                for rec in rollout(model, traj, model_lag, prefix):
                    emitted.append(dict(rec, trajectory=f"{traj['text_id']}|{traj['seed']}",
                                        item=traj["source_id"]))
            collected.setdefault(name, []).extend(emitted)
            if fold_meta is None:
                fold_meta = emitted
            else:
                assert [(r["trajectory"], r["turn"], r["step"]) for r in emitted] == \
                       [(r["trajectory"], r["turn"], r["step"]) for r in fold_meta], \
                       f"{name} scored on different rows"
        meta_rows.extend(fold_meta)

        def panel(tjs, recs=None):
            cols = {"y_next": [], "turn_next": [], "target_level_rank": [], **{c: [] for c in U_COLS}}
            if recs is None:
                for traj in tjs:
                    for tr in transitions(traj, LAG):
                        cols["y_next"].append(tr["z_next"][ELL_INDEX])
                        cols["turn_next"].append(float(tr["turn_next"]))
                        cols["target_level_rank"].append(traj["target_level_rank"])
                        for i, c in enumerate(U_COLS):
                            cols[c].append(float(tr["v"][i]))
            else:
                # Keyed by TRAJECTORY, not by `item`: one `source_id` carries
                # both target levels (A2 and B1), so a source-keyed lookup
                # would hand half the rows the other target's rank.
                rank = {f"{t['text_id']}|{t['seed']}": t["target_level_rank"] for t in tjs}
                u_of = {(f"{t['text_id']}|{t['seed']}", k): t["u"][k]
                        for t in tjs for k in t["u"]}
                for r in recs:
                    cols["y_next"].append(r["y_true"])
                    cols["turn_next"].append(float(r["turn"]))
                    cols["target_level_rank"].append(rank[r["trajectory"]])
                    v = u_of[(r["trajectory"], r["turn"])]
                    for i, c in enumerate(U_COLS):
                        cols[c].append(float(v[i]))
            return {k: np.asarray(v, dtype=float) for k, v in cols.items()}

        preds = trivial_nulls(panel(train), panel(test, fold_meta), exogenous=EXOGENOUS)
        for k, v in preds.items():
            null_pred.setdefault(k, []).append(v)

    y_true = np.asarray([r["y_true"] for r in meta_rows], dtype=float)
    groups = np.asarray([r["item"] for r in meta_rows])
    null_name, y_null = best_null({k: np.concatenate(v) for k, v in null_pred.items()}, y_true)
    se_null = (y_null - y_true) ** 2

    se = {k: (np.asarray([r["y_pred"] for r in v], dtype=float) - y_true) ** 2
          for k, v in collected.items()}

    def cell(se_model):
        out = {"rollout_mse": float(np.mean(se_model)), "null_mse": float(np.mean(se_null)),
               "n_rows": int(se_model.size)}
        try:
            out["skill_h"] = skill_from_squared_errors(se_model, se_null, horizon=HORIZON)
        except DegenerateNullError as exc:
            out["skill_h"], out["degenerate"] = None, str(exc)
            return out
        out["bootstrap"] = bootstrap_ci(
            np.column_stack([se_model, se_null]), groups, seed=bootstrap_seed,
            statistic=lambda v: 1.0 - float(np.mean(v[:, 0])) / float(np.mean(v[:, 1])))
        return out

    table = {"best_null": {"which": null_name, "rollout_mse": float(np.mean(se_null)),
                           "skill_h": 0.0, "n_rows": int(y_true.size)}}
    for k in ROW_ORDER:
        table[k] = cell(se[k])

    contrasts = {}
    for k in ROW_ORDER:
        if k == OURS:
            continue
        contrasts[f"{OURS}_minus_{k}"] = bootstrap_ci(
            np.column_stack([se[OURS], se[k], se_null]), groups, seed=bootstrap_seed,
            statistic=lambda v: (float(np.mean(v[:, 1])) - float(np.mean(v[:, 0]))) / float(np.mean(v[:, 2])))

    return {
        "line": "tsar_cefr", "provenance": provenance(),
        "readout": "level_expected (fixed CEFR classifier, not the controlled model's self-judgement)",
        "excluded_text_ids": list(EXCLUDED_TEXT_IDS),
        "lag": LAG, "horizon": HORIZON, "n_folds": N_FOLDS,
        "actions": list(ACTIONS), "reference_action": REFERENCE_ACTION,
        "exogenous": list(EXOGENOUS),
        "n_items": len(sources), "n_trajectories": len(trajs),
        "n_seeds": len({t["seed"] for t in trajs}),
        "n_scored_rows": int(y_true.size), "best_null_name": null_name,
        "rows": table, "contrasts": contrasts, "lstm_selection": extras,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=pathlib.Path, default=DEFAULT_ROWS_PATH)
    parser.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "tsar_cefr.json"
    if out.exists():
        raise FileExistsError(f"{out} exists; outputs are never overwritten")
    result = run(args.rows)
    out.write_text(json.dumps(result, indent=2, default=float) + "\n")
    print(f"[done] tsar_cefr lag={result['lag']} H={result['horizon']} items={result['n_items']} "
          f"trajectories={result['n_trajectories']} rows={result['n_scored_rows']} "
          f"null={result['best_null_name']}")
    for k, v in result["rows"].items():
        s = v.get("skill_h")
        print(f"  {k:28s} skill_h={'NA' if s is None else f'{s:+.4f}'}  mse={v['rollout_mse']:.4f}")
    for k, v in result["contrasts"].items():
        print(f"  {k:46s} {v['point']:+.4f}  CI[{v['ci_low']:+.4f}, {v['ci_high']:+.4f}]  excl0={v['excludes_zero']}")


if __name__ == "__main__":
    main()
