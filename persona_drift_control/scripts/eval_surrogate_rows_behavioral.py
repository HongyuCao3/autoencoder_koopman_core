#!/usr/bin/env python3
"""Table 1's six rows, on the three multi-turn behavioural datasets.

The core side's caller is `scripts/eval_surrogate_rows.py` in the root
package; this is the behavioural side's. Both score through the same
`surrogate_eval` harness (skill, nulls, bootstrap, folds), which is what makes
the ten columns readable against each other. The MODEL CLASSES here are this
line's own (`modeling.koopman` / `lstm_baseline` / `ae_baseline`), not the
core package's -- the two code bases do not import each other, and using each
line's native classes keeps these rows comparable to that line's own gates.

WHAT IS NEW HERE, AND WHY IT IS NOT A RE-TABULATION. Every gate these lines
signed is ONE-STEP at `nu=1` -- `constraint`'s S2 config is literally
`{nu: 1, mu: 1}`, a memoryless state. Row 6 (delay-embedded + control) has
never been fit on any of them. So these numbers are new measurements, and row
2 (`nu=1` + control) is the row that corresponds to the published operator.

DEPTH AND HORIZON ARE PER LINE, same rule as the core side: the deepest
y-block that still leaves H >= 3 free steps, targeting H = 4 (the MPC planning
horizon). `defense` has only 5 turns, so it runs nu=2/H=3; the others run
nu=4/H=4.

FAIRNESS. Every row is rolled from the SAME observed prefix. The Koopman rows
get it from `build_reduced_state_pairs`; the LSTM gets it from
`LSTMSurrogate.warm_start`, added for this comparison because its native
rollout starts from an all-zero `(h, c)` -- scoring it cold against a
prefix-seeded opponent would make the verdict about the seeding, not the model
class. The AE row reads `z[nu-1]` via `AEKoopmanSurrogate(y_index=...)`,
because the y-block is oldest-first and a reader left at 0 silently scores
against a y from `nu-1` turns ago.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess

import numpy as np

from persona_drift.modeling.ae_baseline import AEKoopmanConfig, AEKoopmanSurrogate
from persona_drift.modeling.dataset import (
    ReducedStateConfig,
    build_identification_dataset,
    build_reduced_state_pairs,
    group_by_trajectory,
)
from persona_drift.modeling.koopman import KoopmanSurrogate, no_extra_features
from persona_drift.modeling.lstm_baseline import (
    mse_from_predictions,
    rollout_predictions,
    train_lstm_surrogate,
)
from surrogate_eval import best_null, bootstrap_ci, seed_aggregate, skill_from_squared_errors, trivial_nulls
from surrogate_eval.skill import DegenerateNullError

REPO = pathlib.Path(__file__).resolve().parents[1]
TARGET_HORIZON = 4
MIN_HORIZON = 3
NU_CANDIDATES = (4, 3, 2, 1)
LSTM_HIDDEN_CANDIDATES = (4, 8, 16)
N_FOLDS = 5

LINES = {
    "constraint": {
        "readout": "outputs/sequor_s1_arm/readout_independent_qwen3_14b.json",
        "item_col": "item_id", "y_col": "y_graded", "u_col": "u_remind",
        "branches": ("bernoulli", "antithetic"),
        "judge": "independent (Qwen/Qwen3-14B)",
        "note": "S2 signed {nu: 1, mu: 1, contemporaneous_v: True}; rows 2 and 6 differ only in nu.",
    },
    "gsm8k_sharded": {
        "trajectories": "outputs/ergo_upB_random_excite/trajectories.jsonl",
        "item_col": "item_id", "y_col": "closeness", "u_col": "u_reset",
        "branches": None,
        "judge": "deterministic closeness (no judge)",
        "note": "upstream prompt_profile excitation arm; EK-A's counterfactual pairs are one-step by construction and cannot be rolled.",
    },
    "defense": {
        "trajectories": "outputs/koopman_defense_phaseB_random_excite/trajectories.jsonl",
        "item_col": "attack_id", "y_col": "y_safety", "u_col": "u_remind",
        "branches": None,
        "judge": "self (Qwen3-4B) -- global.md's named exception; carry the one-sided-miss caveat wherever this column is cited",
        "note": "T=5 only, so nu=2/H=3.",
    },
    # SECOND BACKBONE (google/gemma-4-E4B-it), plan
    # docs/experiments/second_backbone_plan_2026-09-17.md. Same fit, same
    # readout, same judge, same folds as the line above it -- the ONLY thing
    # that differs is which model wrote the text. `defense` has no gemma twin
    # by the 2026-09-17 user ruling (its judge is the agent itself, so swapping
    # the backbone would swap agent and judge at once).
    "constraint_gemma4": {
        "readout": "outputs/sequor_s1_arm_gemma4/readout_independent_qwen3_14b.json",
        "item_col": "item_id", "y_col": "y_graded", "u_col": "u_remind",
        "branches": ("bernoulli", "antithetic"),
        "judge": "independent (Qwen/Qwen3-14B) -- the judge does NOT move with the backbone",
        "note": "gemma-4-E4B-it arm 16021868, decoding model_default (1.0/0.95/64) against the "
                "Qwen column's 0.7/0.80/20: this column is 'another backbone under its own "
                "default decoding', not 'the same decoding with other weights'.",
    },
    "gsm8k_sharded_gemma4": {
        "trajectories": "outputs/ergo_upB_random_excite_gemma4/trajectories.jsonl",
        "item_col": "item_id", "y_col": "closeness", "u_col": "u_reset",
        "branches": None,
        "judge": "deterministic closeness (no judge)",
        "note": "gemma-4-E4B-it arm 16025913 at cap 2048, calibrated on this backbone by the two "
                "smoke passes 16019470/16021852; the Qwen column's cap 512 does not hold here.",
    },
}

# The published three, so `--lines` with no argument still runs exactly what it
# ran before the gemma entries existed.
DEFAULT_LINES = ("constraint", "gsm8k_sharded", "defense")


def provenance() -> dict:
    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(REPO), *args], check=True,
                              capture_output=True, text=True).stdout.strip()
    dirty = git("status", "--porcelain")
    return {"git_sha": git("rev-parse", "HEAD"), "git_dirty": bool(dirty),
            "n_dirty_paths": len(dirty.splitlines()), "n_folds": N_FOLDS,
            "target_horizon": TARGET_HORIZON}


def load_rows(spec: dict) -> list[dict]:
    if "readout" in spec:
        rows = json.loads((REPO / spec["readout"]).read_text())["rows"]
    else:
        rows = [json.loads(line) for line in (REPO / spec["trajectories"]).open()]
    if spec["branches"] is not None:
        rows = [r for r in rows if r.get("branch") in spec["branches"]]
    # A judge that failed to parse writes null, not NaN, and
    # `build_reduced_state_pairs` only tests for NaN -- a None would reach
    # float() and raise mid-fit. Coerce here so the drop happens where it is
    # counted (`_pairs_by_turn` refuses the whole trajectory) rather than
    # crashing or, worse, being silently re-indexed.
    for r in rows:
        if r.get(spec["y_col"]) is None:
            r[spec["y_col"]] = float("nan")
    return rows


def choose_nu(n_turns: int) -> tuple[int, int]:
    """Deepest y-block leaving >= MIN_HORIZON free steps; horizon capped at TARGET."""
    for nu in NU_CANDIDATES:
        horizon = min(TARGET_HORIZON, n_turns - nu)
        if horizon >= MIN_HORIZON:
            return nu, horizon
    raise ValueError(f"no nu leaves >= {MIN_HORIZON} steps at T={n_turns}")


class _NoControl:
    """Row 3: the same fit with the action withheld at fit AND at rollout."""

    def __init__(self, inner):
        self.inner = inner

    def step(self, z, v):
        return self.inner.step(z, np.zeros_like(np.asarray(v, dtype=float)))

    def readout(self, z):
        return self.inner.readout(z)


class _LSTMPrefix:
    """Row 4 wrapped so it is seeded from the same observed window as the rest."""

    def __init__(self, model, contemporaneous_v: bool):
        self.model, self.shift = model, 1 if contemporaneous_v else 0

    def warm(self, traj_rows, upto_index: int, y_col: str, u_col: str) -> np.ndarray:
        ys = [float(r[y_col]) for r in traj_rows[: upto_index + 1]]
        vs = [float(traj_rows[min(i + self.shift, len(traj_rows) - 1)][u_col]) for i in range(len(ys))]
        return self.model.warm_start(ys, vs)

    def step(self, z, v):
        return self.model.step(z, v)

    def readout(self, z):
        return self.model.readout(z)


def _pairs_by_turn(traj_rows, cfg, y_col, u_col):
    """{turn of z_t: pair}, or None if any readout was NaN.

    NaN turns make `build_reduced_state_pairs` drop pairs, which breaks the
    pair-index -> turn mapping every alignment here depends on. Refusing the
    trajectory is the honest response; the count of refusals is reported.
    """
    pairs = build_reduced_state_pairs(traj_rows, cfg, y_col=y_col, u_col=u_col)
    start = max(cfg.nu - 1, cfg.mu - (1 if cfg.contemporaneous_v else 0))
    if len(pairs) != max(0, len(traj_rows) - 1 - start):
        return None
    return {int(traj_rows[start + j]["turn"]): pair for j, pair in enumerate(pairs)}


def _rollout(predictor, traj_rows, window_cfg, model_cfg, y_col, u_col, horizon, lstm_prefix=False):
    """Per-row (turn, v, y_true, y_pred) over sliding windows.

    The WINDOWS come from `window_cfg` (the deepest state in the table) so
    every row scores every model on the identical set of turns; each model is
    then seeded from ITS OWN state at that same absolute turn. Without this
    the Markov row starts earlier than the delay-embedded one and the two are
    silently scored on different rows -- which is what this function's first
    version did, and what the caller's assertion caught.
    """
    window_pairs = _pairs_by_turn(traj_rows, window_cfg, y_col, u_col)
    model_pairs = window_pairs if model_cfg is window_cfg else _pairs_by_turn(traj_rows, model_cfg, y_col, u_col)
    if window_pairs is None or model_pairs is None:
        return []
    turns = sorted(window_pairs)
    index_of_turn = {int(r["turn"]): i for i, r in enumerate(traj_rows)}
    out = []
    for i in range(len(turns) - horizon + 1):
        t0 = turns[i]
        if t0 not in model_pairs:
            continue
        z = predictor.warm(traj_rows, index_of_turn[t0], y_col, u_col) if lstm_prefix else model_pairs[t0]["z"]
        for k in range(horizon):
            pair = window_pairs[turns[i + k]]
            z = predictor.step(z, pair["v"])
            out.append({
                "turn": int(turns[i + k]) + 1,
                "v": float(np.asarray(pair["v"], dtype=float).reshape(-1)[0]),
                "y_true": float(pair["z_next"][window_cfg.nu - 1]),
                "y_pred": float(predictor.readout(z)),
                "step": k,
            })
    return out


def _fit_rows(train_rows, config, markov_cfg, spec, seed, y_true_dim):
    """The five fitted rows on one fold's training items."""
    y_col, u_col = spec["y_col"], spec["u_col"]
    deep = build_identification_dataset(train_rows, config, y_col=y_col, u_col=u_col)
    flat = build_identification_dataset(train_rows, markov_cfg, y_col=y_col, u_col=u_col)

    ours = KoopmanSurrogate(extra_features_fn=no_extra_features).fit(deep)
    markov = KoopmanSurrogate(extra_features_fn=no_extra_features).fit(flat)
    blind_data = dict(deep, V=np.zeros_like(deep["V"]))
    blind = _NoControl(KoopmanSurrogate(extra_features_fn=no_extra_features).fit(blind_data))
    by_traj = list(group_by_trajectory(train_rows).values())
    cut = max(1, int(0.8 * len(by_traj)))
    # The AE gets a validation split carved out of the TRAINING items, the same
    # 80/20 slice the LSTM's hidden size is chosen on, and the same early-stop
    # protocol the core side runs. Without `val_dataset` this class's
    # `early_stopping_patience` is a no-op and it simply runs `num_epochs` --
    # which is how the first version of this script under-fit the AE and then
    # reported it as the worst row on `constraint`.
    val_rows = [r for rs in by_traj[cut:] for r in rs]
    ae_val = build_identification_dataset(val_rows, config, y_col=y_col, u_col=u_col) if val_rows else None
    ae = AEKoopmanSurrogate(
        state_dim=deep["Z"].shape[1], y_index=config.nu - 1,
        config=AEKoopmanConfig(latent_dim=4, num_epochs=3000, random_state=seed,
                               early_stopping_patience=30),
    ).fit(deep, val_dataset=ae_val)

    # Hidden size is selected on the held-out slice of the TRAINING items, never
    # on the fold's test items. Scored explicitly rather than read out of the
    # trainer's history dict, so the selection criterion is visible here.
    best, best_loss, best_hidden = None, float("inf"), LSTM_HIDDEN_CANDIDATES[0]
    for hidden in LSTM_HIDDEN_CANDIDATES:
        model, _ = train_lstm_surrogate(
            hidden, by_traj[:cut], by_traj[cut:], y_col=y_col, u_col=u_col,
            seed=seed, contemporaneous_v=config.contemporaneous_v)
        val = mse_from_predictions([
            rollout_predictions(model, rs, y_col=y_col, u_col=u_col,
                                contemporaneous_v=config.contemporaneous_v)
            for rs in by_traj[cut:]
        ])
        if val < best_loss:
            best, best_loss, best_hidden = model, val, hidden
    lstm = _LSTMPrefix(best, config.contemporaneous_v)
    return {
        "markov_linear_control": (markov, markov_cfg, False),
        "delay_linear_no_control": (blind, config, False),
        "lstm": (lstm, config, True),
        "ae_koopman": (ae, config, False),
        "delay_linear_control": (ours, config, False),
    }, {"lstm_hidden": best_hidden, "lstm_val_mse": best_loss}


def run_line(name: str, *, bootstrap_seed: int = 0) -> dict:
    spec = LINES[name]
    rows = load_rows(spec)
    y_col, u_col, item_col = spec["y_col"], spec["u_col"], spec["item_col"]
    by_traj = group_by_trajectory(rows)
    n_turns = max(len(r) for r in by_traj.values())
    # Drop whole trajectories whose readout has an unparsed turn, before any
    # consumer sees them. `_pairs_by_turn` would refuse them at rollout time,
    # but `train_lstm_surrogate` gets the raw rows: a NaN makes its loss NaN,
    # so it never improves and its `history[-1 - epochs_without_improvement]`
    # lookup runs off the end. Refusing here keeps every row every model is
    # fit on and scored on identical, and puts the count in the artifact
    # instead of losing it.
    refused = [tid for tid, rs in by_traj.items()
               if any(float(r[y_col]) != float(r[y_col]) for r in rs)]
    if refused:
        by_traj = {t: rs for t, rs in by_traj.items() if t not in set(refused)}
        rows = [r for rs in by_traj.values() for r in rs]
    nu, horizon = choose_nu(n_turns)
    config = ReducedStateConfig(nu=nu, mu=1, contemporaneous_v=True)
    markov_cfg = ReducedStateConfig(nu=1, mu=1, contemporaneous_v=True)

    traj_items = {tid: rs[0][item_col] for tid, rs in by_traj.items()}
    items = sorted(set(traj_items.values()))
    # Report folds, never the gate's: `constraint`'s one-step skill WAS the S3
    # admission gate, and a quantity used to select cannot also report.
    from surrogate_eval import make_folds
    folds = make_folds(items, N_FOLDS, purpose="report", seed=bootstrap_seed)

    collected: dict[str, list[dict]] = {}
    null_pred: dict[str, list[np.ndarray]] = {}
    meta_rows: list[dict] = []
    extras: list[dict] = []
    for fold in folds:
        test_items = set(fold["test_groups"].tolist())  # make_folds returns group VALUES, not indices
        train_rows = [r for r in rows if r[item_col] not in test_items]
        test_trajs = {t: rs for t, rs in by_traj.items() if traj_items[t] in test_items}
        fitted, extra = _fit_rows(train_rows, config, markov_cfg, spec, bootstrap_seed, 1)
        extras.append(extra)

        fold_meta: list[dict] | None = None
        for row_name, (model, cfg, prefix) in fitted.items():
            emitted = []
            for tid, traj_rows in test_trajs.items():
                for rec in _rollout(model, traj_rows, config, cfg, y_col, u_col, horizon, lstm_prefix=prefix):
                    rec = dict(rec, trajectory_id=tid, item=traj_items[tid])
                    emitted.append(rec)
            collected.setdefault(row_name, []).extend(emitted)
            if fold_meta is None:
                fold_meta = emitted
            else:
                assert [(r["trajectory_id"], r["turn"], r["step"]) for r in emitted] == \
                       [(r["trajectory_id"], r["turn"], r["step"]) for r in fold_meta], \
                       f"{row_name} scored on different rows"
        meta_rows.extend(fold_meta)

        train_panel = {"y_next": [], "turn_next": [], "v": []}
        for traj_rows in group_by_trajectory(train_rows).values():
            pairs = _pairs_by_turn(traj_rows, config, y_col, u_col)
            if pairs is None:
                continue
            for turn, pair in sorted(pairs.items()):
                train_panel["y_next"].append(float(pair["z_next"][config.nu - 1]))
                train_panel["v"].append(float(np.asarray(pair["v"]).reshape(-1)[0]))
                train_panel["turn_next"].append(float(turn + 1))
        train_panel = {k: np.asarray(v, dtype=float) for k, v in train_panel.items()}
        test_panel = {
            "y_next": np.asarray([r["y_true"] for r in fold_meta], dtype=float),
            "turn_next": np.asarray([r["turn"] for r in fold_meta], dtype=float),
            "v": np.asarray([r["v"] for r in fold_meta], dtype=float),
        }
        preds = trivial_nulls(train_panel, test_panel, exogenous=("turn_next", "v"))
        for k, v in preds.items():
            null_pred.setdefault(k, []).append(v)

    y_true = np.asarray([r["y_true"] for r in meta_rows], dtype=float)
    groups = np.asarray([r["item"] for r in meta_rows])
    null_name, y_null = best_null({k: np.concatenate(v) for k, v in null_pred.items()}, y_true)
    se_null = (y_null - y_true) ** 2

    def cell(se):
        out = {"rollout_mse": float(np.mean(se)), "null_mse": float(np.mean(se_null)), "n_rows": int(se.size)}
        try:
            out["skill_h"] = skill_from_squared_errors(se, se_null, horizon=horizon)
        except DegenerateNullError as exc:
            out["skill_h"] = None
            out["degenerate"] = str(exc)
            return out
        out["bootstrap"] = bootstrap_ci(
            np.column_stack([se, se_null]), groups, seed=bootstrap_seed,
            statistic=lambda v: 1.0 - float(np.mean(v[:, 0])) / float(np.mean(v[:, 1])))
        return out

    se = {k: np.asarray([r["y_pred"] for r in v], dtype=float) for k, v in collected.items()}
    se = {k: (v - y_true) ** 2 for k, v in se.items()}
    table = {"best_null": {"which": null_name, "rollout_mse": float(np.mean(se_null)),
                           "skill_h": 0.0, "n_rows": int(y_true.size)}}
    for k, v in se.items():
        table[k] = cell(v)

    contrasts = {}
    ours = "delay_linear_control"
    for k in se:
        if k != ours and table[k].get("skill_h") is not None and table[ours].get("skill_h") is not None:
            contrasts[f"{ours}_minus_{k}"] = bootstrap_ci(
                np.column_stack([se[ours], se[k], se_null]), groups, seed=bootstrap_seed,
                statistic=lambda v: (float(np.mean(v[:, 1])) - float(np.mean(v[:, 0]))) / float(np.mean(v[:, 2])))

    return {
        "line": name, "provenance": provenance(), "judge": spec["judge"], "note": spec["note"],
        "n_turns": n_turns, "nu": nu, "horizon": horizon, "mu": 1, "contemporaneous_v": True,
        "n_items": len(items), "n_trajectories": len(by_traj), "n_seeds": len({r.get("seed") for r in rows}),
        "n_scored_rows": int(y_true.size), "best_null_name": null_name,
        "n_trajectories_refused_unparsed_readout": len(refused),
        "rows": table, "contrasts": contrasts, "lstm_selection": extras,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lines", nargs="*", default=list(DEFAULT_LINES))
    parser.add_argument("--out-dir", type=pathlib.Path, default=REPO / "outputs" / "surrogate_rows_behavioral")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name in args.lines:
        out = args.out_dir / f"{name}.json"
        if out.exists():
            raise FileExistsError(f"{out} exists; outputs are never overwritten")
        result = run_line(name)
        out.write_text(json.dumps(result, indent=2, default=float) + "\n")
        skills = {k: v.get("skill_h") for k, v in result["rows"].items()}
        print(f"[done] {name} nu={result['nu']} H={result['horizon']} items={result['n_items']} "
              f"rows={result['n_scored_rows']} null={result['best_null_name']} " +
              " ".join(f"{k}={v:+.4f}" if isinstance(v, float) else f"{k}=NA" for k, v in skills.items()), flush=True)


if __name__ == "__main__":
    main()
