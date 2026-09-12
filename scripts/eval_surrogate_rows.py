#!/usr/bin/env python3
"""Table 1's six rows, on the eight core trajectory tasks.

Every cell in the paper's main table has to come out of one code path, or the
ten datasets are not comparable across columns. That path is
`src/surrogate_eval` (skill, nulls, bootstrap, folds); this script is the core
side's caller. `persona_drift_control` has its own caller for the three
behavioural datasets and imports the same harness.

LAG IS PER DATASET, AND WHY. `common_seed_turns=4` with `lag=3` leaves a T=5
task exactly one rollout step -- so five of the eight tasks' published
"rollout_mse" is a one-step number under a rollout name, and ruling 2
(2026-09-12) admits only multi-step skill to the main table. Each task
therefore gets the deepest delay embedding that still leaves H >= 3: lag=3 for
the T=10 tasks (H=6, the repository default, where the published numbers
live), lag=1 for the T=5 tasks (H=3). The table prints lag and H per column.

Every row is rolled out from the SAME `observed_seed_turns`, which is what
that argument exists for: the Markov row has a shorter memory than the others
and would otherwise start earlier and be scored on a different row set.

Additive: does not touch `core.py`, `scripts/train.py`, or any existing
ablation script. Writes to `results/surrogate_rows/`, which is new.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn

from koopman_ae import (
    AugmentedKoopmanModel,
    AugmentedStateConfig,
    DeepAugmentedKoopmanAutoencoder,
    DeepAugmentedKoopmanConfig,
    build_augmented_state_dataset,
    rollout_augmented_from_trajectories,
)
from surrogate_eval import (
    best_null,
    bootstrap_ci,
    seed_aggregate,
    skill_from_squared_errors,
    trivial_nulls,
)
from surrogate_eval.skill import DegenerateNullError

PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[1]
TASKS = [
    "sentence_length_t10",
    "character_length_t5",
    "formality_t5",
    "average_word_length_t5",
    "sentiment_t5",
    "even_odd_t5",
    "vector_count_stage1_t10",
    "vector_count_stage2_t10",
]
SEEDS = (0, 1, 2)
LSTM_HIDDEN_CANDIDATES = (8, 16, 32)
MIN_HORIZON = 3


def provenance() -> dict:
    """Which harness version produced this artifact.

    `.claude/global.md` requires every artifact to be able to answer that;
    without it a harness change forces a full refit because nothing says which
    outputs went stale. Raises rather than writing `git_sha: null` -- a missing
    fingerprint that looks like a present one is worse than a failed run. The
    `constraint` line's `run_provenance.py` takes the same line; this is the
    core side's copy because the two code bases do not import each other.
    """
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(PACKAGE_ROOT), *args],
            check=True, capture_output=True, text=True,
        ).stdout.strip()

    dirty = git("status", "--porcelain")
    return {
        "git_sha": git("rev-parse", "HEAD"),
        "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(dirty),
        "n_dirty_paths": len(dirty.splitlines()),
        "seeds": list(SEEDS),
        "lstm_hidden_candidates": list(LSTM_HIDDEN_CANDIDATES),
        "min_horizon": MIN_HORIZON,
    }


def choose_lag(n_turns: int) -> int:
    """Deepest lag whose rollout still has >= MIN_HORIZON steps."""
    for lag in (3, 1):
        if n_turns - (lag + 1) >= MIN_HORIZON:
            return lag
    raise ValueError(f"no lag leaves >= {MIN_HORIZON} rollout steps at T={n_turns}")


def _split(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    return frame[frame["topic_split"].eq(name)].copy()


def _load(task: str) -> tuple[pd.DataFrame, dict]:
    cfg = yaml.safe_load((PACKAGE_ROOT / "configs" / "dataset" / f"{task}.yaml").read_text())
    frame = pd.read_json(PACKAGE_ROOT / "datasets" / cfg["path"], lines=True)
    frame["turn"] = pd.to_numeric(frame["turn"], errors="coerce").astype(int)
    return frame.sort_values(["trajectory_id", "turn"]).reset_index(drop=True), cfg


def _stacked(rollout: pd.DataFrame, column: str) -> np.ndarray:
    """Rollout rows -> (n_rows, output_dim); scalar tasks store floats, vector
    tasks store lists, and both have to flatten the same way."""
    values = rollout[column].tolist()
    return np.asarray([np.atleast_1d(np.asarray(v, dtype=float)) for v in values], dtype=float)


def _targets(frame: pd.DataFrame, target_columns) -> np.ndarray:
    return frame[list(target_columns)].to_numpy(dtype=float)


class _NoControlModel:
    """Row 3: the same affine Koopman fit, with the reference withheld.

    Fit and predict both see r = 0, so `B` cannot carry anything and the model
    is z_(t+1) = A z_t + c. Withholding the information is the point of the
    row: it is what separates "the operator needs the actuator" from "the
    operator is a decay curve".
    """

    name = "delay_linear_no_control"

    def __init__(self, inner: AugmentedKoopmanModel):
        self.inner = inner

    def predict_next_z(self, z_t, r):
        return self.inner.predict_next_z(z_t, np.zeros_like(np.asarray(r, dtype=float)))

    def predict_y(self, z_t):
        return self.inner.predict_y(z_t)


class _LSTM(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden, out_dim)

    def forward(self, x, state=None):
        out, state = self.lstm(x, state)
        return self.head(out), state


def _sequences(frame: pd.DataFrame, output_columns, target_columns):
    """One (y_seq, r) pair per trajectory, turns in order."""
    seqs = []
    for _, group in frame.groupby("trajectory_id", sort=False):
        group = group.sort_values("turn")
        seqs.append(
            (
                group[list(output_columns)].to_numpy(dtype=float),
                group[list(target_columns)].iloc[0].to_numpy(dtype=float),
                group["turn"].to_numpy(dtype=int),
                str(group["trajectory_id"].iloc[0]),
            )
        )
    return seqs


def _lstm_batch(seqs, device="cpu"):
    ys = torch.tensor(np.stack([s[0] for s in seqs]), dtype=torch.float32, device=device)
    rs = torch.tensor(np.stack([s[1] for s in seqs]), dtype=torch.float32, device=device)
    rs = rs.unsqueeze(1).expand(-1, ys.shape[1], -1)
    return torch.cat([ys, rs], dim=-1), ys


def fit_lstm(train_seqs, val_seqs, out_dim: int, hidden: int, seed: int, max_epochs=4000, patience=200):
    torch.manual_seed(seed)
    x_tr, y_tr = _lstm_batch(train_seqs)
    x_va, y_va = _lstm_batch(val_seqs)
    model = _LSTM(x_tr.shape[-1], out_dim, hidden)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_loss, best_state, waited = float("inf"), None, 0
    for _ in range(max_epochs):
        model.train()
        opt.zero_grad()
        pred, _ = model(x_tr[:, :-1, :])
        loss = nn.functional.mse_loss(pred, y_tr[:, 1:, :])
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            val_pred, _ = model(x_va[:, :-1, :])
            val_loss = float(nn.functional.mse_loss(val_pred, y_va[:, 1:, :]))
        if val_loss < best_loss - 1e-6:
            best_loss, waited = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
            if waited >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_loss


def lstm_rollout(model, seqs, seed_turn: int):
    """Autoregressive rollout after an observed prefix, matching the
    `observed_seed_turns` contract the Koopman rollout uses.

    Returns the predictions and the (trajectory_id, turn) they belong to, so
    the caller can assert they line up with the Koopman rollout's rows instead
    of trusting two different groupby orders to agree.
    """
    model.eval()
    preds, index = [], []
    with torch.no_grad():
        for y_seq, r, turns, traj_id in seqs:
            if int(turns.max()) <= seed_turn:
                continue
            n_obs = int(np.sum(turns <= seed_turn))
            x = torch.tensor(
                np.concatenate([y_seq[:n_obs], np.tile(r, (n_obs, 1))], axis=-1),
                dtype=torch.float32,
            ).unsqueeze(0)
            out, state = model(x)
            y_hat = out[:, -1, :]
            for step in range(len(turns) - n_obs):
                preds.append(y_hat.squeeze(0).numpy().copy())
                index.append((traj_id, int(turns[n_obs + step])))
                nxt = torch.cat([y_hat, torch.tensor(r, dtype=torch.float32).unsqueeze(0)], dim=-1)
                out, state = model(nxt.unsqueeze(1), state)
                y_hat = out[:, -1, :]
    return np.asarray(preds, dtype=float), index


def _null_panels(train_frame, test_rollout, output_columns, target_columns, seed_turn):
    """Fit panels for the three trivial nulls, on the rows the models are scored on.

    All three nulls are functions of exogenous quantities only, so they are
    horizon-agnostic: the prediction for turn t is the same whether it is one
    step out or six. `exogenous` is (turn, r) -- NOT the control. In this code
    base the control is the tracking error r - y_t, which contains the state;
    handing it to a "stateless" null would leak exactly what the null exists to
    withhold. r (`effective_norm`) is fixed per trajectory in advance and is
    the genuinely exogenous input.
    """
    train_rows = train_frame[train_frame["turn"] > seed_turn]
    r_train = _targets(train_rows, target_columns)
    r_test = np.asarray(
        [np.atleast_1d(np.asarray(v, dtype=float)) for v in test_rollout["target_effective_norm"]],
        dtype=float,
    )
    exogenous = ("turn_next", *(f"r_{i}" for i in range(r_train.shape[1])))
    y_train = train_rows[list(output_columns)].to_numpy(dtype=float)
    y_test = _stacked(test_rollout, "y_true")

    per_dim = []
    for j in range(y_train.shape[1]):
        train_panel = {"y_next": y_train[:, j], "turn_next": train_rows["turn"].to_numpy(dtype=float)}
        test_panel = {"y_next": y_test[:, j], "turn_next": test_rollout["turn"].to_numpy(dtype=float)}
        for i in range(r_train.shape[1]):
            train_panel[f"r_{i}"] = r_train[:, i]
            test_panel[f"r_{i}"] = r_test[:, i]
        per_dim.append(trivial_nulls(train_panel, test_panel, exogenous=exogenous))
    names = sorted(per_dim[0])
    return {name: np.stack([d[name] for d in per_dim], axis=1) for name in names}


def _cell(se_model: np.ndarray, se_null: np.ndarray, groups, horizon: int, seed: int) -> dict:
    """One table cell: skill, a by-trajectory bootstrap CI, and the raw MSEs.

    Every row gets the same interval. The first version of this script gave
    the deterministic rows a bootstrap CI and the seeded rows a standard
    deviation over three training seeds -- two different quantities printed in
    one column, which made the AE look a hundred times more precise than the
    linear fit next to it.
    """
    cell = {
        "rollout_mse": float(np.mean(se_model)),
        "null_mse": float(np.mean(se_null)),
        "n_rows": int(se_model.size),
    }
    try:
        cell["skill_h"] = skill_from_squared_errors(se_model, se_null, horizon=horizon)
    except DegenerateNullError as exc:
        cell["skill_h"] = None
        cell["degenerate"] = str(exc)
        return cell
    cell["bootstrap"] = bootstrap_ci(
        np.column_stack([se_model, se_null]),
        groups,
        seed=seed,
        statistic=lambda v: 1.0 - float(np.mean(v[:, 0])) / float(np.mean(v[:, 1])),
    )
    return cell


def _contrast(se_ours, se_other, se_null, groups, seed: int) -> dict:
    """Paired bootstrap of `skill(ours) - skill(other)`, resampling trajectories.

    Two overlapping per-cell CIs are not evidence that two rows are the same,
    and "the nonlinear lift buys nothing" is exactly a claim about a
    difference. Resampled jointly, so the trajectories that are hard for one
    row are hard for the other in the same draw.
    """
    return bootstrap_ci(
        np.column_stack([se_ours, se_other, se_null]),
        groups,
        seed=seed,
        statistic=lambda v: (float(np.mean(v[:, 1])) - float(np.mean(v[:, 0]))) / float(np.mean(v[:, 2])),
    )


def run_task(task: str, *, bootstrap_seed: int = 0, max_epochs: int = 3000) -> dict:
    frame, cfg = _load(task)
    return run_task_frame(
        frame, tuple(cfg["output_columns"]), tuple(cfg["target_columns"]), task,
        bootstrap_seed=bootstrap_seed, max_epochs=max_epochs,
    )


def run_task_frame(
    frame: pd.DataFrame,
    output_columns,
    target_columns,
    task: str,
    *,
    bootstrap_seed: int = 0,
    max_epochs: int = 3000,
    families: tuple[str, ...] = ("ae_koopman", "lstm"),
) -> dict:
    """The six rows on one already-loaded frame.

    Split out from `run_task` so the tests can plant a system with a known
    answer instead of asserting against a dataset file.
    """
    output_columns, target_columns = tuple(output_columns), tuple(target_columns)
    n_turns = int(frame["turn"].max())
    lag = choose_lag(n_turns)
    seed_turn = lag + 1
    horizon = n_turns - seed_turn

    deep_cfg = AugmentedStateConfig(
        output_memory=lag + 1, input_memory=0, control_mode="error",
        output_columns=output_columns, target_columns=target_columns,
    )
    markov_cfg = AugmentedStateConfig(
        output_memory=1, input_memory=0, control_mode="error",
        output_columns=output_columns, target_columns=target_columns,
    )

    train_frame, val_frame, test_frame = (_split(frame, s) for s in ("train", "validation", "test"))
    deep_train = build_augmented_state_dataset(train_frame, deep_cfg)
    markov_train = build_augmented_state_dataset(train_frame, markov_cfg)

    fitted = {}
    ours = AugmentedKoopmanModel(output_dim=deep_train.output_dim, alpha=1e-6)
    ours.name = "delay_linear_control"
    fitted["delay_linear_control"] = (
        ours.fit(deep_train.Z_t, deep_train.R, deep_train.Z_next), deep_cfg)

    markov = AugmentedKoopmanModel(output_dim=markov_train.output_dim, alpha=1e-6)
    markov.name = "markov_linear_control"
    fitted["markov_linear_control"] = (
        markov.fit(markov_train.Z_t, markov_train.R, markov_train.Z_next), markov_cfg)

    blind = AugmentedKoopmanModel(output_dim=deep_train.output_dim, alpha=1e-6)
    blind.fit(deep_train.Z_t, np.zeros_like(deep_train.R), deep_train.Z_next)
    fitted["delay_linear_no_control"] = (_NoControlModel(blind), deep_cfg)

    reference = rollout_augmented_from_trajectories(
        ours, test_frame, deep_cfg, observed_seed_turns=seed_turn)
    reference = reference[~reference["uses_observed_seed"]].reset_index(drop=True)
    reference_keys = list(zip(reference["trajectory_id"].astype(str), reference["turn"].astype(int)))
    y_true = _stacked(reference, "y_true")
    groups = np.repeat(reference["trajectory_id"].to_numpy(), y_true.shape[1])

    nulls = _null_panels(train_frame, reference, output_columns, target_columns, seed_turn)
    null_name, y_null = best_null(
        {k: v.reshape(-1) for k, v in nulls.items()}, y_true.reshape(-1))
    y_null = y_null.reshape(y_true.shape)

    se_null = ((y_null - y_true) ** 2).reshape(-1)
    squared_error: dict[str, np.ndarray] = {}
    predictions: dict[str, np.ndarray] = {"y_true": y_true, "y_null": y_null}
    rows: dict[str, dict] = {
        "best_null": {
            "which": null_name,
            "rollout_mse": float(np.mean(se_null)),
            "skill_h": 0.0,
            "n_rows": int(y_true.size),
        }
    }
    for name, (model, model_cfg) in fitted.items():
        pred = rollout_augmented_from_trajectories(
            model, test_frame, model_cfg, observed_seed_turns=seed_turn)
        pred = pred[~pred["uses_observed_seed"]].reset_index(drop=True)
        keys = list(zip(pred["trajectory_id"].astype(str), pred["turn"].astype(int)))
        assert keys == reference_keys, f"{name} scored on different rows than the reference rollout"
        y_pred = _stacked(pred, "y_pred")
        predictions[name] = y_pred
        squared_error[name] = ((y_pred - y_true) ** 2).reshape(-1)
        rows[name] = _cell(squared_error[name], se_null, groups, horizon, bootstrap_seed)

    train_seqs = _sequences(train_frame, output_columns, target_columns)
    val_seqs = _sequences(val_frame, output_columns, target_columns)
    test_seqs = _sequences(test_frame, output_columns, target_columns)
    for family in families:
        per_seed_cells, per_seed_se, per_seed_pred = [], [], []
        for seed in SEEDS:
            if family == "ae_koopman":
                deep_val = build_augmented_state_dataset(val_frame, deep_cfg)
                ae = DeepAugmentedKoopmanAutoencoder(
                    state_dim=deep_train.state_dim,
                    target_dim=deep_train.target_dim,
                    output_dim=deep_train.output_dim,
                    config=DeepAugmentedKoopmanConfig(
                        latent_dim=16, num_epochs=max_epochs, random_state=seed, device="cpu",
                        training_mode="reconstruction_then_ridge",
                        early_stopping_patience=30, early_stopping_min_delta=1e-6,
                    ),
                    name="ae_koopman",
                )
                ae.fit(
                    deep_train.Z_t, deep_train.R, deep_train.Z_next,
                    Z_val=deep_val.Z_t, R_val=deep_val.R, Z_next_val=deep_val.Z_next,
                )
                pred_frame = rollout_augmented_from_trajectories(
                    ae, test_frame, deep_cfg, observed_seed_turns=seed_turn)
                pred_frame = pred_frame[~pred_frame["uses_observed_seed"]].reset_index(drop=True)
                y_pred = _stacked(pred_frame, "y_pred")
                extra = {}
            else:
                chosen, best_val = None, float("inf")
                for hidden in LSTM_HIDDEN_CANDIDATES:
                    model, val_loss = fit_lstm(train_seqs, val_seqs, y_true.shape[1], hidden, seed)
                    if val_loss < best_val:
                        chosen, best_val, chosen_hidden = model, val_loss, hidden
                y_pred, lstm_index = lstm_rollout(chosen, test_seqs, seed_turn)
                # Align by key, never by position: `rollout_augmented_from_trajectories`
                # sorts with pandas' default (unstable) quicksort, so the
                # trajectory order it emits is not the frame's order.
                by_key = dict(zip(lstm_index, y_pred))
                if set(by_key) != set(reference_keys):
                    raise AssertionError("LSTM rollout covers different rows than the Koopman rollout")
                y_pred = np.asarray([by_key[k] for k in reference_keys], dtype=float)
                extra = {"hidden": chosen_hidden, "validation_mse": best_val}
            se_seed = ((y_pred - y_true) ** 2).reshape(-1)
            per_seed_se.append(se_seed)
            per_seed_pred.append(y_pred)
            cell = _cell(se_seed, se_null, groups, horizon, bootstrap_seed)
            cell.update(extra)
            cell["seed"] = seed
            per_seed_cells.append(cell)
        # The row's point estimate and CI come from the seed-averaged squared
        # error, so both describe the same quantity; the spread across seeds is
        # reported separately rather than doubling as the error bar.
        squared_error[family] = np.mean(np.stack(per_seed_se), axis=0)
        predictions[family] = np.stack(per_seed_pred)
        rows[family] = _cell(squared_error[family], se_null, groups, horizon, bootstrap_seed)
        rows[family]["per_seed"] = per_seed_cells
        skills = [c["skill_h"] for c in per_seed_cells]
        if all(v is not None for v in skills):
            # Both conventions are stored: the repository has not signed which
            # one the paper uses (MAIN_TABLE_DESIGN.md section 6 item 5).
            rows[family]["seed_spread_ddof0"] = seed_aggregate(skills, ddof=0)
            rows[family]["seed_spread_ddof1"] = seed_aggregate(skills, ddof=1)

    ours = "delay_linear_control"
    contrasts = {}
    if rows[ours].get("skill_h") is not None:
        for name in squared_error:
            if name != ours and rows[name].get("skill_h") is not None:
                contrasts[f"{ours}_minus_{name}"] = _contrast(
                    squared_error[ours], squared_error[name], se_null, groups, bootstrap_seed)

    return {
        "task": task,
        "provenance": provenance(),
        "n_turns": n_turns,
        "lag": lag,
        "observed_seed_turns": seed_turn,
        "horizon": horizon,
        "output_dim": int(y_true.shape[1]),
        "n_test_trajectories": int(reference["trajectory_id"].nunique()),
        "n_scored_rows": int(y_true.size),
        "best_null_name": null_name,
        "rows": rows,
        "contrasts": contrasts,
        "_predictions": predictions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="*", default=TASKS)
    parser.add_argument("--out-dir", type=pathlib.Path,
                        default=PACKAGE_ROOT / "results" / "surrogate_rows")
    parser.add_argument("--max-epochs", type=int, default=3000)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for task in args.tasks:
        out_path = args.out_dir / f"{task}.json"
        if out_path.exists():
            raise FileExistsError(f"{out_path} exists; outputs are never overwritten")
        result = run_task(task, max_epochs=args.max_epochs)
        # Keeping every row's predictions means a later contrast never needs
        # the two-hour refit again; the JSON stays readable.
        np.savez_compressed(args.out_dir / f"{task}_predictions.npz", **result.pop("_predictions"))
        out_path.write_text(json.dumps(result, indent=2, default=float) + "\n")
        summary[task] = result
        skills = {k: v.get("skill_h") for k, v in result["rows"].items()}
        print(f"[done] {task} lag={result['lag']} H={result['horizon']} "
              f"null={result['best_null_name']} " +
              " ".join(f"{k}={v:+.4f}" if isinstance(v, float) else f"{k}=NA"
                       for k, v in skills.items()), flush=True)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")


if __name__ == "__main__":
    main()
