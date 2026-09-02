"""Shared data loading and reduced-state construction for every surrogate
model fit on `trajectories.jsonl` -- the Koopman surrogate, the ARX baseline,
and any future baseline (LSTM, ...). Every one of them MUST go through this
module rather than re-deriving z_t/v_t/y_t or the train/val/test split
itself: if two methods preprocessed the data even slightly differently, a
measured quality difference could come from the preprocessing instead of the
model, which would make the comparison meaningless.

Deliberately NOT a thin wrapper around `koopman_ae.core`'s Model III dataset
builder (`build_augmented_state_dataset`): that one hardcodes the control
input as the tracking error `u_t = r - y_t` (closed-loop, `control_mode:
error`/`error_abs_sign`). DATA_COLLECTION_PROTOCOL.md's whole premise is that
persona-drift's `u_remind` is an independently, randomly excited open-loop
input with `r` absent from the prompt entirely during collection -- exactly
the r/u conflation `docs/references/Control_of_Foundational_Model_revised.pdf`
argues against. There is no `r` to compute an error against here, so that
dataset builder does not apply; this module is the open-loop-input
equivalent, built for `y_probe`/`u_remind` instead of `normalized_output`/
`effective_norm`.
"""

from __future__ import annotations

import json
import pathlib
import random
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ReducedStateConfig:
    """z_t = [y_t, ..., y_(t-nu+1), v_(t-1), ..., v_(t-mu), aux_1(t), ...,
    aux_k(t)] -- the first two blocks are PDF eq. (8) (`nu` past outputs
    including the current one, `mu` past inputs NOT including the current
    one, since v_t enters the transition separately per eq. (15)). `aux_cols`
    is an optional extension (docs/experiments/koopman_detection_design.md
    option 4): each name is an extra per-turn scalar column (e.g. a
    content-derived feature computed outside this module, see
    modeling.content_similarity) appended to z as its CURRENT-turn value
    only, no lag history of its own -- `z_next`'s aux block is the same
    column's value one turn later, so it participates in the fit like any
    other state component (predictable or not; a column the linear model
    can't predict just contributes noise in that dimension, it doesn't break
    anything). Defaults to `()` so every existing caller is unaffected.

    `contemporaneous_v` (default `False`, preserves all existing behavior):
    the PDF's eq. (8)/(15) convention pairs `v_t` with `z_t` (which already
    contains `y_t`) to predict `z_(t+1)`, i.e. it assumes the actuator's
    effect on `y` shows up one step LATER than the action. That matches
    persona-drift's original probe-then-decide loop, but not
    attack_trajectory.py's adversarial-defense timing, where `u_remind` for
    turn `t` is inserted BEFORE that same turn's reply is generated and
    scored -- `u_t` acts on `y_t` in the SAME turn, and `y_t` is already
    inside `z_t` by the time `v_t` gets fit. Fit that way, the learned `B`
    coefficient is not the reminder's direct effect; it is only the
    residual/carryover effect of turn `t`'s reminder text still sitting in
    the chat history one turn later, on `y_(t+1)`. See
    docs/experiments/koopman_defense_pilot.md's "错位" analysis (2026-09-02)
    for the derivation and the anomalies it explains (sign flip between
    mu=1/mu=2, the state-independent constant marginal-value channel,
    interaction term's reversed-looking direction).

    `contemporaneous_v=True` shifts every `v`-indexed slot (the lag history
    AND the free control input) forward by one turn relative to `z_t`'s `y`
    block, so the pair becomes `(z_t, v=u_(t+1)) -> z_(t+1)` with
    `z_(t+1)`'s `y` component `y_(t+1)` directly caused by that same
    `u_(t+1)` -- i.e. `z_t`'s `mu`-lag block now runs up to and including
    `u_t` (already realized by the time `z_t` is known) instead of stopping
    at `u_(t-1)`. `control.py::KoopmanMPCController._current_state` mirrors
    this shift so a controller built on a `contemporaneous_v=True` surrogate
    evaluates its candidate action in the same slot the surrogate was fit
    on."""

    nu: int = 1
    mu: int = 1
    aux_cols: tuple[str, ...] = ()
    contemporaneous_v: bool = False

    def __post_init__(self) -> None:
        if self.nu < 1:
            raise ValueError("nu must be >= 1 (z_t must contain the current y_t)")
        if self.mu < 0:
            raise ValueError("mu must be >= 0")

    @property
    def state_dim(self) -> int:
        return self.nu + self.mu + len(self.aux_cols)


def load_trajectories(path: str | pathlib.Path) -> list[dict]:
    rows = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def group_by_trajectory(rows: list[dict], id_col: str = "trajectory_id") -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row[id_col], []).append(row)
    for traj_rows in groups.values():
        traj_rows.sort(key=lambda r: r["turn"])
    return groups


def split_by_system_prompt_id(
    rows: list[dict],
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int = 0,
    split_col: str = "system_prompt_id",
) -> dict[str, list[dict]]:
    """Rollout-level split (by `split_col`, not by individual turn):
    DATA_COLLECTION_PROTOCOL.md section 6 and the PDF's validation protocol
    (section 5/8) both require this, because adjacent turns within one
    dialogue are dependent -- a turn-level random split would leak.

    `split_col` defaults to `system_prompt_id` (persona-drift's rollout
    grouping key); other domains pass their own grouping key -- e.g. the
    adversarial-defense domain's `attack_id`, so that a fixed attack's
    different seeds always land in the same split -- see
    docs/experiments/koopman_defense_pilot.md."""

    prompt_ids = sorted({row[split_col] for row in rows})
    rng = random.Random(seed)
    rng.shuffle(prompt_ids)
    n = len(prompt_ids)
    n_train = round(n * train_frac)
    n_val = round(n * val_frac)
    assignment = {
        pid: ("train" if i < n_train else "val" if i < n_train + n_val else "test")
        for i, pid in enumerate(prompt_ids)
    }
    split: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for row in rows:
        split[assignment[row[split_col]]].append(row)
    return split


def build_reduced_state_pairs(
    traj_rows: list[dict], config: ReducedStateConfig, y_col: str = "y_probe", u_col: str = "u_remind"
) -> list[dict]:
    """One trajectory's rows (sorted by turn) -> transition pairs
    {z, v, y, z_next}. Turns with a NaN readout (scorer failure, see
    prompt_bank.score_response) are dropped from every pair that would
    otherwise include them, rather than propagating NaN into a fit.

    `y_col`/`u_col` default to persona-drift's `y_probe`/`u_remind`; other
    domains (e.g. adversarial-defense's `y_safety`) pass their own column
    names instead of renaming their data to fit this module's original
    schema -- see docs/experiments/koopman_defense_pilot.md."""

    nu, mu = config.nu, config.mu
    shift = 1 if config.contemporaneous_v else 0
    ys = [row[y_col] for row in traj_rows]
    vs = [float(row[u_col]) for row in traj_rows]
    aux_series = [[float(row[col]) for row in traj_rows] for col in config.aux_cols]
    pairs: list[dict] = []
    start = max(nu - 1, mu - shift)
    for t in range(start, len(traj_rows) - 1):
        tv = t + shift
        y_hist = ys[t - nu + 1 : t + 1]
        v_hist = vs[tv - mu : tv]
        y_hist_next = ys[t - nu + 2 : t + 2]
        v_hist_next = vs[tv - mu + 1 : tv + 1]
        aux_now = [series[t] for series in aux_series]
        aux_next = [series[t + 1] for series in aux_series]
        values = y_hist + v_hist + [vs[tv]] + y_hist_next + v_hist_next + aux_now + aux_next
        if any(value != value for value in values):  # NaN check without numpy
            continue
        pairs.append(
            {
                "z": np.array(y_hist + v_hist + aux_now, dtype=float),
                "v": np.array([vs[tv]], dtype=float),
                "y": float(ys[t]),
                "z_next": np.array(y_hist_next + v_hist_next + aux_next, dtype=float),
            }
        )
    return pairs


def build_identification_dataset(
    rows: list[dict],
    config: ReducedStateConfig,
    id_col: str = "trajectory_id",
    y_col: str = "y_probe",
    u_col: str = "u_remind",
) -> dict[str, np.ndarray]:
    """`rows` (any split, any mix of trajectories) -> stacked arrays Z, V,
    Z_next, Y, ready for `modeling.koopman.KoopmanSurrogate.fit` or any other
    predictor built on the same state representation. Groups by `id_col`
    first so state history never leaks across trajectories.

    `id_col`/`y_col`/`u_col` default to persona-drift's column names; other
    domains pass their own (e.g. adversarial-defense: unchanged
    `trajectory_id`, but `y_safety` instead of `y_probe`) -- see
    docs/experiments/koopman_defense_pilot.md."""

    state_dim = config.state_dim
    Z, V, Z_next, Y = [], [], [], []
    for traj_rows in group_by_trajectory(rows, id_col=id_col).values():
        for pair in build_reduced_state_pairs(traj_rows, config, y_col=y_col, u_col=u_col):
            Z.append(pair["z"])
            V.append(pair["v"])
            Z_next.append(pair["z_next"])
            Y.append(pair["y"])
    return {
        "Z": np.stack(Z) if Z else np.zeros((0, state_dim)),
        "V": np.stack(V) if V else np.zeros((0, 1)),
        "Z_next": np.stack(Z_next) if Z_next else np.zeros((0, state_dim)),
        "Y": np.array(Y, dtype=float),
    }
