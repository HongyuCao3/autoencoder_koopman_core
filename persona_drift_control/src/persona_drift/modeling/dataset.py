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
    """z_t = [y_t, ..., y_(t-nu+1), v_(t-1), ..., v_(t-mu)] per
    Control_of_Foundational_Model_revised.pdf eq. (8): `nu` past outputs
    (including the current one) and `mu` past inputs (NOT including the
    current one -- v_t enters the transition separately, eq. (15))."""

    nu: int = 1
    mu: int = 1

    def __post_init__(self) -> None:
        if self.nu < 1:
            raise ValueError("nu must be >= 1 (z_t must contain the current y_t)")
        if self.mu < 0:
            raise ValueError("mu must be >= 0")

    @property
    def state_dim(self) -> int:
        return self.nu + self.mu


def load_trajectories(path: str | pathlib.Path) -> list[dict]:
    rows = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def group_by_trajectory(rows: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["trajectory_id"], []).append(row)
    for traj_rows in groups.values():
        traj_rows.sort(key=lambda r: r["turn"])
    return groups


def split_by_system_prompt_id(
    rows: list[dict], train_frac: float = 0.7, val_frac: float = 0.15, seed: int = 0
) -> dict[str, list[dict]]:
    """Rollout-level split (by `system_prompt_id`, not by individual turn):
    DATA_COLLECTION_PROTOCOL.md section 6 and the PDF's validation protocol
    (section 5/8) both require this, because adjacent turns within one
    dialogue are dependent -- a turn-level random split would leak."""

    prompt_ids = sorted({row["system_prompt_id"] for row in rows})
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
        split[assignment[row["system_prompt_id"]]].append(row)
    return split


def build_reduced_state_pairs(traj_rows: list[dict], config: ReducedStateConfig) -> list[dict]:
    """One trajectory's rows (sorted by turn) -> transition pairs
    {z, v, y, z_next}. Turns with a NaN y_probe (scorer failure, see
    prompt_bank.score_response) are dropped from every pair that would
    otherwise include them, rather than propagating NaN into a fit."""

    nu, mu = config.nu, config.mu
    ys = [row["y_probe"] for row in traj_rows]
    vs = [float(row["u_remind"]) for row in traj_rows]
    pairs: list[dict] = []
    start = max(nu - 1, mu)
    for t in range(start, len(traj_rows) - 1):
        y_hist = ys[t - nu + 1 : t + 1]
        v_hist = vs[t - mu : t]
        y_hist_next = ys[t - nu + 2 : t + 2]
        v_hist_next = vs[t - mu + 1 : t + 1]
        values = y_hist + v_hist + [vs[t]] + y_hist_next + v_hist_next
        if any(value != value for value in values):  # NaN check without numpy
            continue
        pairs.append(
            {
                "z": np.array(y_hist + v_hist, dtype=float),
                "v": np.array([vs[t]], dtype=float),
                "y": float(ys[t]),
                "z_next": np.array(y_hist_next + v_hist_next, dtype=float),
            }
        )
    return pairs


def build_identification_dataset(rows: list[dict], config: ReducedStateConfig) -> dict[str, np.ndarray]:
    """`rows` (any split, any mix of trajectories) -> stacked arrays Z, V,
    Z_next, Y, ready for `modeling.koopman.KoopmanSurrogate.fit` or any other
    predictor built on the same state representation. Groups by
    `trajectory_id` first so state history never leaks across trajectories."""

    state_dim = config.state_dim
    Z, V, Z_next, Y = [], [], [], []
    for traj_rows in group_by_trajectory(rows).values():
        for pair in build_reduced_state_pairs(traj_rows, config):
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
