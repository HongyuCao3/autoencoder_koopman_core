"""Shared evaluation harness for any surrogate predictor built on
modeling.dataset's z_t/v_t/y_t representation. Used identically for the
Koopman surrogate, the ARX baseline (koopman.KoopmanSurrogate with
extra_features_fn=no_extra_features), and any future baseline (e.g. an
LSTM) that implements the `Predictor` protocol below -- a quality difference
between methods should never be attributable to the evaluation code itself.

Implements the first two checks of
Control_of_Foundational_Model_revised.pdf section 8's validation protocol:
one-step prediction and multi-step rollout. Reachable-set / controllable-set
agreement need real collected data to compare against and are not
implemented yet (see docs/BASELINES.md's GenCtrl entry for the intended
black-box counterpart).
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .dataset import ReducedStateConfig, build_reduced_state_pairs, group_by_trajectory


class Predictor(Protocol):
    def step(self, z: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Predict z_(t+1) from the current reduced state and input."""
        ...

    def readout(self, z: np.ndarray) -> float:
        """Predict y_t from the reduced state z_t."""
        ...


def one_step_error(predictor: Predictor, dataset: dict[str, np.ndarray]) -> float:
    """Mean squared error of z_(t+1) predictions on held-out (z, v, z_next)
    triples -- PDF section 8, "one-step prediction". Pass a dataset built
    from a held-out split (e.g. dataset.split_by_system_prompt_id's "test")
    for this to mean anything."""

    Z, V, Z_next = dataset["Z"], dataset["V"], dataset["Z_next"]
    if Z.shape[0] == 0:
        return float("nan")
    errors = [
        float(np.sum((predictor.step(z, v) - z_next) ** 2))
        for z, v, z_next in zip(Z, V, Z_next)
    ]
    return float(np.mean(errors))


def rollout_output_error(
    predictor: Predictor, rows: list[dict], config: ReducedStateConfig
) -> float:
    """Multi-step rollout (PDF section 8, "multi-step rollout"): seed each
    held-out trajectory's z from the true initial reading, then propagate
    purely from the predictor using the trajectory's actual recorded input
    sequence -- no re-grounding on the true z_t along the way -- and compare
    the predicted y against the true y at every subsequent turn. `rows`
    should come from a held-out split, same as `one_step_error`."""

    squared_errors: list[float] = []
    for traj_rows in group_by_trajectory(rows).values():
        pairs = build_reduced_state_pairs(traj_rows, config)
        if not pairs:
            continue
        z = pairs[0]["z"]
        squared_errors.append((predictor.readout(z) - pairs[0]["y"]) ** 2)
        for pair in pairs:
            z = predictor.step(z, pair["v"])
            true_y_next = pair["z_next"][config.nu - 1]  # y_(t+1): last of z_next's y-block
            squared_errors.append((predictor.readout(z) - true_y_next) ** 2)
    return float(np.mean(squared_errors)) if squared_errors else float("nan")
