"""The trivial predictors every surrogate must beat before it means anything.

This is the authoritative copy. It is lifted verbatim (same arithmetic, same
column order) from `fit_koopman_defense_model._null_predictions`, which
G-K2-1 and G-S2-1 were both computed with; `fit_koopman_sequor_model` already
imported that one rather than writing its own, with the reason in its comment:
two copies of a null definition drift apart. A third copy would have been the
drift.

Why the nulls are shaped this way: a surrogate that only recovers a
deterministic exogenous ramp has learned nothing about the system. So every
null is handed `turn` -- and any other quantity the environment fixes in
advance -- and the surrogate has to beat the best of them.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def trivial_nulls(
    train: Mapping[str, np.ndarray],
    test: Mapping[str, np.ndarray],
    *,
    exogenous: Sequence[str],
) -> dict[str, np.ndarray]:
    """Fit the three nulls on `train`, predict `test['y_next']`.

    `exogenous` names the deterministic, environment-fixed columns the
    stateless regression is allowed to use, in the order they enter the design
    matrix. It must contain `turn_next`: a null that is not told the turn index
    can be beaten by a surrogate that has only learned the ramp, and that
    verdict would be an artifact.
    """
    if "turn_next" not in exogenous:
        raise ValueError(
            "every null gets `turn`: exogenous must contain 'turn_next', got "
            f"{tuple(exogenous)}"
        )
    missing = [k for k in ("y_next", "turn_next", *exogenous) if k not in train or k not in test]
    if missing:
        raise KeyError(f"train/test are missing columns {missing}")

    const = float(np.mean(train["y_next"]))

    turn_means = {
        t: float(np.mean(train["y_next"][train["turn_next"] == t]))
        for t in np.unique(train["turn_next"])
    }
    turn_mean = np.array([turn_means.get(t, const) for t in test["turn_next"]])

    X_train = np.column_stack([np.ones(len(train["y_next"])), *(train[k] for k in exogenous)])
    beta, *_ = np.linalg.lstsq(X_train, train["y_next"], rcond=None)
    X_test = np.column_stack([np.ones(len(test["y_next"])), *(test[k] for k in exogenous)])

    return {
        "const": np.full(len(test["y_next"]), const),
        "turn_mean": turn_mean,
        "stateless": X_test @ beta,
    }


def best_null(predictions: Mapping[str, np.ndarray], y_true: np.ndarray) -> tuple[str, np.ndarray]:
    """The null that is hardest to beat on these rows, and its name.

    Returned as a pair so the name can be printed next to the number: which
    null won is itself a diagnostic (a `stateless` win means the action or the
    ramp carries the row, a `turn_mean` win means the level does).
    """
    if not predictions:
        raise ValueError("no null predictions given")
    name = min(predictions, key=lambda k: float(np.mean((predictions[k] - y_true) ** 2)))
    return name, predictions[name]
