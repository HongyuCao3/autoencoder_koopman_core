"""`Skill_H`: the one definition the paper's main table is allowed to use.

Ruling 2 (2026-09-12) lifted `global.md`'s ban on prediction error as a
headline number for exactly one quantity -- multi-step, held-out,
pre-registered skill against the best trivial null -- and left the ban on
one-step error standing. Keeping that distinction in one function is the
point of this module: there is no argument here that turns it back into a
one-step number.
"""
from __future__ import annotations

import numpy as np


class DegenerateNullError(RuntimeError):
    """The best null already has (near-)zero error, so skill is undefined.

    Raised rather than returned as NaN because this is not a bad score, it is
    a readout with no range -- `even_odd_t5` in the core suite, the ceiling on
    `defense`, `c_prev`'s zero IQR on `gsm8k_sharded`. Callers that want to
    report the degeneracy (appendix A1) catch this deliberately.
    """


def skill_from_squared_errors(
    se_model: np.ndarray,
    se_null: np.ndarray,
    *,
    horizon: int,
    null_mse_floor: float = 1e-12,
) -> float:
    """`skill_h` for callers that already hold per-row squared errors.

    Needed because a row that averages several training seeds has no single
    prediction vector, only a mean squared error per row. `skill_h` delegates
    here so there is still exactly one definition of skill and one degeneracy
    guard, rather than a second copy that can drift from the first.
    """
    if horizon < 2:
        raise ValueError(
            f"horizon={horizon}: one-step error stays banned as a headline number "
            "(ruling 2, 2026-09-12); report it in the appendix instead"
        )
    se_model, se_null = np.asarray(se_model, float), np.asarray(se_null, float)
    if se_model.shape != se_null.shape:
        raise ValueError(f"shape mismatch: model {se_model.shape}, null {se_null.shape}")
    if se_model.size == 0:
        raise ValueError("no rows to score")
    mse_null = float(np.mean(se_null))
    if mse_null <= null_mse_floor:
        raise DegenerateNullError(
            f"best-null MSE {mse_null:.3e} <= {null_mse_floor:.0e}: the readout has no range "
            "on these rows, so skill is undefined"
        )
    return 1.0 - float(np.mean(se_model)) / mse_null


def skill_h(
    y_true: np.ndarray,
    pred_model: np.ndarray,
    pred_null: np.ndarray,
    *,
    horizon: int,
    null_mse_floor: float = 1e-12,
) -> float:
    """1 - MSE_H(model) / MSE_H(best null), on H-step rollout predictions.

    Takes predictions, never models: the rollout itself is each code base's
    own (core rolls an augmented state, the behavioural lines roll a lifted
    one), and pulling either of them in here would tie the two systems
    together for no gain.
    """
    y_true, pred_model, pred_null = (np.asarray(a, dtype=float) for a in (y_true, pred_model, pred_null))
    if not (y_true.shape == pred_model.shape == pred_null.shape):
        raise ValueError(
            f"shape mismatch: y_true {y_true.shape}, model {pred_model.shape}, null {pred_null.shape}"
        )
    return skill_from_squared_errors(
        (pred_model - y_true) ** 2, (pred_null - y_true) ** 2,
        horizon=horizon, null_mse_floor=null_mse_floor,
    )
