"""Uncertainty for the main table: one bootstrap, one seed aggregation.

Both carry a refusal. The bootstrap refuses to resample rows (turns inside one
dialogue are dependent, so a row bootstrap reports an interval that is too
narrow); the seed aggregation refuses fewer than three seeds, which is
`global.md`'s report caliber written as a guard instead of a sentence --
"Phase J: the budget setting holds, but 2 seeds cannot separate the arms".
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np


def bootstrap_ci(
    values: np.ndarray,
    groups: Sequence,
    *,
    seed: int,
    n_resamples: int = 2000,
    statistic: Callable[[np.ndarray], float] = lambda v: float(np.mean(v)),
    alpha: float = 0.05,
) -> dict[str, float]:
    """Percentile CI, resampling whole `groups` (item / trajectory), not rows.

    For a paired contrast, pass the per-row difference as `values`; the pairing
    is the caller's, and grouping by item is what removes the between-item
    level that would otherwise dominate the variance (the S3 sizing bug of
    2026-09-10 was exactly an unpaired sd priced into a paired design).
    """
    values = np.asarray(values, dtype=float)
    groups = np.asarray(groups)
    if values.shape[0] != groups.shape[0]:
        raise ValueError(f"values has {values.shape[0]} rows but groups has {groups.shape[0]}")
    unique = np.unique(groups)
    if unique.size < 2:
        raise ValueError(f"need >= 2 groups to bootstrap over, got {unique.size}")

    index_of = {g: np.flatnonzero(groups == g) for g in unique}
    rng = np.random.default_rng(seed)
    draws = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        picked = rng.choice(unique, size=unique.size, replace=True)
        draws[i] = statistic(values[np.concatenate([index_of[g] for g in picked])])

    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return {
        "point": statistic(values),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "bootstrap_sd": float(np.std(draws, ddof=1)),
        "n_groups": int(unique.size),
        "n_rows": int(values.shape[0]),
        "excludes_zero": bool(lo > 0 or hi < 0),
    }


def seed_aggregate(per_seed: Sequence[float], *, ddof: int) -> dict:
    """`mean +/- std (n)` over seeds, with `n` the seed count, not the sample count.

    `ddof` has no default on purpose. The repository currently carries two
    conventions -- `ABLATION_STUDY.md` reports a population sd, the
    `constraint` line reports a sample sd -- and a default here would silently
    pick one for the main table. See MAIN_TABLE_DESIGN.md section 6 item 5.
    """
    values = np.asarray(list(per_seed), dtype=float)
    if values.size < 3:
        raise ValueError(
            f"n={values.size} seeds: report caliber requires >= 3 "
            "(single-seed point estimates and 2 seeds are both banned)"
        )
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=ddof))
    return {
        "mean": mean,
        "std": std,
        "ddof": ddof,
        "n_seeds": int(values.size),
        "per_seed": [float(v) for v in values],
        "reportable": f"{mean:+.4f} +/- {std:.4f} (n={values.size} seeds)",
    }
