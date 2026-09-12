"""Group-disjoint folds, split by what the numbers will be used for.

Ruling 2 (2026-09-12) let multi-step `Skill_H` into the main table but kept
`global.md`'s rule that a quantity used to select cannot also be used to
report. `constraint`'s `Skill_H` was the S3 admission gate, so the main-table
cell has to be computed on a partition that gate never saw. That is not a
sentence in a document here: asking for one purpose returns a partition, and
asking for a partition that coincides with the other purpose's raises.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

# Fixed salts, not `hash()`: PYTHONHASHSEED randomises str hashing per process,
# which would make a fold assignment unreproducible across runs.
_PURPOSE_SALT = {"gate": 0x9E3779B9, "report": 0x85EBCA6B}


def _partition(unique: np.ndarray, n_folds: int, effective_seed: int) -> list[np.ndarray]:
    order = np.random.default_rng(effective_seed).permutation(unique)
    return [np.sort(chunk) for chunk in np.array_split(order, n_folds)]


def make_folds(
    groups: Sequence,
    n_folds: int,
    *,
    purpose: str,
    seed: int,
) -> list[dict[str, np.ndarray]]:
    """`n_folds` folds that are disjoint in `groups` (item / attack / trajectory).

    Splitting by group rather than by row is the anti-leak rule both fit
    scripts already use: turns inside one dialogue are dependent, and two
    seeds of one item are the same item.
    """
    if purpose not in _PURPOSE_SALT:
        raise ValueError(f"purpose must be one of {sorted(_PURPOSE_SALT)}, got {purpose!r}")
    groups = np.asarray(groups)
    unique = np.unique(groups)
    if n_folds < 2:
        raise ValueError(f"n_folds={n_folds}: need at least 2")
    if n_folds > unique.size:
        raise ValueError(f"n_folds={n_folds} exceeds the {unique.size} distinct groups")

    effective = seed ^ _PURPOSE_SALT[purpose]
    mine = _partition(unique, n_folds, effective)
    other = _partition(unique, n_folds, seed ^ _PURPOSE_SALT["report" if purpose == "gate" else "gate"])
    as_set = lambda p: frozenset(frozenset(c.tolist()) for c in p)
    if as_set(mine) == as_set(other):
        raise RuntimeError(
            f"the {purpose} partition coincides with the other purpose's at seed={seed}; "
            "a cell computed on it would be reporting on the folds that selected it -- "
            "change the seed"
        )

    folds = []
    for test_groups in mine:
        is_test = np.isin(groups, test_groups)
        folds.append(
            {
                "test_groups": test_groups,
                "test_rows": np.flatnonzero(is_test),
                "train_rows": np.flatnonzero(~is_test),
            }
        )
    return folds
