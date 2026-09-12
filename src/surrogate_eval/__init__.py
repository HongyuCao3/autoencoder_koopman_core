"""Shared surrogate-model evaluation for the paper's main table.

Depends on neither code base (`.claude/global.md`: the two systems in this
repository do not mix). Both call in: `src/koopman_ae` for the eight core
trajectory tasks, `persona_drift_control` for `defense` / `gsm8k_sharded` /
`constraint`. The table is only readable across ten datasets if one
definition of skill, one bootstrap and one fold protocol produced every cell.
"""
from surrogate_eval.folds import make_folds
from surrogate_eval.nulls import best_null, trivial_nulls
from surrogate_eval.resample import bootstrap_ci, seed_aggregate
from surrogate_eval.skill import DegenerateNullError, skill_from_squared_errors, skill_h

__all__ = [
    "DegenerateNullError",
    "best_null",
    "bootstrap_ci",
    "make_folds",
    "seed_aggregate",
    "skill_from_squared_errors",
    "skill_h",
    "trivial_nulls",
]
