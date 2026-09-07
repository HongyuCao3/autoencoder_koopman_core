"""ERGO/Laban line's Koopman-MPC controller (Phase C scaffold).

`control.KoopmanMPCController._current_state` hardcodes persona-drift's
`y_probe`/`u_remind` column names and has no `aux_cols` support at all --
both are correct for `defense`/`stance` and must not change under those
lines' feet (they're shared, tested infra). This module extends it by
**inheritance only**, per the 2026-09-07 decision in
docs/experiments/ergo_multiturn_reliability_pilot.md's Phase C planning: a
new subclass here, zero edits to control.py/controller_cli.py.

Open design question this file deliberately does NOT resolve: `_simulate`
is inherited unchanged from `KoopmanMPCController`, so multi-step lookahead
still lets the fitted A/B/b row for `shard_frac` free-run forward instead of
substituting the true, deterministically known `(turn+k)/num_shards` at each
lookahead step. That is a placeholder, not an oversight -- see
docs/experiments/signal_resolution_plan.md sections 4.0-4.2, which RESOLVED it
    (truth-override the shard_frac dimension during lookahead; k=1 reset budget
    justified by measured token cost; y_col made configurable so the state can be
    `closeness`). The original open-question write-up is archived at
    docs/experiments/backup/ergo_koopman_mpc_opus_design_questions.md for why it needs
a design decision before Phase C can run for real (variable per-item
episode length interacts with it too, same doc, question 2).
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .control import KoopmanMPCController
from .modeling.dataset import ReducedStateConfig
from .modeling.koopman import abs_sign_extra_features, no_extra_features, surrogate_from_arrays

EXTRA_FEATURES_FNS = {"arx": no_extra_features, "richer_abs_sign": abs_sign_extra_features}


def shard_frac(row: dict[str, Any]) -> float:
    """The one aux feature Phase B's fit used (`scripts/fit_koopman_ergo_model.py`
    `_add_shard_frac`): fraction of an item's shards revealed by this row's
    turn. Computed from `turn`/`num_shards`, both already present on every
    `ergo_math_trajectory.run_ergo_math_trajectory` row -- no separate
    `shard_frac` column needs to exist on the row."""

    return row["turn"] / row["num_shards"]


@dataclass
class ErgoKoopmanMPCController(KoopmanMPCController):
    """`_current_state` override only: configurable `y_col`/`u_col` (ERGO
    uses `y_task_success`/`u_reset`, not persona-drift's `y_probe`/
    `u_remind`) and `aux_fns` (row -> float callables, evaluated at the most
    recent row, appended after `y_hist`/`v_hist` -- matches the state
    ordering `modeling.dataset.build_reduced_state_pairs` used when fitting,
    see that function's `aux_now` construction). Everything else
    (`_simulate`, `_remaining_budget`, `_planning_steps`, `next_u_remind`) is
    inherited unchanged."""

    y_col: str = "y_task_success"
    u_col: str = "u_reset"
    aux_fns: tuple[Callable[[dict[str, Any]], float], ...] = ()

    def _current_state(self, history: list[dict[str, Any]]) -> np.ndarray | None:
        nu, mu = self.state_config.nu, self.state_config.mu
        shift = 1 if self.state_config.contemporaneous_v else 0
        min_len = nu if self.pad_short_history else max(nu - 1, mu - shift) + 1
        if len(history) < min_len:
            return None
        ys = [row[self.y_col] for row in history]
        vs = [float(row[self.u_col]) for row in history]
        t = len(history) - 1
        tv = t + shift
        y_hist = ys[t - nu + 1 : t + 1]
        if mu == 0:
            v_hist = []
        elif self.pad_short_history:
            v_hist = [vs[i] if 0 <= i < len(vs) else 0.0 for i in range(tv - mu, tv)]
        else:
            v_hist = vs[tv - mu : tv]
        if any(value != value for value in y_hist):  # NaN: scorer failure upstream
            return None
        aux_now = [fn(history[t]) for fn in self.aux_fns]
        return np.array(y_hist + v_hist + aux_now, dtype=float)


def load_ergo_koopman_mpc_controller(
    model_path: pathlib.Path,
    model_key: str,
    nu: int,
    mu: int,
    horizon: int,
    repeat_penalty: float,
    remind_budget: int | None = None,
    episode_length: int | None = None,
    name: str = "koopman_mpc",
) -> ErgoKoopmanMPCController:
    """Mirrors `controller_cli.load_koopman_mpc_controller`, but builds an
    `ErgoKoopmanMPCController` with `contemporaneous_v=True` and
    `aux_cols=("shard_frac",)` hardcoded -- both are load-bearing decisions
    for this domain (pilot doc's "Koopman 建模，Phase B" section), not
    optional flags the way they are for the defense line."""

    report = json.loads(model_path.read_text())
    fit = report[model_key]
    state_config = ReducedStateConfig(nu=nu, mu=mu, contemporaneous_v=True, aux_cols=("shard_frac",))
    surrogate = surrogate_from_arrays(
        fit["A"],
        fit["B"],
        fit["b"],
        fit["C"],
        state_dim=state_config.state_dim,
        extra_features_fn=EXTRA_FEATURES_FNS[model_key],
    )
    return ErgoKoopmanMPCController(
        surrogate=surrogate,
        state_config=state_config,
        horizon=horizon,
        repeat_penalty=repeat_penalty,
        remind_budget=remind_budget,
        episode_length=episode_length,
        y_col="y_task_success",
        u_col="u_reset",
        aux_fns=(shard_frac,),
        name=name,
    )
