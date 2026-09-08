"""ERGO/Laban line's Koopman-MPC controller (Phase C).

`control.KoopmanMPCController._current_state` hardcodes persona-drift's
`y_probe`/`u_remind` column names and has no `aux_cols` support at all --
both are correct for `defense`/`stance` and must not change under those
lines' feet (they're shared, tested infra). This module extends it by
**inheritance only**: a new subclass here, zero edits to
control.py/controller_cli.py.

**2026-09-07 (docs/experiments/signal_resolution_plan.md section 4.1)**:
`_simulate` is now overridden to truth-override the aux (`shard_frac`)
dimension at every lookahead step instead of letting the fitted A/B/b row
free-run it -- `shard_frac_{t+k} = (turn+k)/num_shards` is deterministic and
fully known at decision time (no reason to let a linear model "predict" an
already-known quantity, and its fitted self-coefficient is >1, which
diverges under naive multi-step rollout). `episode_length` is likewise
taken from `history[-1]["num_shards"]` per trajectory rather than a fixed
CLI constant, since ERGO's episode length varies by item. The original
open-question write-up (before this was resolved) is archived at
docs/experiments/backup/ergo_koopman_mpc_opus_design_questions.md.
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
    """`_current_state` override: configurable `y_col`/`u_col` (ERGO uses
    `closeness`/`u_reset` as of F2 -- not persona-drift's `y_probe`/
    `u_remind`, and not the binary `y_task_success` either, see
    docs/experiments/signal_resolution_plan.md section 3.2) and `aux_fns`
    (row -> float callables, evaluated at the most recent row, appended
    after `y_hist`/`v_hist` -- matches the state ordering
    `modeling.dataset.build_reduced_state_pairs` used when fitting, see
    that function's `aux_now` construction).

    `_simulate` and `next_u_remind` overrides: truth-overrides the aux
    (`shard_frac`) dimension at every lookahead step (signal_resolution_plan.md
    section 4.1) and takes `episode_length` from the trajectory's own
    `num_shards` instead of a fixed constant. `next_u_remind` stashes
    `_lookahead_turn`/`_num_shards` as plain instance attributes before
    delegating to the parent (which is what actually invokes `_simulate`) --
    the controller instance is reused sequentially across trajectories/turns
    (see `controller_cli.make_controller_factory`), never concurrently, so
    this mutate-then-delegate pattern is safe. `_remaining_budget` re-reads
    `self.u_col` (not inherited unchanged: delegating to the parent's own
    `_remaining_budget` would read the parent's hardcoded `u_remind` column,
    the F3 bug's shape).

    **2026-09-08 (E4a, docs/experiments/two_task_success_plan.md section 2 E4
    / section 12.2-12.3)**: `objective` ("terminal", the new default, or
    "sum", the prior/ablation behavior) controls what `_simulate` returns --
    "terminal" propagates only the leaf (final-turn) value up through the
    recursion (every non-leaf step's own value, including any
    `repeat_penalty`, is discarded -- B4(a)); the inherited tie-break in the
    parent's `next_u_remind` (`if value > best_value`, strict, so an exact
    tie keeps action=0) is unchanged (B4(b)). `forced_last_reset` makes the
    last turn of every trajectory an unconditional reset (u=1), not a
    decision: `_remaining_budget` reserves one extra unit for every turn
    before the last, and `next_u_remind` returns 1 unconditionally on the
    last turn, bypassing the parent's `remaining_budget <= 0` early return
    (B3 -- a uniform "budget - 1 every turn" scheme would let the parent
    silently eat that forced reset, the F3 bug's shape again).
    `pad_short_history` now defaults to `True` here (C1), overriding the
    parent's `False` default -- construction-layer only, no CLI."""

    y_col: str = "closeness"
    u_col: str = "u_reset"
    aux_fns: tuple[Callable[[dict[str, Any]], float], ...] = ()
    objective: str = "terminal"
    forced_last_reset: bool = False
    pad_short_history: bool = True
    _lookahead_turn: int | None = None
    _num_shards: int | None = None
    n_missing_num_shards: int = 0

    def __post_init__(self) -> None:
        if self.objective not in ("terminal", "sum"):
            raise ValueError(f"objective must be 'terminal' or 'sum', got {self.objective!r}")

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

    def _remaining_budget(self, history: list[dict[str, Any]]) -> int | None:
        """Override: the parent counts spend via the hardcoded `u_remind`
        column, which is never present on an ERGO row (`u_reset` is) -- so
        the inherited version always read 0 rows spent and silently never
        enforced `remind_budget` at all. Same logic otherwise, just reading
        `self.u_col`.

        B3 (`docs/experiments/two_task_success_plan.md` section 12.2): when
        `forced_last_reset` is set and the turn being decided (`
        self._lookahead_turn`, stashed by `next_u_remind` before this is
        called) is strictly before the trajectory's own last turn
        (`self._num_shards`), one more unit is reserved on top of the
        parent's count -- the forced reset that `next_u_remind` will place
        unconditionally on the last turn is a known future spend, not a
        turn-by-turn decision, so it must not compete with the turns before
        it for the same budget."""

        if self.remind_budget is None:
            return None
        budget = self.remind_budget - sum(int(row.get(self.u_col, 0)) for row in history)
        if (
            self.forced_last_reset
            and self._num_shards is not None
            and self._lookahead_turn is not None
            and self._lookahead_turn < self._num_shards
        ):
            return budget - 1
        return budget

    def next_u_remind(self, turn: int, history: list[dict[str, Any]]) -> int:
        self._lookahead_turn = int(turn)
        raw_num_shards = history[-1].get("num_shards") if history else None
        self._num_shards = int(raw_num_shards) if raw_num_shards is not None else None
        if self._num_shards is None:
            self.n_missing_num_shards += 1
        else:
            self.episode_length = self._num_shards
        if self.forced_last_reset and self._num_shards is not None and turn == self._num_shards:
            # B3: unconditional, bypassing the parent's `remaining_budget <=
            # 0` early return at control.py -- with `_remaining_budget`
            # reserving exactly one unit for this turn (above), a uniform
            # "budget - 1 every turn" scheme would let this turn's own
            # reservation be silently swallowed by that early return, the
            # F3 bug's shape.
            return 1
        return super().next_u_remind(turn, history)

    def _simulate(
        self,
        z: np.ndarray,
        action: int,
        remaining_steps: int,
        remaining_budget: int | None = None,
        lookahead_offset: int = 0,
    ) -> float:
        z_next = self.surrogate.step(z, np.array([float(action)]))
        if self._num_shards is not None:
            z_next[-1] = min(1.0, (self._lookahead_turn + lookahead_offset) / self._num_shards)
        value = float(self.surrogate.readout(z_next)) - (self.repeat_penalty if action else 0.0)
        if remaining_steps <= 0:
            return value
        budget_after = None if remaining_budget is None else remaining_budget - action
        next_turn = None
        if self.forced_last_reset and self._num_shards is not None and self._lookahead_turn is not None:
            next_turn = self._lookahead_turn + lookahead_offset + 1
        if next_turn is not None and next_turn == self._num_shards:
            # B3: the planner treats the last turn's action as a known
            # constant (u=1), not a decision variable -- it is not
            # enumerated over (0, 1) and is not masked by budget (the
            # budget for it was already reserved in `_remaining_budget`).
            candidates: tuple[int, ...] = (1,)
        else:
            candidates = (0, 1) if budget_after is None or budget_after >= 1 else (0,)
        children = (
            self._simulate(z_next, a, remaining_steps - 1, budget_after, lookahead_offset + 1) for a in candidates
        )
        if self.objective == "terminal":
            # Only the leaf (final-turn) `value` reaches the returned total
            # -- every intermediate step's own `value` (readout + repeat
            # penalty) is discarded, only the subtree max propagates.
            return max(children)
        if self.objective == "sum":
            return value + max(children)
        raise ValueError(f"objective must be 'terminal' or 'sum', got {self.objective!r}")


def load_ergo_koopman_mpc_controller(
    model_path: pathlib.Path,
    model_key: str,
    nu: int,
    mu: int,
    horizon: int,
    repeat_penalty: float,
    remind_budget: int | None = None,
    y_col: str = "closeness",
    name: str = "koopman_mpc",
) -> ErgoKoopmanMPCController:
    """Mirrors `controller_cli.load_koopman_mpc_controller`, but builds an
    `ErgoKoopmanMPCController` with `contemporaneous_v=True` and
    `aux_cols=("shard_frac",)` hardcoded -- both are load-bearing decisions
    for this domain (pilot doc's "Koopman 建模，Phase B" section), not
    optional flags the way they are for the defense line.

    `y_col` defaults to `"closeness"` (F2's fitted state, `model_path`
    should then point at `koopman_fit_report_closeness.json`) -- the
    reported/evaluated metric stays the binary `final_turn_success`
    regardless (signal_resolution_plan.md section 3.2), since the
    controller's state and the headline metric are deliberately different
    (though same-instrument) quantities. `episode_length` is not a
    parameter here: `ErgoKoopmanMPCController.next_u_remind` sets it
    per-trajectory from `history[-1]["num_shards"]`."""

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
        y_col=y_col,
        u_col="u_reset",
        aux_fns=(shard_frac,),
        name=name,
    )
