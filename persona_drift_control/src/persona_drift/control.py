"""Pluggable turn-level controllers for channel A (u_remind).

`run_trajectory` (selfchat.py) asks a `Controller` for the next `u_remind`
value and otherwise knows nothing about how that value was chosen. This
decouples the generation/measurement harness from the control policy, so the
random open-loop excitation used for system identification
(DATA_COLLECTION_PROTOCOL.md section 7) and simple baseline feedback policies
share the exact same agent model, simulated user, system prompts, seeds, and
probe scoring -- the precondition for any of them to be fairly comparable to
a future Koopman-MPC controller on the same channel.

`ZeroControlController.name` and `RandomExciteController.name` must stay
"zero_control" and "iid": analysis.analyze_screening filters rows by those
exact `excitation_design` values.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from .modeling.dataset import ReducedStateConfig
from .modeling.koopman import KoopmanSurrogate


class Controller(Protocol):
    name: str

    def next_u_remind(self, turn: int, history: list[dict[str, Any]]) -> int:
        """Return the u_remind value (0 or 1) for `turn`, given the rows
        already recorded for this trajectory (oldest first, one per prior
        turn)."""
        ...


@dataclass
class ZeroControlController:
    """u_remind == 0 every turn: the free-drift baseline, and the u==0
    condition required by gate question 1."""

    name: str = "zero_control"

    def next_u_remind(self, turn: int, history: list[dict[str, Any]]) -> int:
        return 0


@dataclass
class ConstantRemindController:
    """u_remind == 1 every turn: the "brute force" always-remind baseline."""

    name: str = "constant_remind"

    def next_u_remind(self, turn: int, history: list[dict[str, Any]]) -> int:
        return 1


@dataclass
class RandomExciteController:
    """u_remind drawn i.i.d. Bernoulli(p) each turn: open-loop excitation for
    system identification (gate questions 2/3, and the real B/A/E1 protocol
    scale collection)."""

    p: float
    seed: int
    name: str = "iid"
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def next_u_remind(self, turn: int, history: list[dict[str, Any]]) -> int:
        return int(self._rng.random() < self.p)


@dataclass
class PeriodicController:
    """Remind every `period` turns (turn indices start at 1). A fixed-schedule
    policy, closer to how a real deployment might space out reminders than
    either the zero or constant-remind extremes."""

    period: int
    name: str = "periodic"

    def next_u_remind(self, turn: int, history: list[dict[str, Any]]) -> int:
        return int(turn % self.period == 0)


@dataclass
class ThresholdController:
    """Bang-bang feedback: remind iff the most recently measured y_probe fell
    below y_min. The classical-control baseline a Koopman-MPC controller must
    beat to justify its added modeling complexity."""

    y_min: float
    name: str = "threshold"

    def next_u_remind(self, turn: int, history: list[dict[str, Any]]) -> int:
        if not history:
            return 0
        last_y = history[-1]["y_probe"]
        if last_y != last_y:  # nan: scorer failure, don't act on it
            return 0
        return int(last_y < self.y_min)


@dataclass
class KoopmanMPCController:
    """Receding-horizon MPC over the binary u_remind action space, using a
    fitted `modeling.koopman.KoopmanSurrogate` to roll forward its own
    predicted y_safety under each candidate 0/1 action sequence and picking
    whichever first action leads to the best predicted total safety over
    `horizon` future turns. The action space is small enough (2 values,
    horizon typically 2-3) to brute-force enumerate every sequence rather
    than needing a QP/MPC solver library -- see
    docs/experiments/koopman_defense_pilot.md.

    `state_config` must match whatever `ReducedStateConfig` the surrogate
    was fit with (same nu/mu/contemporaneous_v), so the state built here
    from `history` lines up with what `modeling.dataset.build_reduced_state_pairs`
    built during fitting. Falls back to u_remind=0 (same convention as
    `ThresholdController`) whenever there isn't even `nu` turns of history to
    read `y` from at all -- the one thing that's unavoidable regardless of
    `mu`/`pad_short_history`, since deciding turn T's action always needs
    turn T-1's own `y` (`nu`'s worth of it) to build a state from in the
    first place; this is why turn 1 always defaults to 0 for every
    `KoopmanMPCController` config.

    `pad_short_history` (default `False`, matches all prior behavior):
    when there's `>=nu` history but fewer than `mu` real actions to fill the
    lag window with (e.g. only turn 1 completed and `mu=2` needs 2 lagged
    actions), the default `False` still falls back to 0 -- exactly the
    positions `build_reduced_state_pairs` itself skips during fitting (see
    `ReducedStateConfig.contemporaneous_v` for what `shift` is and why: with
    it set, the candidate action passed to `_simulate` lands in the same
    "next action directly causes next y" slot the surrogate was fit on,
    instead of the contemporaneous/residual slot). Set `pad_short_history=True`
    to instead zero-pad the missing (pre-trajectory) lag slots -- treating
    "no reminder was ever inserted before the trajectory started" as a
    reasonable prior -- and make a real (if less certain) decision as early
    as turn 2, instead of only from turn `max(nu-1, mu-shift)+1` onward.
    Found while executing docs/next step.md (2026-09-02): with
    `contemporaneous_v=True` and `mu=2`, the earliest real decision is
    turn 3, one turn later than `PeriodicController(period=2)`'s turn 2 --
    a real closed-loop test (Phase I, docs/experiments/koopman_case_study_design.md)
    showed a v-aligned interaction model reverses Phase H's wrong-direction
    failure and is more economical, but still loses the new-Q1 significance
    test because the natural early decline (turns 1-3) happens before this
    reactive policy's first real decision -- `pad_short_history` is a cheap,
    offline-checkable way to test whether letting it act one turn earlier
    closes that gap, matching the same lag-window zero-padding
    `scripts/analyze_compounding_hypothesis.py`'s rollout already used.
    """

    surrogate: KoopmanSurrogate
    state_config: ReducedStateConfig
    horizon: int = 2
    repeat_penalty: float = 0.0
    name: str = "koopman_mpc"
    pad_short_history: bool = False

    def _current_state(self, history: list[dict[str, Any]]) -> np.ndarray | None:
        nu, mu = self.state_config.nu, self.state_config.mu
        shift = 1 if self.state_config.contemporaneous_v else 0
        min_len = nu if self.pad_short_history else max(nu - 1, mu - shift) + 1
        if len(history) < min_len:
            return None
        ys = [row["y_probe"] for row in history]
        vs = [float(row["u_remind"]) for row in history]
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
        return np.array(y_hist + v_hist, dtype=float)

    def _simulate(self, z: np.ndarray, action: int, remaining_steps: int) -> float:
        z_next = self.surrogate.step(z, np.array([float(action)]))
        value = float(self.surrogate.readout(z_next)) - (self.repeat_penalty if action else 0.0)
        if remaining_steps <= 0:
            return value
        return value + max(self._simulate(z_next, a, remaining_steps - 1) for a in (0, 1))

    def next_u_remind(self, turn: int, history: list[dict[str, Any]]) -> int:
        z = self._current_state(history)
        if z is None:
            return 0
        best_action, best_value = 0, float("-inf")
        for action in (0, 1):
            value = self._simulate(z, action, self.horizon - 1)
            if value > best_value:
                best_value, best_action = value, action
        return best_action
