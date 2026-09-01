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
    was fit with (same nu/mu), so the state built here from `history` lines
    up with what `modeling.dataset.build_reduced_state_pairs` built during
    fitting. Falls back to u_remind=0 (same convention as
    `ThresholdController`) whenever there isn't yet enough history to form a
    state -- this only affects the first `max(nu-1, mu)` turns of a
    trajectory, exactly the positions `build_reduced_state_pairs` itself
    skips during fitting.
    """

    surrogate: KoopmanSurrogate
    state_config: ReducedStateConfig
    horizon: int = 2
    repeat_penalty: float = 0.0
    name: str = "koopman_mpc"

    def _current_state(self, history: list[dict[str, Any]]) -> np.ndarray | None:
        nu, mu = self.state_config.nu, self.state_config.mu
        if len(history) < max(nu - 1, mu) + 1:
            return None
        ys = [row["y_probe"] for row in history]
        vs = [float(row["u_remind"]) for row in history]
        t = len(history) - 1
        y_hist = ys[t - nu + 1 : t + 1]
        v_hist = vs[t - mu : t] if mu > 0 else []
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
