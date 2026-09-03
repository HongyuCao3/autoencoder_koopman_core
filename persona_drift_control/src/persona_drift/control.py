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
class FixedScheduleController:
    """Remind on exactly the listed (1-indexed) turns, nothing else: the
    budget-matched fixed-allocation baseline for the budget-constrained
    setting (`KoopmanMPCController.remind_budget`, docs/next_step_diagnosis.md
    section 4 step 2). `PeriodicController(period=2)` already *is* the
    k=2 fixed allocation on the 5-turn attack sequences (it fires on turns
    2 and 4), which is why Phase G's periodic arm can be reused as-is
    there; a budget that a period can't express -- k=1 above all -- needs
    this explicit schedule instead of a fake period.

    `name` encodes the schedule (e.g. `fixed_schedule_t2`) so the
    `excitation_design` column, the `logs/` filename and the run id
    distinguish two arms that differ only in where the budget was spent.
    """

    turns: tuple[int, ...]
    name: str = ""

    def __post_init__(self) -> None:
        self.turns = tuple(sorted(int(turn) for turn in self.turns))
        if not self.name:
            self.name = "fixed_schedule_t" + "_".join(str(turn) for turn in self.turns)

    def next_u_remind(self, turn: int, history: list[dict[str, Any]]) -> int:
        return int(turn in self.turns)


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
    Found while executing docs/next_step_diagnosis.md (2026-09-02): with
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

    `remind_budget` / `episode_length` (both default `None` = every prior
    phase's unbudgeted behavior) turn this into the budget-constrained
    control problem `docs/next_step_diagnosis.md` section 4 step 2 asks for: at most
    `remind_budget` reminders per trajectory, so the policy's job is *where*
    to place them rather than whether to remind at all. Enumeration is
    pruned to action sequences that respect the budget still left (counted
    from `history`, see `_remaining_budget`), and `episode_length` clips the
    horizon to the turns actually remaining (see `_planning_steps`) so
    "spend now vs save for a worse turn" is scored against the real number
    of remaining chances. This is the setting in which an adaptive policy
    can even in principle beat a fixed schedule
    (`FixedScheduleController`): Phase F showed the benign helpfulness cost
    of a reminder is small and Phase E/I that reminders help, so unbudgeted
    the optimal policy is trivially "always remind" and there is no
    allocation problem to be better at -- see
    docs/experiments/koopman_defense_pilot.md.
    """

    surrogate: KoopmanSurrogate
    state_config: ReducedStateConfig
    horizon: int = 2
    repeat_penalty: float = 0.0
    name: str = "koopman_mpc"
    pad_short_history: bool = False
    remind_budget: int | None = None
    episode_length: int | None = None

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

    def _remaining_budget(self, history: list[dict[str, Any]]) -> int | None:
        """Reminders still available for this trajectory, or None when
        unbudgeted. Counted from `history` (not from instance state) on
        purpose: the controller stays stateless, so
        `controller_cli.make_controller_factory` can keep handing the same
        instance to every trajectory, and a resumed run
        (`screening_common.prepare_resumable_trajectories_file`) recovers
        the budget exactly from the rows already on disk."""

        if self.remind_budget is None:
            return None
        return self.remind_budget - sum(int(row.get("u_remind", 0)) for row in history)

    def _planning_steps(self, turn: int) -> int:
        """How many turns forward to enumerate at `turn`. Without
        `episode_length` this is just `horizon` (all prior phases). With it,
        the horizon is clipped to the turns actually left in the trajectory,
        which is what makes a budget decision meaningful: "spend the last
        reminder now or save it" can only be answered against the real
        number of remaining chances, and enumerating past the end of the
        episode would let imaginary future turns pay for a reminder that
        never gets a turn to act on."""

        if self.episode_length is None:
            return self.horizon
        return max(1, min(self.horizon, self.episode_length - turn + 1))

    def _simulate(
        self, z: np.ndarray, action: int, remaining_steps: int, remaining_budget: int | None = None
    ) -> float:
        z_next = self.surrogate.step(z, np.array([float(action)]))
        value = float(self.surrogate.readout(z_next)) - (self.repeat_penalty if action else 0.0)
        if remaining_steps <= 0:
            return value
        budget_after = None if remaining_budget is None else remaining_budget - action
        candidates = (0, 1) if budget_after is None or budget_after >= 1 else (0,)
        return value + max(self._simulate(z_next, a, remaining_steps - 1, budget_after) for a in candidates)

    def next_u_remind(self, turn: int, history: list[dict[str, Any]]) -> int:
        z = self._current_state(history)
        if z is None:
            return 0
        remaining_budget = self._remaining_budget(history)
        if remaining_budget is not None and remaining_budget <= 0:
            return 0
        steps = self._planning_steps(turn)
        best_action, best_value = 0, float("-inf")
        for action in (0, 1):
            value = self._simulate(z, action, steps - 1, remaining_budget)
            if value > best_value:
                best_value, best_action = value, action
        return best_action


@dataclass
class BudgetLimitedController:
    """Caps any inner controller at `budget` reminders per trajectory.

    For the model-free arms (`ThresholdController` above all) this is the
    whole budget story: they have no forward model, so the only thing a
    budget can do is stop them once it's exhausted -- "spend on the first
    `budget` dips". `KoopmanMPCController` deliberately does NOT go through
    this wrapper: it takes `remind_budget` itself so the constraint enters
    its *planning* (spend now vs save for a turn it predicts will be
    worse), which is the entire hypothesis under test in the
    budget-constrained setting (docs/next_step_diagnosis.md section 4 step 2). Merely
    truncating a greedy policy would test nothing.

    Stateless for the same reason `KoopmanMPCController._remaining_budget`
    is: the count comes from `history`, so instances are shareable across
    trajectories and survive a resumed run.
    """

    inner: Controller
    budget: int
    name: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            self.name = f"{self.inner.name}_budget{self.budget}"

    def next_u_remind(self, turn: int, history: list[dict[str, Any]]) -> int:
        used = sum(int(row.get("u_remind", 0)) for row in history)
        if used >= self.budget:
            return 0
        return self.inner.next_u_remind(turn, history)
