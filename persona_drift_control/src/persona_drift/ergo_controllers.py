"""Mode-B fixed/random-schedule opponent controllers for the ERGO terminal-
objective MPC (docs/experiments/two_task_success_plan.md section 2 E4 item 6
/ section 12.2 B2).

`control.py`'s `FixedScheduleController` and `RandomScheduleController` only
ever spend a k=1 budget. `KoopmanMPCController(forced_last_reset=True)`
(ergo_koopman_mpc.py, mode B) always spends exactly k=2: one reset the
planner is free to place, plus an unconditional reset on the item's own
last turn. For that arm's "does the *first* reset's timing carry
information" comparison to mean anything, its opponents must spend the
same k=2 budget with the same last-turn reset held fixed, varying only
*where* the first reset goes:

- `FixedTAndLastController`: the first reset is a fixed absolute turn `t`
  (the fixed-schedule opponent, mirrors `FixedScheduleController`).
- `RandSchedTAndLastController`: the first reset is drawn uniformly from
  `{1, ..., T-1}` per item (the random-allocation opponent, mirrors
  `RandomScheduleController`).

Both implement the `control.Controller` protocol
(`next_u_remind(self, turn, history) -> int` plus a `.name` attribute) and
are meant to be driven the same way `run_ergo_math_screening.py`'s existing
`fixed_last` branch drives `FixedScheduleController`: a local
`controller_factory(seed, entry_id)` closure looks up
`entry.item_id -> num_shards` from `ergo_math_bank.load_ergo_math_bank()`
and passes it in as a constructor argument.

**Why `num_shards` is a constructor argument here, not read from
`history`:** `ergo_math_trajectory.run_ergo_math_trajectory` calls
`next_u_remind(turn, rows)` where `rows` holds only turns already scored
--  at `turn == 1`, `history` is `[]`, so there is no row to read a
`num_shards` column from yet. Both controllers here need to know
`num_shards` (= T for this item) at *construction* time -- before turn 1's
decision -- to know whether turn 1 is already this item's last turn
(`num_shards == 1`), whether `t == num_shards` (degenerate collapse to a
single reset), and, for the random controller, what range to draw from.
This is exactly the constraint that already forced
`run_ergo_math_screening.py`'s `fixed_last` and `random_schedule` branches
to look `num_shards` up from `load_ergo_math_bank()` outside the controller
rather than read it from `history`; these two controllers follow the same
pattern rather than inventing a second convention.

Neither controller is wired into `run_ergo_math_screening.py`'s
`CONTROLLER_CHOICES` or factory branches here -- that CLI plumbing is a
separate task (E4c). This module only provides the two controller classes
and is otherwise self-contained: it does not import from or modify
`control.py`.
"""

from __future__ import annotations

import random
import warnings
from dataclasses import dataclass, field
from typing import Any


@dataclass
class FixedTAndLastController:
    """Reset on one fixed absolute turn `t` AND on this item's own last
    turn (`num_shards`): the fixed-schedule, k=2, mode-B opponent for
    `KoopmanMPCController(forced_last_reset=True)`.

    Degenerate collapses are surfaced, never silent (docs/experiments/
    two_task_success_plan.md section 2 E4 item 6):

    - `t == num_shards`: the two reset turns coincide, so only ONE reset
      actually fires instead of two. `self.degenerate` is set `True` and
      `self.degenerate_reason = "t_equals_num_shards"`; `self.turns`
      collapses to a 1-element tuple. A `UserWarning` is also raised at
      construction time so a caller that does not inspect `.degenerate`
      still sees this on stderr/pytest capture rather than getting a
      silently-halved reset count.
    - `t > num_shards`: turn `t` never occurs in a trajectory with only
      `num_shards` turns, so again only the last-turn reset fires. Same
      `self.degenerate = True` treatment, `self.degenerate_reason =
      "t_exceeds_num_shards"`. This is the same one-reset collapse as the
      equality case, not a third distinct behavior.
    """

    t: int
    num_shards: int
    name: str = ""
    turns: tuple[int, ...] = field(init=False)
    degenerate: bool = field(init=False, default=False)
    degenerate_reason: str | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        if self.num_shards < 1:
            raise ValueError(f"num_shards must be >= 1, got {self.num_shards}")
        if self.t < 1:
            raise ValueError(f"t must be >= 1, got {self.t}")

        if self.t == self.num_shards:
            self.degenerate = True
            self.degenerate_reason = "t_equals_num_shards"
        elif self.t > self.num_shards:
            self.degenerate = True
            self.degenerate_reason = "t_exceeds_num_shards"

        if self.degenerate:
            warnings.warn(
                f"FixedTAndLastController(t={self.t}, num_shards={self.num_shards}) is degenerate "
                f"({self.degenerate_reason}): only one reset (at turn {self.num_shards}) fires "
                "instead of two.",
                UserWarning,
                stacklevel=2,
            )
            self.turns = (self.num_shards,)
        else:
            self.turns = tuple(sorted((self.t, self.num_shards)))

        if not self.name:
            self.name = f"fixed_t{self.t}_and_last"

    def next_u_remind(self, turn: int, history: list[dict[str, Any]]) -> int:
        return int(turn in self.turns)


@dataclass
class RandSchedTAndLastController:
    """Reset on one turn drawn uniformly from `{1, ..., num_shards - 1}`
    AND on this item's own last turn (`num_shards`): the random-allocation,
    k=2, mode-B opponent for `KoopmanMPCController(forced_last_reset=True)`.

    Per-item seeding is copied verbatim from `control.RandomScheduleController`
    (control.py:120): the draw happens once, in `__post_init__`, from
    `random.Random(self.seed)`, so the same `(num_shards, seed)` pair always
    produces the same schedule -- construct the controller twice with the
    same arguments and `.t`/`.turns` compare equal.

    `num_shards == 1` is a degenerate edge case handled explicitly rather
    than left to raise from an empty `range(1, 1)`: there is no turn in
    `{1, ..., T-1}` to draw from, so only the (also turn-1) last-turn reset
    fires -- one reset instead of two. `self.degenerate = True`,
    `self.degenerate_reason = "num_shards_equals_1"`, and a `UserWarning` is
    raised at construction time, mirroring `FixedTAndLastController`'s
    degenerate-collapse reporting rather than inventing a different
    convention for the same "two resets collapsed to one" situation.
    """

    num_shards: int
    seed: int
    name: str = ""
    t: int | None = field(init=False, default=None)
    turns: tuple[int, ...] = field(init=False)
    degenerate: bool = field(init=False, default=False)
    degenerate_reason: str | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        if self.num_shards < 1:
            raise ValueError(f"num_shards must be >= 1, got {self.num_shards}")

        if self.num_shards == 1:
            self.degenerate = True
            self.degenerate_reason = "num_shards_equals_1"
            warnings.warn(
                f"RandSchedTAndLastController(num_shards=1, seed={self.seed}) is degenerate "
                "(num_shards_equals_1): there is no turn in {1,...,T-1} to draw from, so only "
                "the last-turn reset (turn 1) fires instead of two.",
                UserWarning,
                stacklevel=2,
            )
            self.turns = (1,)
        else:
            rng = random.Random(self.seed)
            self.t = rng.choice(range(1, self.num_shards))  # U{1, ..., T-1}
            self.turns = tuple(sorted((self.t, self.num_shards)))

        if not self.name:
            self.name = "randsched_t_and_last"

    def next_u_remind(self, turn: int, history: list[dict[str, Any]]) -> int:
        return int(turn in self.turns)
