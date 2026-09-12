"""S3's closed-loop policies and the model they plan with
(docs/experiments/constraint_retention_plan.md section 6, arm table rewritten
2026-09-12 after the targeted-reminder gate passed).

WHY THE ACTION SPACE IS THE WAY IT IS. S2 twice found that the best SCHEDULE is
the same for every trajectory -- under a scalar operator and under a
per-constraint binary state, and again when the objective was changed to a
priced threshold. All three degeneracies share one cause: when the action is a
single switch, the state can only scale what a reminder is worth, never change
which turn is the best one to spend it on. The targeted-branch arm (15792569)
measured what changes when the action can NAME a subset: restating only the
broken constraints recovers them at +0.0748 over a blanket restatement
(1.47x MDE), and loses -0.1059 on the constraints it leaves unnamed. Both
numbers depend on WHICH constraints are broken, which is not knowable in
advance -- so the action set available to an open-loop schedule and the one
available to a closed loop are genuinely different sets.

That asymmetry is the experiment, not a handicap. An open-loop arm may play
`none` or `blanket` only, because "name the broken ones" is undefined without
reading the state. It is stated here so no one later reads the restriction as
a baseline that was starved.

WHAT THE MODEL IS. Per constraint j, a binary state (followed / not) and a
transition that depends on three things the data actually separates:

    p(followed_j at t | followed_j at t-1, whether j was named at t, how many
                        constraints were named at t)

The third term is not decoration: blanket and targeted BOTH name a broken
constraint, and the only thing that differs is how many others were named
alongside it -- so the whole measured effect of targeting lives in that term.
Constraints are conditionally independent given the action, so the joint state
transition factorises and the dynamic program below is exact over all 8 states
rather than approximate.

CELLS THAT ARE NEVER OBSERVED ARE NEVER USED. `(followed, named, n_named<3)`
-- a constraint that is already held and is named without the other two --
occurs in no arm this line has run, because a targeted reminder names only
broken constraints. `candidate_actions` therefore refuses to propose an action
that would enter that cell, and `ActionModel.probability` raises rather than
extrapolating into it. Filling it with a fitted surface would be inventing the
one number the whole comparison turns on.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

from .sequor_bank import SequorItem, targeted_constraint_block

NONE: tuple[int, ...] = ()
POLICIES = ("koopman_mpc", "greedy_targeted", "best_fixed_schedule", "equal_cost_random")
OPEN_LOOP_POLICIES = ("best_fixed_schedule", "equal_cost_random")


def state_key(followed: tuple[bool, ...]) -> tuple[bool, ...]:
    return tuple(bool(b) for b in followed)


def broken_indices(followed: tuple[bool, ...]) -> tuple[int, ...]:
    return tuple(i for i, b in enumerate(followed) if not b)


def candidate_actions(followed: tuple[bool, ...], k: int) -> list[tuple[int, ...]]:
    """The actions a controller reading this state may play.

    Exactly three, and no more: `none`, name ALL the broken constraints, and
    the blanket action that names all k.

    PROPER SUBSETS OF THE BROKEN SET ARE DELIBERATELY ABSENT. Naming one of two
    broken constraints would leave the other broken-and-unnamed while one
    constraint was named -- cell `(False, False, 1)` -- and no arm on this line
    has ever produced that cell, because the targeted arm always named every
    constraint the judge called broken. `ActionModel.probability` refuses it
    rather than extrapolating, so a controller offered that action would be
    planning against a number nobody measured. The cheaper-but-partial action
    may well be worth something; buying it means running an arm that produces
    the cell, not inventing it here.
    """

    broken = broken_indices(followed)
    everything = tuple(range(k))
    actions = [NONE]
    if broken and broken != everything:
        actions.append(broken)
    actions.append(everything)
    return actions


def action_cost(item: SequorItem, action: tuple[int, ...], count_tokens) -> int:
    """The insertion cost of an action, in the same unit the budget is held in.

    Computed from the text that will actually be sent rather than from a table
    of averages: the constraints differ in length across items by more than the
    difference between a targeted and a blanket block does, so a per-item
    average would price the action wrong on exactly the items where the budget
    binds hardest.
    """

    if not action:
        return 0
    return count_tokens(targeted_constraint_block(item, action))


@dataclass(frozen=True)
class ActionModel:
    """`p[(prev_followed, named, n_named)]` -> probability the constraint is
    followed at the next turn, plus the cell counts it was estimated from."""

    probabilities: dict[tuple[bool, bool, int], float]
    counts: dict[tuple[bool, bool, int], int]
    k: int

    def probability(self, prev_followed: bool, named: bool, n_named: int) -> float:
        cell = (bool(prev_followed), bool(named), int(n_named))
        if cell not in self.probabilities:
            raise KeyError(
                f"cell {cell} was never observed in any arm on this line, so it has no estimate. "
                f"An action reaching it would be planned against an invented number.")
        return self.probabilities[cell]

    def transition(self, followed: tuple[bool, ...], action: tuple[int, ...]) -> list[float]:
        """Per-constraint probability of being followed after `action`."""

        named = set(action)
        return [self.probability(followed[j], j in named, len(action)) for j in range(self.k)]


def fit_action_model(observations: list[dict], k: int = 3, min_count: int = 30) -> ActionModel:
    """Cell means, with the cells too thin to support one refused rather than smoothed.

    Each observation is one constraint at one turn: what it was at t-1, whether
    the action named it, how many the action named, and what it became. No
    pooling across cells and no functional form -- the point of the targeted
    arm was to measure the `n_named` dependence rather than assume it has a
    shape, and a logistic link fitted to five cells would put that shape back.
    """

    totals: dict[tuple[bool, bool, int], list[int]] = {}
    for record in observations:
        cell = (bool(record["prev_followed"]), bool(record["named"]), int(record["n_named"]))
        bucket = totals.setdefault(cell, [0, 0])
        bucket[0] += int(bool(record["followed"]))
        bucket[1] += 1
    thin = {cell: n for cell, (_, n) in totals.items() if n < min_count}
    if thin:
        raise ValueError(
            f"cells with fewer than {min_count} observations: {thin}. Estimating a transition from "
            f"a handful of rows and then planning against it is how a controller ends up optimising "
            f"noise; drop the action that needs the cell or get more rows.")
    return ActionModel(
        probabilities={cell: hits / n for cell, (hits, n) in totals.items()},
        counts={cell: n for cell, (_, n) in totals.items()},
        k=k,
    )


def observations_from_rows(rows: list[dict], named_of) -> list[dict]:
    """Turn scored trajectory rows into per-constraint transition records.

    `named_of(row)` returns the indices that row's action named. Kept as a
    callback because the arms record it differently: the schedule arms carry a
    0/1 `u_remind` that means "all k or none", while the branch arm records the
    explicit `violated_indices` it targeted.
    """

    by_trajectory: dict[str, dict[int, dict]] = {}
    for row in rows:
        by_trajectory.setdefault(row["trajectory_id"], {})[row["turn"]] = row

    records = []
    for _, turns in sorted(by_trajectory.items()):
        for turn, row in sorted(turns.items()):
            previous = turns.get(turn - 1)
            if previous is None or previous.get("followed") is None or row.get("followed") is None:
                continue
            if any(b is None for b in previous["followed"]) or any(b is None for b in row["followed"]):
                continue
            named = set(named_of(row))
            for j, became in enumerate(row["followed"]):
                records.append({
                    "prev_followed": previous["followed"][j], "named": j in named,
                    "n_named": len(named), "followed": became,
                })
    return records


def joint_transition(per_constraint: list[float]) -> list[tuple[tuple[bool, ...], float]]:
    """The 2^k next states and their probabilities. Enumerated rather than
    sampled: k=3 makes it eight terms, and a sampled rollout would put Monte
    Carlo noise inside a comparison whose whole question is a 0.06-sized gap."""

    out = []
    for combo in itertools.product((False, True), repeat=len(per_constraint)):
        probability = 1.0
        for followed, p in zip(combo, per_constraint):
            probability *= p if followed else (1.0 - p)
        out.append((combo, probability))
    return out


def _reward(state: tuple[bool, ...]) -> float:
    return sum(state) / len(state)


def plan_table(item: SequorItem, model: ActionModel, budget: int, count_tokens,
               n_turns: int, late_from: int, n_buckets: int = 40) -> dict:
    """Exact dynamic program over (turn, state, remaining budget).

    The value is the expected mean of `y` over the LATE WINDOW -- the same
    window, on the same readout, that the arm's primary quantity is computed
    on. A controller optimising anything else would be answering a question the
    result table does not ask.

    Budget is bucketed and every action's cost is rounded UP into buckets, so
    the plan can only ever be conservative: it never commits to a spend the
    exact token budget cannot cover. The arm's own accounting is in exact
    tokens; this quantisation lives only inside the planner.

    Turn indexing, which this line has already got wrong once (the 2026-09-11
    off-by-one that made a rollout run to a turn the arms do not have): the
    action at turn `t` is chosen from the state left by turn `t-1`, and the
    reward it earns is the state turn `t` itself produces. Turn 1 takes no
    action, so the table starts at turn 2 and ends at `n_turns`.
    """

    width = max(1, -(-budget // n_buckets))
    n_levels = -(-budget // width) + 1
    states = list(itertools.product((False, True), repeat=model.k))

    costs: dict[tuple[bool, ...], dict[tuple[int, ...], tuple[int, int]]] = {}
    for state in states:
        costs[state] = {}
        for action in candidate_actions(state, model.k):
            exact = action_cost(item, action, count_tokens)
            costs[state][action] = (exact, -(-exact // width))

    value = [{state: [0.0] * n_levels for state in states} for _ in range(n_turns + 2)]
    choice = [{state: [NONE] * n_levels for state in states} for _ in range(n_turns + 2)]

    for turn in range(n_turns, 1, -1):
        earns = turn >= late_from
        for state in states:
            for level in range(n_levels):
                best, best_action = None, NONE
                for action, (_, in_buckets) in costs[state].items():
                    if in_buckets > level:
                        continue
                    remaining = level - in_buckets
                    total = 0.0
                    for nxt, probability in joint_transition(model.transition(state, action)):
                        if probability == 0.0:
                            continue
                        total += probability * ((_reward(nxt) if earns else 0.0)
                                                + value[turn + 1][nxt][remaining])
                    if best is None or total > best:
                        best, best_action = total, action
                value[turn][state][level] = 0.0 if best is None else best
                choice[turn][state][level] = best_action

    return {"choice": choice, "value": value, "bucket_width": width, "n_levels": n_levels,
            "costs": costs, "budget": budget, "late_from": late_from, "n_turns": n_turns}


def mpc_action(table: dict, turn: int, state: tuple[bool, ...], tokens_left: int) -> tuple[int, ...]:
    """The planned action, then re-checked against the EXACT budget.

    The table is indexed by a rounded-up bucket, so its answer is already
    affordable; the exact check is here because the budget the arm reports
    spending must be the budget the arm actually enforced, not the planner's
    rounding of it.
    """

    level = min(tokens_left // table["bucket_width"], table["n_levels"] - 1)
    action = table["choice"][turn][state][level]
    if table["costs"][state][action][0] > tokens_left:
        return NONE
    return action


def greedy_action(item: SequorItem, state: tuple[bool, ...], tokens_left: int,
                  count_tokens) -> tuple[int, ...]:
    """Name every constraint the judge currently calls broken, while the budget
    lasts, and nothing otherwise.

    This arm exists to split the closed loop in two. It reads the state and it
    spends on what is broken, but it has no model and no plan, so the gap
    between it and `koopman_mpc` is what the fitted operator is worth -- as
    opposed to what merely reacting is worth. Without it a positive MPC result
    cannot tell those two apart, and that is the first question a reader asks.

    It falls back to `none` rather than to a partial naming for the same reason
    `candidate_actions` offers no proper subsets: a partial naming enters a
    cell no arm has measured. Spending nothing is a measured action.
    """

    broken = broken_indices(state)
    if broken and action_cost(item, broken, count_tokens) <= tokens_left:
        return broken
    return NONE


def open_loop_value(model: ActionModel, schedule: list[int], late_from: int) -> float:
    """Expected mean late-window `y` of a state-blind schedule.

    A schedule that cannot read the state can only play `none` or `blanket`,
    and under both of those every constraint is treated alike -- so the belief
    collapses to one number, the probability that any given constraint is
    followed, and it evolves deterministically. That is why the open-loop
    optimum below is a one-dimensional problem and does not need the 8-state
    table.
    """

    everything = tuple(range(model.k))
    q = 1.0
    total, n = 0.0, 0
    for turn, u in enumerate(schedule, start=1):
        action = everything if u else NONE
        p_from_followed = model.probability(True, bool(u), len(action))
        p_from_broken = model.probability(False, bool(u), len(action))
        q = q * p_from_followed + (1.0 - q) * p_from_broken
        if turn >= late_from:
            total += q
            n += 1
    return total / n if n else 0.0


def best_fixed_schedule(item: SequorItem, model: ActionModel, budget: int, count_tokens,
                        n_turns: int, late_from: int, n_bins: int = 400) -> list[int]:
    """The best open-loop schedule at this budget, found by DP on the collapsed
    belief rather than by enumerating C(19, m) schedules.

    The belief `q` is discretised into bins; the reported schedule is then
    re-scored with `open_loop_value` at full precision, so the number that goes
    into the report is never the discretised one.
    """

    blanket_cost = action_cost(item, tuple(range(model.k)), count_tokens)
    max_reminders = min(n_turns - 1, budget // blanket_cost) if blanket_cost else n_turns - 1
    everything = tuple(range(model.k))
    step = {
        0: (model.probability(True, False, 0), model.probability(False, False, 0)),
        1: (model.probability(True, True, len(everything)), model.probability(False, True, len(everything))),
    }

    # Forward pass over (turn, reminders used, binned belief), keeping for each
    # cell the best accumulated late-window total and the path that achieved it.
    start = (0, _bin(1.0, n_bins))
    frontier = {start: (0.0, 1.0, [])}
    for turn in range(1, n_turns + 1):
        nxt: dict[tuple[int, int], tuple[float, float, list[int]]] = {}
        for (used, _), (total, q, path) in frontier.items():
            for u in (0, 1):
                if turn == 1 and u:  # turn 1 states the constraints; reminding is not an action
                    continue
                if u and used + 1 > max_reminders:
                    continue
                p_followed, p_broken = step[u]
                q_next = q * p_followed + (1.0 - q) * p_broken
                total_next = total + (q_next if turn >= late_from else 0.0)
                key = (used + u, _bin(q_next, n_bins))
                if key not in nxt or total_next > nxt[key][0]:
                    nxt[key] = (total_next, q_next, path + [u])
        frontier = nxt

    _, _, best = max(frontier.values(), key=lambda cell: cell[0])
    return best


def _bin(q: float, n_bins: int) -> int:
    return min(n_bins - 1, max(0, int(q * n_bins)))


def equal_cost_random_schedule(item: SequorItem, budget: int, count_tokens, n_turns: int,
                               rng) -> list[int]:
    """The same token spend, placed at random. State-blind by construction, and
    it plays the blanket action for the same reason the fixed schedule does."""

    blanket_cost = action_cost(item, tuple(range(3)), count_tokens)
    n_reminders = min(n_turns - 1, budget // blanket_cost) if blanket_cost else 0
    turns = sorted(rng.sample(range(2, n_turns + 1), n_reminders))
    return [1 if turn in set(turns) else 0 for turn in range(1, n_turns + 1)]


def observations_from_branches(rows: list[dict], meta: dict[str, dict],
                               base_verdicts: dict[tuple[str, int], list]) -> list[dict]:
    """Transition records from the one-step branch arm.

    The branch rows are single turns, not trajectories, so the state at t-1
    comes from the base trajectory the branch hangs off rather than from a
    previous row of the same id. These are the only rows that populate the
    `named` cells at `n_named < k` -- the schedule arms only ever name all k or
    none -- which is exactly why the targeted arm had to be run before the arm
    table could be written.
    """

    records = []
    for row in rows:
        record = meta[row["trajectory_id"]]
        previous = base_verdicts.get((record["base_trajectory_id"], record["turn"] - 1))
        if previous is None or row.get("followed") is None:
            continue
        if any(b is None for b in previous) or any(b is None for b in row["followed"]):
            continue
        named = set(record["violated_indices"] if record["branch"] == "targeted_remind"
                    else range(len(row["followed"])))
        for j, became in enumerate(row["followed"]):
            records.append({"prev_followed": previous[j], "named": j in named,
                            "n_named": len(named), "followed": became})
    return records
