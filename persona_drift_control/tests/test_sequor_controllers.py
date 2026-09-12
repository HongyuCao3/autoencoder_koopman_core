"""Tests for src/persona_drift/sequor_controllers.py — S3's policies and the
model they plan with.

Two controls run on every claim, the pattern this line settled on after the
schedule degeneracies: a MEMORYLESS model, on which a closed loop must buy
EXACTLY zero over the best fixed schedule, and a model with real state
dependence, on which the same machinery must find a positive gap. A planner
that reports a gap on the memoryless model is manufacturing one; a planner that
reports none on the second is blind to the thing S3 exists to measure.
"""

from __future__ import annotations

import itertools

import pytest

from persona_drift.sequor_bank import SequorItem
from persona_drift.sequor_controllers import (
    NONE, ActionModel, action_cost, best_fixed_schedule, candidate_actions, equal_cost_random_schedule,
    fit_action_model, greedy_action, greedy_action as _greedy, joint_transition, mpc_action,
    observations_from_branches, observations_from_rows, open_loop_value, plan_table)

K = 3
N_TURNS = 8
LATE_FROM = 6


def _item(n_words_per_constraint: int = 4) -> SequorItem:
    constraints = [" ".join(f"c{j}w{w}" for w in range(n_words_per_constraint)) for j in range(K)]
    preamble = "In all your responses, make sure to adhere to these rules:"
    block = preamble + "\n" + "\n".join(f"{j + 1}. {c}" for j, c in enumerate(constraints))
    return SequorItem(conversation_id="item0", tuple_id="t0", constraints=constraints,
                      constraint_ids=[f"id{j}" for j in range(K)], preamble=preamble,
                      constraint_block=block, turns=[f"turn {t}" for t in range(N_TURNS)],
                      n_turns_available=N_TURNS)


def _words(text: str) -> int:
    return len(text.split())


def _model(probabilities: dict) -> ActionModel:
    return ActionModel(probabilities=probabilities, counts={cell: 999 for cell in probabilities}, k=K)


MEMORYLESS = _model({
    (prev, named, n): 0.6
    for prev in (False, True) for named, n in ((False, 0), (True, 1), (True, 2), (True, 3),
                                               (False, 1), (False, 2), (True, 3))
})

STATEFUL = _model({
    (False, False, 0): 0.20, (True, False, 0): 0.90,
    (False, True, 1): 0.80, (True, False, 1): 0.85,
    (False, True, 2): 0.60, (True, False, 2): 0.85,
    (False, True, 3): 0.45, (True, True, 3): 0.95,
})


def test_candidate_actions_stay_inside_the_measured_cells():
    """The whole action set must be plannable against measured numbers. A
    proper subset of the broken set would put a broken constraint in a
    `not named` cell while something else was named -- never observed."""

    for followed in itertools.product((False, True), repeat=K):
        for action in candidate_actions(followed, K):
            for j in range(K):
                STATEFUL.probability(followed[j], j in set(action), len(action))


def test_candidate_actions_are_none_all_broken_and_blanket():
    assert candidate_actions((True, True, True), K) == [NONE, (0, 1, 2)]
    assert candidate_actions((False, True, True), K) == [NONE, (0,), (0, 1, 2)]
    assert candidate_actions((False, False, True), K) == [NONE, (0, 1), (0, 1, 2)]
    assert candidate_actions((False, False, False), K) == [NONE, (0, 1, 2)]


def test_a_memoryless_model_gives_the_closed_loop_exactly_nothing():
    """The negative control. When the next state does not depend on the action
    or the current one, every policy at every budget lands in the same place,
    and the planner must report that rather than a small positive number."""

    item = _item()
    budget = 3 * action_cost(item, (0, 1, 2), _words)
    table = plan_table(item, MEMORYLESS, budget, _words, N_TURNS, LATE_FROM)
    for turn in range(2, N_TURNS + 1):
        for state in itertools.product((False, True), repeat=K):
            values = {
                action: sum(p * (sum(nxt) / K)
                            for nxt, p in joint_transition(MEMORYLESS.transition(state, action)))
                for action in candidate_actions(state, K)
            }
            assert len(set(round(v, 12) for v in values.values())) == 1
        assert table["value"][turn][(True,) * K][-1] == pytest.approx(
            table["value"][turn][(False,) * K][-1])


def test_a_stateful_model_makes_the_best_action_depend_on_the_state():
    """The positive control, and the mechanism S3 is built to test: with one
    constraint broken the cheap targeted action is best, with none broken it is
    not even available, so the best action is a function of the state and not
    of the turn index."""

    item = _item()
    budget = 20 * action_cost(item, (0, 1, 2), _words)
    table = plan_table(item, STATEFUL, budget, _words, N_TURNS, LATE_FROM)
    one_broken = mpc_action(table, 2, (False, True, True), budget)
    none_broken = mpc_action(table, 2, (True, True, True), budget)
    assert one_broken == (0,)
    assert none_broken != one_broken


def test_the_planner_never_plans_a_spend_the_budget_cannot_cover():
    item = _item()
    blanket = action_cost(item, (0, 1, 2), _words)
    table = plan_table(item, STATEFUL, blanket, _words, N_TURNS, LATE_FROM)
    for tokens_left in range(blanket + 1):
        for state in itertools.product((False, True), repeat=K):
            action = mpc_action(table, 2, state, tokens_left)
            assert action_cost(item, action, _words) <= tokens_left


def test_turn_one_takes_no_action_and_the_table_stops_at_the_last_real_turn():
    """The 2026-09-11 off-by-one made a rollout run to a turn the arms do not
    have. The table must be defined on turns 2..n_turns and nowhere else."""

    item = _item()
    table = plan_table(item, STATEFUL, 500, _words, N_TURNS, LATE_FROM)
    state = (False, True, True)
    assert all(table["value"][N_TURNS + 1][state][level] == 0.0
               for level in range(table["n_levels"]))
    assert table["value"][N_TURNS][state][-1] > 0.0
    assert table["value"][1][state][-1] == 0.0


def test_the_late_window_is_the_only_thing_that_earns():
    """`y` at turns before `late_from` must not enter the objective: the arm's
    primary quantity is the late window, and a planner optimising the whole
    trajectory would be answering a different question."""

    item = _item()
    early = plan_table(item, STATEFUL, 500, _words, N_TURNS, late_from=N_TURNS + 1)
    assert early["value"][2][(False, True, True)][-1] == 0.0


def test_open_loop_value_counts_exactly_the_late_window_turns():
    flat = _model({(prev, named, n): 1.0 if prev else 0.0
                   for prev in (False, True)
                   for named, n in ((False, 0), (True, 3))})
    assert open_loop_value(flat, [0] * N_TURNS, LATE_FROM) == pytest.approx(1.0)


def test_best_fixed_schedule_matches_brute_force():
    item = _item()
    blanket = action_cost(item, (0, 1, 2), _words)
    budget = 3 * blanket
    found = best_fixed_schedule(item, STATEFUL, budget, _words, N_TURNS, LATE_FROM)
    assert sum(found) <= 3 and found[0] == 0

    best = max(
        ([0] + [1 if t in set(turns) else 0 for t in range(2, N_TURNS + 1)]
         for m in range(4) for turns in itertools.combinations(range(2, N_TURNS + 1), m)),
        key=lambda schedule: open_loop_value(STATEFUL, schedule, LATE_FROM))
    assert open_loop_value(STATEFUL, found, LATE_FROM) == pytest.approx(
        open_loop_value(STATEFUL, best, LATE_FROM), abs=1e-6)


def test_greedy_falls_back_to_nothing_rather_than_a_partial_naming():
    item = _item()
    state = (False, False, True)
    assert greedy_action(item, state, 10_000, _words) == (0, 1)
    assert greedy_action(item, state, 1, _words) == NONE


def test_equal_cost_random_spends_the_budget_and_never_acts_at_turn_one():
    import random
    item = _item()
    blanket = action_cost(item, (0, 1, 2), _words)
    schedule = equal_cost_random_schedule(item, 3 * blanket, _words, N_TURNS, random.Random(0))
    assert schedule[0] == 0
    assert sum(schedule) == 3
    assert len(schedule) == N_TURNS


def test_observations_from_schedule_rows_pair_each_turn_with_the_one_before():
    rows = [
        {"trajectory_id": "a", "turn": 1, "u_remind": 0, "followed": [True, False, True]},
        {"trajectory_id": "a", "turn": 2, "u_remind": 1, "followed": [True, True, False]},
    ]
    records = observations_from_rows(rows, lambda row: range(3) if row["u_remind"] else ())
    assert len(records) == 3
    assert records[1] == {"prev_followed": False, "named": True, "n_named": 3, "followed": True}


def test_a_turn_with_an_unparsed_verdict_contributes_no_transition():
    rows = [
        {"trajectory_id": "a", "turn": 1, "u_remind": 0, "followed": [True, None, True]},
        {"trajectory_id": "a", "turn": 2, "u_remind": 1, "followed": [True, True, False]},
    ]
    assert observations_from_rows(rows, lambda row: range(3) if row["u_remind"] else ()) == []


def test_branch_observations_take_their_previous_state_from_the_base_trajectory():
    rows = [{"trajectory_id": "b", "followed": [True, True, False]}]
    meta = {"b": {"branch": "targeted_remind", "turn": 5, "violated_indices": [0],
                  "base_trajectory_id": "base"}}
    records = observations_from_branches(rows, meta, {("base", 4): [False, True, True]})
    assert records[0] == {"prev_followed": False, "named": True, "n_named": 1, "followed": True}
    assert records[1]["named"] is False and records[1]["n_named"] == 1


def test_fit_refuses_a_cell_too_thin_to_plan_against():
    observations = [{"prev_followed": True, "named": False, "n_named": 0, "followed": True}] * 5
    with pytest.raises(ValueError, match="fewer than"):
        fit_action_model(observations, k=K, min_count=30)
