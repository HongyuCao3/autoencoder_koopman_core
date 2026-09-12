"""Tests for src/persona_drift/sequor_closed_loop.py — the S3 generation loop.

The loop is where a controller can quietly cheat: by spending past its budget,
by acting at turn 1, by carrying a stale state through a turn the judge could
not read, or by letting an open-loop arm see the state. Each of those gets a
test that fails loudly, because every one of them would leave the artifact
looking exactly like a clean run.
"""

from __future__ import annotations

import pytest

from persona_drift.sequor_bank import SequorItem
from persona_drift.sequor_closed_loop import (
    budget_accounting, closed_loop_action, run_closed_loop_arm)
from persona_drift.sequor_controllers import (
    NONE, ActionModel, action_cost, best_fixed_schedule, plan_table)
from persona_drift.sequor_trajectory import BranchArmConfig

K = 3
N_TURNS = 6
LATE_FROM = 4

STATEFUL = ActionModel(probabilities={
    (False, False, 0): 0.20, (True, False, 0): 0.90,
    (False, True, 1): 0.80, (True, False, 1): 0.85,
    (False, True, 2): 0.60, (True, False, 2): 0.85,
    (False, True, 3): 0.45, (True, True, 3): 0.95,
}, counts={}, k=K)


def _item(item_id: str = "item0") -> SequorItem:
    constraints = [f"constraint {j} text here" for j in range(K)]
    preamble = "In all your responses, make sure to adhere to these rules:"
    block = preamble + "\n" + "\n".join(f"{j + 1}. {c}" for j, c in enumerate(constraints))
    return SequorItem(conversation_id=item_id, tuple_id="t0", constraints=constraints,
                      constraint_ids=[f"id{j}" for j in range(K)], preamble=preamble,
                      constraint_block=block, turns=[f"turn {t}" for t in range(N_TURNS)],
                      n_turns_available=N_TURNS)


def _words(text: str) -> int:
    return len(text.split())


def _config(seed: int = 0) -> BranchArmConfig:
    return BranchArmConfig(model_id="test-model", n_turns=N_TURNS, seed=seed,
                           max_new_tokens=64, constraints_in_system=True)


def _harness(items, verdict_plan):
    """A generator that echoes the turn and a judge that replays a script.

    `verdict_plan(turn, item_id)` returns the k verdicts for that turn, so a
    test can make a constraint break exactly when it wants to and watch what
    the controller does about it.
    """

    clock = {"turn": 0}

    def generate_batch(conversations):
        clock["turn"] += 1
        return [{"text": f"reply at turn {clock['turn']}", "finish_reason": "stop",
                 "n_output_tokens": 5} for _ in conversations]

    def judge_batch(pairs):
        out = []
        for i, item in enumerate(items):
            out.extend(verdict_plan(clock["turn"], item.conversation_id))
        assert len(out) == len(pairs)
        return out

    return generate_batch, judge_batch


def test_no_arm_acts_at_turn_one():
    """Turn 1 states the constraints; a reminder there is not a distinct
    action, and `sequor_bank.user_message` refuses it."""

    item = _item()
    for policy in ("koopman_mpc", "greedy_targeted", "best_fixed_schedule", "equal_cost_random"):
        assert closed_loop_action(policy, item, (False, False, False), 10_000, 1, None,
                                  [1] * N_TURNS, _words) == NONE


def test_a_blind_turn_plays_nothing_instead_of_a_stale_state():
    """When the in-loop judge does not parse, the state is None. Carrying the
    previous reading forward would let the controller act on a stale state and
    would hide the blindness from the artifact."""

    item = _item()
    table = plan_table(item, STATEFUL, 10_000, _words, N_TURNS, LATE_FROM)
    assert closed_loop_action("koopman_mpc", item, None, 10_000, 3, table, None, _words) == NONE
    assert closed_loop_action("greedy_targeted", item, None, 10_000, 3, None, None, _words) == NONE


def test_blind_turns_are_counted_in_the_rows():
    items = [_item()]
    generate, judge = _harness(items, lambda turn, _: [None, True, True] if turn == 3
                               else [True, True, True])
    table = plan_table(items[0], STATEFUL, 10_000, _words, N_TURNS, LATE_FROM)
    rows = run_closed_loop_arm(items, generate, judge, _config(), "run", "koopman_mpc",
                               budgets={"item0": 10_000}, tables={"item0": table}, schedules={},
                               count_tokens=_words)
    blind = [row for row in rows if row["y_inloop"] is None]
    assert [row["turn"] for row in blind] == [3]
    assert rows[3]["state_before_action"] is None  # turn 4 acts on no state
    assert rows[3]["n_named"] == 0
    assert all(row["n_blind_turns"] == 1 for row in rows)


def test_the_budget_is_never_exceeded_and_is_accounted_exactly():
    items = [_item()]
    broken_forever = lambda turn, _: [False, False, False]  # noqa: E731
    generate, judge = _harness(items, broken_forever)
    blanket = action_cost(items[0], (0, 1, 2), _words)
    budget = 2 * blanket
    table = plan_table(items[0], STATEFUL, budget, _words, N_TURNS, LATE_FROM)
    rows = run_closed_loop_arm(items, generate, judge, _config(), "run", "koopman_mpc",
                               budgets={"item0": budget}, tables={"item0": table}, schedules={},
                               count_tokens=_words)
    assert sum(row["inserted_tokens"] for row in rows) <= budget
    assert rows[-1]["tokens_left_after"] == budget - sum(row["inserted_tokens"] for row in rows)
    assert all(row["tokens_left_after"] >= 0 for row in rows)

    spend = budget_accounting(rows, ["koopman_mpc"])
    assert spend["koopman_mpc"]["overspent_items"] == []
    assert spend["koopman_mpc"]["tokens_spent_total"] == sum(r["inserted_tokens"] for r in rows)


def test_a_closed_loop_arm_targets_what_the_judge_says_is_broken():
    items = [_item()]
    generate, judge = _harness(items, lambda turn, _: [False, True, True])
    rows = run_closed_loop_arm(items, generate, judge, _config(), "run", "greedy_targeted",
                               budgets={"item0": 10_000}, tables={}, schedules={},
                               count_tokens=_words)
    assert rows[0]["action_named"] == []  # turn 1 takes no action
    assert all(row["action_named"] == [0] for row in rows[1:])
    assert all(row["n_named"] == 1 for row in rows[1:])


def test_an_open_loop_arm_ignores_the_state_entirely():
    """Two runs whose judges disagree about everything must produce the same
    action sequence under a fixed schedule. If they do not, the 'open-loop' arm
    is reading the state."""

    items = [_item()]
    schedule = [0, 1, 0, 1, 0, 0]
    actions = []
    for plan in (lambda turn, _: [True, True, True], lambda turn, _: [False, False, False]):
        generate, judge = _harness(items, plan)
        rows = run_closed_loop_arm(items, generate, judge, _config(), "run", "best_fixed_schedule",
                                   budgets={"item0": 10_000}, tables={},
                                   schedules={"item0": schedule}, count_tokens=_words)
        actions.append([row["n_named"] for row in rows])
    assert actions[0] == actions[1]
    assert [int(bool(n)) for n in actions[0]] == schedule


def test_an_open_loop_arm_can_only_play_none_or_blanket():
    """'Name the broken ones' is undefined without reading the state, so the
    open-loop action set has two members. This is the asymmetry under test, and
    a schedule arm that somehow emitted a partial naming would be a bug."""

    items = [_item()]
    generate, judge = _harness(items, lambda turn, _: [False, True, True])
    rows = run_closed_loop_arm(items, generate, judge, _config(), "run", "equal_cost_random",
                               budgets={"item0": 10_000}, tables={},
                               schedules={"item0": [0, 1, 1, 0, 1, 0]}, count_tokens=_words)
    assert {row["n_named"] for row in rows} <= {0, K}


def test_an_open_loop_arm_skips_an_action_it_cannot_afford():
    items = [_item()]
    generate, judge = _harness(items, lambda turn, _: [True, True, True])
    blanket = action_cost(items[0], (0, 1, 2), _words)
    rows = run_closed_loop_arm(items, generate, judge, _config(), "run", "best_fixed_schedule",
                               budgets={"item0": blanket}, tables={},
                               schedules={"item0": [0, 1, 1, 1, 0, 0]}, count_tokens=_words)
    assert sum(row["n_named"] > 0 for row in rows) == 1
    assert sum(row["inserted_tokens"] for row in rows) == blanket


def test_the_inloop_verdicts_are_stored_under_their_own_name():
    """The reporting judge writes `followed`; this loop writes
    `followed_inloop`. One field holding both is how a selection signal ends up
    in a reported number."""

    items = [_item()]
    generate, judge = _harness(items, lambda turn, _: [True, False, True])
    rows = run_closed_loop_arm(items, generate, judge, _config(), "run", "greedy_targeted",
                               budgets={"item0": 10_000}, tables={}, schedules={},
                               count_tokens=_words)
    assert rows[0]["followed_inloop"] == [True, False, True]
    assert rows[0]["y_inloop"] == pytest.approx(2 / 3)
    assert "followed" not in rows[0]
    assert "y_graded" not in rows[0]


def test_items_run_in_lockstep_and_keep_their_own_budgets():
    items = [_item("a"), _item("b")]
    generate, judge = _harness(items, lambda turn, item_id: [False, True, True] if item_id == "a"
                               else [True, True, True])
    rows = run_closed_loop_arm(items, generate, judge, _config(), "run", "greedy_targeted",
                               budgets={"a": 10_000, "b": 10_000}, tables={}, schedules={},
                               count_tokens=_words)
    by_item = {item_id: [r for r in rows if r["item_id"] == item_id] for item_id in ("a", "b")}
    assert all(r["n_named"] == 1 for r in by_item["a"][1:])
    assert all(r["n_named"] == 0 for r in by_item["b"])
    assert len(by_item["a"]) == len(by_item["b"]) == N_TURNS
