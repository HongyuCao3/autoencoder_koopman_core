"""Tests for the S1 schedule arms (no GPU: generation is a stub).

The branch arm's claim is about a shared prefix; the schedule arm's claim is
the opposite one, and just as easy to get silently wrong: here the action a
turn takes must ENTER the history, so that a reminder at turn 5 is visible at
turn 6. An arm that quietly threw the reminded response away would still
produce plausible rows, a plausible retention curve, and a `B` estimated on a
trajectory nobody ever walked.

The other half is the design's arithmetic: turn 1 is action-free in every arm,
the antithetic arm is an exact complement so each (item, turn) cell has one
reminded side, and the arms all cover the same items.
"""

from __future__ import annotations

import pathlib

import pytest

from persona_drift.sequor_bank import load_sequor_bank
from persona_drift.sequor_trajectory import (
    SCHEDULE_ARMS,
    BranchArmConfig,
    assert_schedules_are_well_formed,
    build_schedules,
    run_schedule_arm,
)

TUPLES = pathlib.Path(__file__).resolve().parents[1] / "resources" / "sequor" / "sequor_tuples3.jsonl"
N_ITEMS, N_TURNS = 3, 6


@pytest.fixture
def items():
    return load_sequor_bank(TUPLES, n_turns=N_TURNS)[:N_ITEMS]


@pytest.fixture
def config():
    return BranchArmConfig(model_id="stub/model", n_turns=N_TURNS, seed=0)


class StubGenerator:
    def __init__(self):
        self.calls: list[list[list[dict]]] = []

    def __call__(self, conversations):
        self.calls.append([list(c) for c in conversations])
        return [
            {"text": f"reply at depth {len(c)}", "finish_reason": "stop",
             "n_output_tokens": 11, "n_inserted_tokens": 0}
            for c in conversations
        ]


def test_turn_one_is_action_free_in_every_arm(items):
    schedules = build_schedules(items, N_TURNS, seed=0)
    assert set(schedules) == set(SCHEDULE_ARMS)
    for arm, by_item in schedules.items():
        for item_id, u in by_item.items():
            assert u[0] == 0, f"{arm}/{item_id} reminds at turn 1, which is not an action"


def test_the_antithetic_arm_is_an_exact_complement(items):
    schedules = build_schedules(items, N_TURNS, seed=0)
    for item_id, coin in schedules["bernoulli"].items():
        anti = schedules["antithetic"][item_id]
        assert all(a + b == 1 for a, b in zip(coin[1:], anti[1:]))
        assert anti[0] == coin[0] == 0


def test_every_cell_has_exactly_one_reminded_side(items):
    """What makes `B` a matched-pair estimate rather than a pooled one."""

    schedules = build_schedules(items, N_TURNS, seed=0)
    for item in items:
        for turn in range(2, N_TURNS + 1):
            pair = (schedules["bernoulli"][item.conversation_id][turn - 1],
                    schedules["antithetic"][item.conversation_id][turn - 1])
            assert sum(pair) == 1


def test_the_zero_and_constant_arms_are_the_dose_contrast(items):
    schedules = build_schedules(items, N_TURNS, seed=0)
    for item in items:
        assert schedules["zero_control"][item.conversation_id] == [0] * N_TURNS
        assert schedules["constant_remind"][item.conversation_id] == [0] + [1] * (N_TURNS - 1)


def test_schedules_are_reproducible_and_seed_dependent(items):
    a = build_schedules(items, N_TURNS, seed=0)["bernoulli"]
    again = build_schedules(items, N_TURNS, seed=0)["bernoulli"]
    other = build_schedules(items, N_TURNS, seed=1)["bernoulli"]
    assert a == again
    assert a != other, "a seed that does not change the schedule gives three copies of one arm"


def test_the_structural_check_records_the_u_balance(items):
    record = assert_schedules_are_well_formed(build_schedules(items, N_TURNS, seed=0), N_TURNS)
    assert record["n_items"] == N_ITEMS and record["n_turns"] == N_TURNS
    assert 0.0 < record["bernoulli_u_mean"] < 1.0


def test_the_structural_check_catches_a_broken_complement(items):
    schedules = build_schedules(items, N_TURNS, seed=0)
    victim = items[0].conversation_id
    schedules["antithetic"][victim] = list(schedules["bernoulli"][victim])
    with pytest.raises(ValueError, match="not the complement"):
        assert_schedules_are_well_formed(schedules, N_TURNS)


def test_the_structural_check_catches_a_reminded_turn_one(items):
    schedules = build_schedules(items, N_TURNS, seed=0)
    schedules["constant_remind"][items[0].conversation_id][0] = 1
    with pytest.raises(ValueError, match="turn 1 carries u=1"):
        assert_schedules_are_well_formed(schedules, N_TURNS)


def test_the_structural_check_catches_arms_over_different_items(items):
    schedules = build_schedules(items, N_TURNS, seed=0)
    schedules["bernoulli"].pop(items[0].conversation_id)
    with pytest.raises(ValueError, match="different item sets"):
        assert_schedules_are_well_formed(schedules, N_TURNS)


def test_the_action_enters_the_history(items, config):
    """The property that separates this arm from the branch arm. Under
    `constant_remind` every turn from 2 on is reminded, so by turn t the
    prompt must contain t-1 reminded user messages -- if the runner generated
    the reminded turn and then advanced on something else, this count stays
    at zero and everything downstream is fitted to a trajectory that was
    never walked."""

    # The fidelity harness (block in the system message), so that a user turn
    # carrying the block is a REMINDER and never the turn-1 statement -- with
    # the block in turn 1 the two are indistinguishable by content.
    fidelity = BranchArmConfig(model_id="stub/model", n_turns=N_TURNS, seed=0,
                               constraints_in_system=True)
    schedules = build_schedules(items, N_TURNS, seed=0)
    gen = StubGenerator()
    rows = run_schedule_arm(items, gen, fidelity, "run", "constant_remind",
                            schedules["constant_remind"])
    assert len(rows) == N_ITEMS * N_TURNS
    block = items[0].constraint_block
    last_prompt = gen.calls[-1][0]
    first_user = next(m for m in last_prompt if m["role"] == "user")
    assert block not in first_user["content"], (
        "under this harness turn 1 must not carry the block, or a reminder and the "
        "original statement would be indistinguishable and this count would be off by one")
    reminded_in_history = sum(
        1 for m in last_prompt[:-1] if m["role"] == "user" and block in m["content"]
    )
    assert reminded_in_history == N_TURNS - 2, "turns 2..5 must be in the history as reminders"
    assert block in last_prompt[-1]["content"], "and turn 6 is being reminded right now"


def test_rows_carry_the_action_they_actually_took(items, config):
    schedules = build_schedules(items, N_TURNS, seed=0)
    for arm in SCHEDULE_ARMS:
        rows = run_schedule_arm(items, StubGenerator(), config, "run", arm, schedules[arm])
        for row in rows:
            expected = schedules[arm][row["item_id"]][row["turn"] - 1]
            assert row["u_remind"] == expected, f"{arm} row misreports its own action"
            assert row["branch"] == arm
            assert (row["inserted_chars"] > 0) == bool(expected)


def test_two_arms_do_not_share_a_prefix_after_turn_one(items, config):
    """They must never be paired at the row level: the trajectories diverge as
    soon as one of them acts. S1a pairs by item, S1b by the complementary
    action inside a cell."""

    schedules = build_schedules(items, N_TURNS, seed=0)
    zero = run_schedule_arm(items, StubGenerator(), config, "run", "zero_control",
                            schedules["zero_control"])
    const = run_schedule_arm(items, StubGenerator(), config, "run", "constant_remind",
                             schedules["constant_remind"])
    by_key = {(r["item_id"], r["turn"]): r for r in zero}
    assert all(by_key[(r["item_id"], r["turn"])]["prefix_sha256"] == r["prefix_sha256"]
               for r in const if r["turn"] == 1)
    diverged = [r for r in const if r["turn"] > 2
                and by_key[(r["item_id"], r["turn"])]["prefix_sha256"] != r["prefix_sha256"]]
    assert len(diverged) == N_ITEMS * (N_TURNS - 2)


def test_an_unknown_arm_raises(items, config):
    with pytest.raises(ValueError, match="unknown arm"):
        run_schedule_arm(items, StubGenerator(), config, "run", "mpc", {})
