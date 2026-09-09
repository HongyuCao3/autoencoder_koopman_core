"""Tests for persona_drift.sequor_bank.

The two load-bearing ones:

- `test_screening_selection_reproduces_the_documented_sixteen` pins the item
  selection to the number written in constraint_signal_screening.md section
  9.2. If the vendored gold set or tuples file is ever re-fetched and the
  overlap moves, the arm's items move with it and the gate thresholds were set
  against a different design.
- `test_reminded_turn_only_appends` is the counterfactual-branch premise: the
  u=0 and u=1 branches of a turn must share a byte-identical prefix, or the
  measured one-step gain is contaminated by a prompt difference.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from persona_drift.sequor_bank import (
    K_CONSTRAINTS,
    first_turn_user_message,
    gold_constraints,
    gold_coverage,
    load_sequor_bank,
    select_screening_items,
    user_message,
)

RESOURCES = pathlib.Path(__file__).resolve().parents[1] / "resources" / "sequor"
TUPLES = RESOURCES / "sequor_tuples3.jsonl"
GOLD = RESOURCES / "sequor_gold_judge_calibration.jsonl"

DOCUMENTED_FULLY_COVERED = 16


def test_bank_loads_the_vendored_two_hundred():
    items = load_sequor_bank(TUPLES)
    assert len(items) == 200
    assert all(len(it.constraints) == K_CONSTRAINTS for it in items)
    assert all(len(it.turns) == 30 for it in items)


def test_turn_truncation():
    items = load_sequor_bank(TUPLES, n_turns=20)
    assert all(len(it.turns) == 20 for it in items)


def test_requesting_more_turns_than_vendored_raises():
    """The vendor step truncated upstream's 120-210 turns to 30. Asking for 40
    must fail loudly rather than yield short dialogues."""
    with pytest.raises(ValueError, match="turns available"):
        load_sequor_bank(TUPLES, n_turns=40)


@pytest.mark.parametrize("mutate,match", [
    (lambda raw: raw.__setitem__("constraints", raw["constraints"][:2]), "constraints, expected"),
    (lambda raw: raw.__setitem__("constraint_block", "Follow these:\n1. x"), "does not open with its preamble"),
    (lambda raw: raw.__setitem__("constraints", ["not in the block"] * 3), "absent from the block"),
])
def test_boundary_validation(tmp_path, mutate, match):
    raw = json.loads(TUPLES.open().readline())
    mutate(raw)
    path = tmp_path / "one.jsonl"
    path.write_text(json.dumps(raw) + "\n")
    with pytest.raises(ValueError, match=match):
        load_sequor_bank(path)


def test_screening_selection_reproduces_the_documented_sixteen():
    items = load_sequor_bank(TUPLES, n_turns=20)
    gold = gold_constraints(GOLD)
    fully = [it for it in items if gold_coverage(it, gold) == K_CONSTRAINTS]
    assert len(fully) == DOCUMENTED_FULLY_COVERED


def test_screening_selection_is_deterministic_and_fully_covered():
    items = load_sequor_bank(TUPLES, n_turns=20)
    gold = gold_constraints(GOLD)
    twelve = select_screening_items(items, gold, 12)
    assert [it.conversation_id for it in twelve] == \
           [it.conversation_id for it in select_screening_items(items, gold, 12)]
    assert len(twelve) == 12
    assert all(gold_coverage(it, gold) == K_CONSTRAINTS for it in twelve)


def test_asking_for_more_items_than_are_fully_covered_raises():
    items = load_sequor_bank(TUPLES, n_turns=20)
    gold = gold_constraints(GOLD)
    with pytest.raises(ValueError, match="fully gold-covered"):
        select_screening_items(items, gold, DOCUMENTED_FULLY_COVERED + 1)


def test_first_turn_carries_the_block_before_the_question():
    item = load_sequor_bank(TUPLES, n_turns=20)[0]
    msg = first_turn_user_message(item)
    assert msg.startswith(item.constraint_block)
    assert msg.endswith(item.turns[0])
    for constraint in item.constraints:
        assert constraint in msg


def test_reminded_turn_only_appends():
    item = load_sequor_bank(TUPLES, n_turns=20)[0]
    base = user_message(item, 5, remind=False)
    reminded = user_message(item, 5, remind=True)
    assert reminded.startswith(base)
    assert reminded == f"{base}\n\n{item.constraint_block}"


def test_only_turn_one_carries_the_block_when_not_reminded():
    item = load_sequor_bank(TUPLES, n_turns=20)[0]
    for turn in range(1, 20):
        assert item.constraint_block not in user_message(item, turn, remind=False)


def test_reminding_on_turn_one_is_refused():
    """Turn 1 already states the constraints, so u=1 there is not a distinct
    action -- a pair generated at turn 1 would have zero dose difference and
    would dilute the arm's effect estimate with a guaranteed null."""
    item = load_sequor_bank(TUPLES, n_turns=20)[0]
    with pytest.raises(ValueError, match="turn 1 already carries"):
        user_message(item, 0, remind=True)
