"""Item bank for the `constraint` line (SEQUOR multi-turn constraint retention,
docs/experiments/constraint_retention_plan.md).

Reads the vendored `resources/sequor/sequor_tuples3.jsonl` (200 dialogues,
k=3 constraints each, `turns` truncated to the first 30 upstream turns; see
resources/PROVENANCE.md). Every dialogue states its constraints ONCE, in turn
1, and never restates them -- verified on all 200, which is what makes "how
many of the k are still satisfied at turn t" a pure retention readout with
nothing propping it up.

Two properties of the file that shape this module:

- The preamble has SIX wordings across the 200 dialogues ("In all your
  responses, make sure to adhere to these rules:", "As we talk, always comply
  with these constraints:", ...). Nothing here parses the block as text: the
  constraints arrive as a structured list and the block is used verbatim.
- `constraints` is the judged unit. Upstream's judge sees one constraint and
  one answer, never the history (sequor_constraint_judge.py), so the k
  decisions stay independent -- the property the graded readout needs.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass

K_CONSTRAINTS = 3


@dataclass(frozen=True)
class SequorItem:
    conversation_id: str
    tuple_id: str
    constraint_ids: tuple[str, ...]
    constraints: tuple[str, ...]
    preamble: str
    constraint_block: str
    turns: tuple[str, ...]
    n_turns_available: int


def load_sequor_bank(path: pathlib.Path, n_turns: int | None = None) -> list[SequorItem]:
    """Load the bank, optionally truncating each dialogue to `n_turns` turns.

    Validates at the file boundary rather than trusting the vendor step: k=3,
    the block carries the preamble and every constraint verbatim, and enough
    turns exist. A dialogue that silently had 2 constraints would make `y`
    take a different number of values for that item alone -- the quiet readout
    drift that cost this project two lines.
    """

    items: list[SequorItem] = []
    for line_no, line in enumerate(path.open(), start=1):
        if not line.strip():
            continue
        raw = json.loads(line)
        constraints = tuple(raw["constraints"])
        if len(constraints) != K_CONSTRAINTS:
            raise ValueError(f"{path}:{line_no}: {len(constraints)} constraints, expected {K_CONSTRAINTS}")
        block = raw["constraint_block"]
        if not block.startswith(raw["preamble"]):
            raise ValueError(f"{path}:{line_no}: constraint_block does not open with its preamble")
        missing = [c for c in constraints if c not in block]
        if missing:
            raise ValueError(f"{path}:{line_no}: constraints absent from the block: {missing}")
        turns = tuple(raw["turns"])
        if n_turns is not None:
            if len(turns) < n_turns:
                raise ValueError(f"{path}:{line_no}: {len(turns)} turns available, {n_turns} requested")
            turns = turns[:n_turns]
        items.append(SequorItem(
            conversation_id=raw["conversation_id"], tuple_id=raw["tuple_id"],
            constraint_ids=tuple(raw["constraint_ids"]), constraints=constraints,
            preamble=raw["preamble"], constraint_block=block, turns=turns,
            n_turns_available=raw["n_turns_available"],
        ))
    return items


def gold_constraints(gold_path: pathlib.Path) -> set[str]:
    """The constraint strings the S0 judge calibration covers."""
    return {json.loads(line)["constraint"] for line in gold_path.open() if line.strip()}


def gold_coverage(item: SequorItem, gold: set[str]) -> int:
    return sum(1 for c in item.constraints if c in gold)


def select_screening_items(items: list[SequorItem], gold: set[str], n: int) -> list[SequorItem]:
    """The S0-0 screening arm's items: dialogues whose k constraints are ALL in
    the judge-calibration gold set, ordered by conversation_id, first `n`.

    Why fully-covered only (constraint_signal_screening.md section 9.2): the
    gold set covers 163 of the 439 constraints these dialogues use, so 16 of
    the 200 dialogues are fully covered. The screening arm's job is to decide
    whether a state EXISTS; making it also carry "does the judge hold up on
    constraints it was never calibrated on" would mean a failed gate has two
    possible causes and closes the line for the wrong one. S1/S3 cannot have
    this (only 16 dialogues qualify) and must report the gap instead.

    Raises rather than quietly returning fewer: an arm that ran on 9 items
    when 12 were pre-registered has the wrong MDE, and the gate threshold was
    set against the design, not against what the file happened to yield.
    """

    covered = sorted(
        (it for it in items if gold_coverage(it, gold) == K_CONSTRAINTS),
        key=lambda it: it.conversation_id,
    )
    if len(covered) < n:
        raise ValueError(
            f"{n} fully gold-covered dialogues requested, {len(covered)} exist. "
            f"Do NOT relax the coverage requirement to fill the arm -- that trades a "
            f"known instrument for an unknown one mid-screening."
        )
    return covered[:n]


def system_message(item: SequorItem) -> str:
    """The constraint block as a system message.

    The alternative placement, for the upstream-fidelity check
    (`constraint_signal_screening.md`, the 2026-09-09 fidelity arm): our turn-1
    placement was INFERRED from the vendored `constraint_block` field, never
    checked against upstream's own conversation builder, and the ERGO line
    demonstrated that moving instructions between a system message and a user
    turn changes the answer to the closed-loop question (LEDGER section 3,
    `prompt_profile`). The block is used verbatim in both placements, so the
    only thing that varies is where it sits.
    """

    return item.constraint_block


def first_turn_user_message(item: SequorItem) -> str:
    """Turn 1: the constraint block, then the user's first question.

    This is the only turn that carries the constraints. The block is used
    verbatim, preamble wording and all, so nothing about the item's phrasing is
    normalized away.
    """

    return f"{item.constraint_block}\n\n{item.turns[0]}"


def user_message(item: SequorItem, turn: int, remind: bool, constraints_in_system: bool = False) -> str:
    """The user turn as the agent sees it. `turn` is 0-based.

    `remind` is the executor (`u=1`): the item's own constraint block, appended
    after the user's question. Appended, not prepended, and reusing the item's
    own wording rather than a new instruction style -- the action under study
    is "restate the constraints", and a reworded reminder would confound the
    dose with a prompt change.

    `constraints_in_system` moves the block out of turn 1 (see
    `system_message`); the executor is unchanged either way, since restating
    the constraints in the user turn is the action being studied.
    """

    if turn == 0:
        base = item.turns[0] if constraints_in_system else first_turn_user_message(item)
    else:
        base = item.turns[turn]
    if not remind:
        return base
    if turn == 0:
        raise ValueError("turn 1 already carries the constraint block; u=1 there is not a distinct action")
    return f"{base}\n\n{item.constraint_block}"
