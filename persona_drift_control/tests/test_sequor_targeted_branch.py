"""Tests for the targeted-reminder pilot (src/persona_drift/sequor_targeted_branch.py).

The guards that matter here are the ones protecting the counterfactual premise:
the branch must hang off a prefix that is byte-identical to the stored row's,
and the targeted reminder must be the item's own text with constraints removed
-- not a reminder written in a new style, which would confound targeting with a
prompt change.
"""

from __future__ import annotations

import pytest

from persona_drift.sequor_bank import (
    SequorItem,
    targeted_constraint_block,
    targeted_user_message,
    user_message,
)
from persona_drift.sequor_targeted_branch import (
    BLANKET,
    TARGETED,
    base_trajectories,
    branch_points,
    rebuild_history,
    run_targeted_branches,
    targeting_agreement,
    violated_indices,
)
from persona_drift.sequor_trajectory import BranchArmConfig, prefix_digest

N_TURNS = 6


def _item(name: str = "tuple_1") -> SequorItem:
    constraints = ("Exaggerate it", "Write bullet points", "Write in a cyberpunk theme")
    preamble = "In all your responses, make sure to adhere to these rules:"
    block = "\n".join([preamble] + [f"{i}. {c}" for i, c in enumerate(constraints, 1)])
    return SequorItem(
        conversation_id=name, tuple_id="t1", constraint_ids=("a", "b", "c"),
        constraints=constraints, preamble=preamble, constraint_block=block,
        turns=tuple(f"question {i}" for i in range(N_TURNS)), n_turns_available=N_TURNS,
    )


def _config(item: SequorItem) -> BranchArmConfig:
    return BranchArmConfig(model_id="agent", n_turns=N_TURNS, seed=0, max_new_tokens=2048,
                           temperature=0.7, top_p=0.8, top_k=20, constraints_in_system=True)


def _stored_arm(item: SequorItem, config: BranchArmConfig, seeds=(0, 1)) -> list[dict]:
    """A stored zero_control arm, built the way the real one was: the history
    grows turn by turn and each row records the digest of the prefix it was
    generated from."""

    from persona_drift.sequor_bank import system_message

    rows = []
    for seed in seeds:
        history = [{"role": "system", "content": system_message(item)}]
        for turn in range(1, N_TURNS + 1):
            message = user_message(item, turn - 1, remind=False, constraints_in_system=True)
            reply = f"reply s{seed} t{turn}"
            rows.append({
                "trajectory_id": f"{item.conversation_id}__zero_control__s{seed}",
                "item_id": item.conversation_id, "turn": turn, "branch": "zero_control",
                "u_remind": 0, "seed": seed, "n_turns": N_TURNS,
                "user_message": message, "agent_message": reply,
                "prefix_sha256": prefix_digest(history),
                "constraints_in_system": True, "system_prompt": None, "model": "agent",
                "decoding_config": {"temperature": 0.7, "top_p": 0.8, "top_k": 20,
                                    "max_new_tokens": 2048},
                "n_output_tokens": 5, "finish_reason": "stop",
            })
            history = history + [{"role": "user", "content": message},
                                 {"role": "assistant", "content": reply}]
    return rows


def _verdicts(rows: list[dict], broken_at: dict[int, list[int]]) -> dict:
    """`followed` per (trajectory, turn); `broken_at[turn]` lists the indices
    reported violated, and a turn mapped to None is a parse failure."""

    out = {}
    for row in rows:
        broken = broken_at.get(row["turn"], [])
        out[(row["trajectory_id"], row["turn"])] = (
            None if broken is None else [i not in broken for i in range(3)])
    return out


def _generate(conversations, seeds):
    assert len(seeds) == len(conversations)
    return [{"text": f"branch reply {i}", "finish_reason": "stop", "n_output_tokens": 7,
             "n_inserted_tokens": None} for i, _ in enumerate(conversations)]


def test_the_targeted_block_is_the_items_own_text_minus_the_kept_rules():
    item = _item()
    assert targeted_constraint_block(item, [1]) == (
        "In all your responses, make sure to adhere to these rules:\n2. Write bullet points"
        .replace("2.", "1."))
    # and asking for all three must reproduce the blanket block byte for byte
    assert targeted_constraint_block(item, [0, 1, 2]) == item.constraint_block


def test_a_block_we_cannot_rebuild_is_refused_rather_than_reformatted():
    """The negative control for the text: if an item's block is not 'preamble +
    numbered constraints', a targeted block would have to be written in a
    format the blanket arm never used, and the comparison would carry a prompt
    change. It must raise instead."""

    item = _item()
    odd = SequorItem(**{**item.__dict__, "constraint_block": item.constraint_block + "\nPlease comply."})
    with pytest.raises(ValueError, match="own format"):
        targeted_constraint_block(odd, [0])


def test_the_targeted_message_differs_from_the_blanket_one_only_in_the_block():
    item = _item()
    blanket = user_message(item, 2, remind=True, constraints_in_system=True)
    targeted = targeted_user_message(item, 2, [2], constraints_in_system=True)
    assert blanket.split("\n\n")[0] == targeted.split("\n\n")[0]
    assert targeted.endswith("1. Write in a cyberpunk theme")
    with pytest.raises(ValueError, match="not a distinct action"):
        targeted_user_message(item, 0, [0])


def test_every_branch_is_generated_from_the_stored_prefix():
    item = _item()
    config = _config(item)
    rows = _stored_arm(item, config)
    base = base_trajectories(rows)
    verdicts = _verdicts(rows, {2: [1], 3: [0, 2]})
    points = branch_points(base, verdicts, N_TURNS)
    out = run_targeted_branches({item.conversation_id: item}, base, points, _generate, config, "run")
    assert {r["branch"] for r in out} == {BLANKET, TARGETED}
    assert len(out) == 2 * len(points)
    for row in out:
        stored = base[(row["item_id"], row["seed"])][row["turn"] - 1]
        assert row["prefix_sha256"] == stored["prefix_sha256"]
        assert row["u_remind"] == 1
        # the two branches of one point must not collide on an id
    assert len({r["trajectory_id"] for r in out}) == len(out)
    # both seeds' trajectories must be branched, each row labelled with its own
    assert {r["seed"] for r in out} == {0, 1}
    for row in out:
        assert row["trajectory_id"].endswith(f"__s{row['seed']}__t{row['turn']}")


def test_a_prefix_that_does_not_match_its_digest_stops_the_arm():
    """The guard the whole design rests on. If the stored trajectory and the
    rebuilt history ever disagree, the 'counterfactual' would compare two
    different conversations and the measured gain would be a prompt effect."""

    item = _item()
    config = _config(item)
    rows = _stored_arm(item, config)
    for row in rows:
        if row["turn"] == 2:
            row["agent_message"] = "a reply nobody generated"
    base = base_trajectories(rows)
    points = branch_points(base, _verdicts(rows, {3: [0]}), N_TURNS)
    with pytest.raises(ValueError, match="not be a counterfactual"):
        run_targeted_branches({item.conversation_id: item}, base, points, _generate, config, "run")


def test_only_turns_with_something_broken_become_branch_points():
    item = _item()
    config = _config(item)
    rows = _stored_arm(item, config, seeds=(0,))
    base = base_trajectories(rows)
    # turn 1 has everything kept, turn 2 has one broken, turn 3 unparsed
    verdicts = _verdicts(rows, {2: [0], 3: None})
    points = branch_points(base, verdicts, N_TURNS)
    turns = sorted(p["turn"] for p in points)
    assert turns == [3]          # branch at t uses the verdict at t-1
    assert points[0]["violated_indices"] == [0]
    assert violated_indices([True, None, True]) is None
    assert violated_indices([True, True, True]) == []


def test_the_target_follows_the_readout_that_was_handed_in():
    item = _item()
    config = _config(item)
    rows = _stored_arm(item, config, seeds=(0,))
    base = base_trajectories(rows)
    points = branch_points(base, _verdicts(rows, {2: [2]}), N_TURNS)
    out = run_targeted_branches({item.conversation_id: item}, base, points, _generate, config, "run")
    targeted = [r for r in out if r["branch"] == TARGETED][0]
    blanket = [r for r in out if r["branch"] == BLANKET][0]
    assert targeted["targeted_constraints"] == ["Write in a cyberpunk theme"]
    assert blanket["targeted_constraints"] == list(item.constraints)
    assert targeted["user_message"].endswith("1. Write in a cyberpunk theme")
    assert blanket["user_message"].endswith(item.constraint_block)


def test_the_cheaper_judges_disagreement_is_counted_not_assumed():
    item = _item()
    config = _config(item)
    rows = _stored_arm(item, config, seeds=(0,))
    base = base_trajectories(rows)
    primary = _verdicts(rows, {2: [0], 3: [1], 4: [2]})
    secondary = _verdicts(rows, {2: [0], 3: [0], 4: None})  # agrees, differs, then fails to parse
    points = branch_points(base, primary, N_TURNS)
    agreement = targeting_agreement(primary, secondary, points)
    assert agreement["same_target"] == 1
    assert agreement["different_target"] == 1
    assert agreement["would_have_skipped_unparsed"] == 1


def test_rebuild_history_stops_before_the_turn_it_is_asked_for():
    item = _item()
    config = _config(item)
    rows = _stored_arm(item, config, seeds=(0,))
    turns = base_trajectories(rows)[(item.conversation_id, 0)]
    history = rebuild_history(item, turns, upto_turn=3, config=config)
    assert history[0]["role"] == "system"
    assert [m["role"] for m in history[1:]] == ["user", "assistant", "user", "assistant"]
    assert history[-1]["content"] == "reply s0 t2"
