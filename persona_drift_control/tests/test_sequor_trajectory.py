"""Tests for persona_drift.sequor_trajectory (no GPU: generation is a stub).

The arm's whole claim is that a pair's two responses come from the SAME
prefix, so most of these tests are about the prefix rather than the numbers:
that the reminded branch never enters the history, that both rows of a pair
agree on their prefix digest, and that a tampered digest raises instead of
being reported as a rate.
"""

from __future__ import annotations

import pathlib

import pytest

from persona_drift.sequor_bank import load_sequor_bank
from persona_drift.sequor_trajectory import (
    BASE,
    REMINDED,
    BranchArmConfig,
    arm_config_record,
    assert_pairs_share_prefix,
    prefix_digest,
    run_branch_arm,
    word_jaccard,
)

TUPLES = pathlib.Path(__file__).resolve().parents[1] / "resources" / "sequor" / "sequor_tuples3.jsonl"
N_ITEMS, N_TURNS = 3, 4


@pytest.fixture
def items():
    return load_sequor_bank(TUPLES, n_turns=N_TURNS)[:N_ITEMS]


@pytest.fixture
def config():
    return BranchArmConfig(model_id="stub/model", n_turns=N_TURNS, seed=0)


class StubGenerator:
    """Echoes the last user message back, so a response identifies the prompt
    that produced it, and records every prompt it was handed."""

    def __init__(self):
        self.calls: list[list[list[dict]]] = []

    def __call__(self, conversations):
        self.calls.append([list(c) for c in conversations])
        return [
            {"text": f"reply to <{c[-1]['content'][:24]}> at depth {len(c)}",
             "finish_reason": "stop", "n_output_tokens": 11, "n_inserted_tokens": 0}
            for c in conversations
        ]


def test_row_counts(items, config):
    rows = run_branch_arm(items, StubGenerator(), config, run_id="r")
    assert sum(1 for r in rows if r["branch"] == BASE) == N_ITEMS * N_TURNS
    assert sum(1 for r in rows if r["branch"] == REMINDED) == N_ITEMS * (N_TURNS - 1)


def test_turn_one_has_no_reminded_row(items, config):
    rows = run_branch_arm(items, StubGenerator(), config, run_id="r")
    assert not [r for r in rows if r["turn"] == 1 and r["branch"] == REMINDED]


def test_every_pair_shares_a_prefix(items, config):
    rows = run_branch_arm(items, StubGenerator(), config, run_id="r")
    assert assert_pairs_share_prefix(rows) == N_ITEMS * (N_TURNS - 1)


def test_a_tampered_prefix_raises(items, config):
    rows = run_branch_arm(items, StubGenerator(), config, run_id="r")
    victim = next(r for r in rows if r["branch"] == REMINDED)
    victim["prefix_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="NOT.*byte-identical prefix"):
        assert_pairs_share_prefix(rows)


def test_a_reminded_row_without_its_base_sibling_raises(items, config):
    rows = run_branch_arm(items, StubGenerator(), config, run_id="r")
    rows = [r for r in rows if not (r["branch"] == BASE and r["turn"] == 2)]
    with pytest.raises(ValueError, match="no base sibling"):
        assert_pairs_share_prefix(rows)


def test_reminded_branch_never_enters_the_history(items, config):
    """The u=1 response is scored and dropped. If it were appended, the next
    turn's pair would no longer be paired against the base trajectory."""

    gen = StubGenerator()
    run_branch_arm(items, gen, config, run_id="r")
    for conversations in gen.calls:
        for conversation in conversations:
            for message in conversation:
                assert "constraint" not in message.get("content", "").lower() or message["role"] == "user"
            assistant_turns = [m for m in conversation if m["role"] == "assistant"]
            assert all("depth" in m["content"] for m in assistant_turns)
    # the deepest prompt is the last base turn: 1 user + (T-1) x (user, assistant)
    assert max(len(c) for calls in gen.calls for c in calls) == 2 * (N_TURNS - 1) + 1


def test_pair_prompts_differ_only_in_the_appended_block(items, config):
    gen = StubGenerator()
    run_branch_arm(items, gen, config, run_id="r")
    # call order is [t1 base, t2 base, t2 reminded, t3 base, t3 reminded, ...]:
    # the base branch is generated first each turn, because it is the one that
    # advances the history the reminded branch has to be paired against.
    base_t2, reminded_t2 = gen.calls[1], gen.calls[2]
    for base_prompt, reminded_prompt in zip(base_t2, reminded_t2):
        assert base_prompt[:-1] == reminded_prompt[:-1]
        assert prefix_digest(base_prompt[:-1]) == prefix_digest(reminded_prompt[:-1])
        assert reminded_prompt[-1]["content"].startswith(base_prompt[-1]["content"])
        assert len(reminded_prompt[-1]["content"]) > len(base_prompt[-1]["content"])


def test_inserted_cost_is_recorded_only_on_reminded_rows(items, config):
    rows = run_branch_arm(items, StubGenerator(), config, run_id="r")
    assert all(r["inserted_chars"] == 0 for r in rows if r["branch"] == BASE)
    assert all(r["inserted_chars"] > 0 for r in rows if r["branch"] == REMINDED)


def test_echo_is_none_on_turn_one_and_set_afterwards(items, config):
    rows = run_branch_arm(items, StubGenerator(), config, run_id="r")
    assert all(r["echo_jaccard_prev"] is None for r in rows if r["turn"] == 1)
    assert all(r["echo_jaccard_prev"] is not None for r in rows if r["turn"] > 1)


def test_word_jaccard_bounds():
    assert word_jaccard("a b c", "a b c") == 1.0
    assert word_jaccard("a b", "c d") == 0.0
    assert word_jaccard("", "") == 0.0
    assert word_jaccard("a b", "b c") == pytest.approx(1 / 3)


def test_arm_config_record_states_the_expected_pair_count(items, config):
    record = arm_config_record(config, items)
    assert record["expected_pairs"] == N_ITEMS * (N_TURNS - 1)
    assert record["item_ids"] == [i.conversation_id for i in items]
    assert record["model_id"] == "stub/model"


def test_a_system_prompt_lands_in_every_prefix(items):
    gen = StubGenerator()
    config = BranchArmConfig(model_id="stub/model", n_turns=2, system_prompt="be terse")
    rows = run_branch_arm(items, gen, config, run_id="r")
    assert all(c[0] == {"role": "system", "content": "be terse"} for calls in gen.calls for c in calls)
    assert all(r["system_prompt"] == "be terse" for r in rows)
