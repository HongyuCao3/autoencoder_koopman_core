import json

import pytest

from persona_drift.mc_sycophancy_bank import (
    PUSHBACK_TURNS,
    load_mc_sycophancy_bank,
    select_items_by_id,
    select_screening_items,
)


@pytest.fixture
def resources_dir(tmp_path):
    rows = [
        {"item_id": "mmlu_mc_0000", "category": "topicA", "question": "Q0?", "choices": {"A": "x", "B": "y"}, "correct_letter": "A"},
        {"item_id": "mmlu_mc_0001", "category": "topicA", "question": "Q1?", "choices": {"A": "x", "B": "y"}, "correct_letter": "B"},
        {"item_id": "mmlu_mc_0002", "category": "topicB", "question": "Q2?", "choices": {"A": "x", "B": "y"}, "correct_letter": "A"},
    ]
    path = tmp_path / "mmlu_sycophancy_mc.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return tmp_path


def test_load_mc_sycophancy_bank_groups_by_category(resources_dir):
    bank = load_mc_sycophancy_bank(resources_dir)
    assert set(bank) == {"topicA", "topicB"}
    assert len(bank["topicA"]) == 2
    assert len(bank["topicB"]) == 1
    item = bank["topicA"][0]
    assert item.correct_letter in ("A", "B")
    assert item.choices == {"A": "x", "B": "y"}


def test_rendered_question_includes_choices_and_format_instruction(resources_dir):
    bank = load_mc_sycophancy_bank(resources_dir)
    item = bank["topicB"][0]
    rendered = item.rendered_question
    assert "Q2?" in rendered
    assert "(A) x" in rendered
    assert "(B) y" in rendered
    assert "Final answer" in rendered


def test_multi_turn_queries_is_question_then_fixed_pushback_script(resources_dir):
    bank = load_mc_sycophancy_bank(resources_dir)
    item = bank["topicB"][0]
    queries = item.multi_turn_queries
    assert len(queries) == 1 + len(PUSHBACK_TURNS)
    assert queries[0] == item.rendered_question
    assert queries[1:] == PUSHBACK_TURNS


def test_select_screening_items_is_deterministic_and_stratified(resources_dir):
    bank = load_mc_sycophancy_bank(resources_dir)
    selected = select_screening_items(bank, num_items=2, rng_seed=0)
    assert len(selected) == 2
    assert {item.category for item in selected} == {"topicA", "topicB"}
    again = select_screening_items(bank, num_items=2, rng_seed=0)
    assert [item.item_id for item in selected] == [item.item_id for item in again]


def test_select_items_by_id_preserves_requested_order(resources_dir):
    bank = load_mc_sycophancy_bank(resources_dir)
    selected = select_items_by_id(bank, ["mmlu_mc_0002", "mmlu_mc_0000"])
    assert [item.item_id for item in selected] == ["mmlu_mc_0002", "mmlu_mc_0000"]


def test_select_items_by_id_raises_on_unknown_id(resources_dir):
    bank = load_mc_sycophancy_bank(resources_dir)
    with pytest.raises(KeyError):
        select_items_by_id(bank, ["not_a_real_item_id"])


def test_vendored_resource_loads_and_has_the_expected_shape():
    bank = load_mc_sycophancy_bank()
    total = sum(len(items) for items in bank.values())
    assert total == 1000
    assert len(bank) == 57
    for items in bank.values():
        for item in items:
            assert item.correct_letter in item.choices
            assert set(item.choices) <= {"A", "B", "C", "D"}
