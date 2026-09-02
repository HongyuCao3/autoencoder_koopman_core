import pytest

from persona_drift.sycophancy_bank import load_sycophancy_bank, select_items_by_id, select_screening_items


def test_load_sycophancy_bank_has_expected_category_and_count():
    bank = load_sycophancy_bank()
    assert set(bank.keys()) == {"false_presupposition"}
    assert sum(len(v) for v in bank.values()) == 200


def test_every_item_has_all_ground_truth_fields_and_pushback_turns():
    bank = load_sycophancy_bank()
    for items in bank.values():
        for item in items:
            assert item.question
            assert item.presupposition
            assert item.correction
            assert len(item.pushback_turns) == 4
            assert item.item_id.startswith("sycon_fp_")


def test_multi_turn_queries_prepends_question_to_pushback_turns():
    bank = load_sycophancy_bank()
    item = bank["false_presupposition"][0]
    assert item.multi_turn_queries == (item.question, *item.pushback_turns)
    assert len(item.multi_turn_queries) == 5


def test_select_screening_items_is_stratified_and_deterministic():
    bank = load_sycophancy_bank()
    selected_a = select_screening_items(bank, num_items=20, rng_seed=0)
    selected_b = select_screening_items(bank, num_items=20, rng_seed=0)
    assert len(selected_a) == 20
    assert [i.item_id for i in selected_a] == [i.item_id for i in selected_b]


def test_select_screening_items_differs_across_rng_seeds():
    bank = load_sycophancy_bank()
    selected_a = select_screening_items(bank, num_items=20, rng_seed=0)
    selected_b = select_screening_items(bank, num_items=20, rng_seed=1)
    assert [i.item_id for i in selected_a] != [i.item_id for i in selected_b]


def test_select_items_by_id_preserves_requested_order():
    bank = load_sycophancy_bank()
    sample = select_screening_items(bank, num_items=20, rng_seed=0)
    requested_ids = [sample[3].item_id, sample[0].item_id, sample[7].item_id]
    selected = select_items_by_id(bank, requested_ids)
    assert [i.item_id for i in selected] == requested_ids


def test_select_items_by_id_raises_on_unknown_id():
    bank = load_sycophancy_bank()
    with pytest.raises(KeyError):
        select_items_by_id(bank, ["not_a_real_item_id"])
