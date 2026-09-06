import json

import pytest

from persona_drift.ergo_math_bank import load_ergo_math_bank, select_items_by_id, select_screening_items


@pytest.fixture
def resources_dir(tmp_path):
    rows = [
        {"item_id": "ergo_GSM8K_0", "gold_answer": "10", "shards": ["clue A", "clue B"]},
        {"item_id": "ergo_GSM8K_1", "gold_answer": "20", "shards": ["clue C", "clue D", "clue E"]},
        {"item_id": "ergo_GSM8K_2", "gold_answer": "30", "shards": ["clue F"]},
    ]
    path = tmp_path / "ergo_gsm8k_sharded.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return tmp_path


def test_load_ergo_math_bank_returns_flat_list(resources_dir):
    bank = load_ergo_math_bank(resources_dir)
    assert len(bank) == 3
    assert {item.item_id for item in bank} == {"ergo_GSM8K_0", "ergo_GSM8K_1", "ergo_GSM8K_2"}
    item = next(i for i in bank if i.item_id == "ergo_GSM8K_1")
    assert item.shards == ("clue C", "clue D", "clue E")
    assert item.gold_answer == "20"


def test_select_screening_items_is_deterministic(resources_dir):
    bank = load_ergo_math_bank(resources_dir)
    selected = select_screening_items(bank, num_items=2, rng_seed=0)
    assert len(selected) == 2
    again = select_screening_items(bank, num_items=2, rng_seed=0)
    assert [item.item_id for item in selected] == [item.item_id for item in again]


def test_select_screening_items_caps_at_bank_size(resources_dir):
    bank = load_ergo_math_bank(resources_dir)
    selected = select_screening_items(bank, num_items=100, rng_seed=0)
    assert len(selected) == 3


def test_select_items_by_id_preserves_order(resources_dir):
    bank = load_ergo_math_bank(resources_dir)
    selected = select_items_by_id(bank, ["ergo_GSM8K_2", "ergo_GSM8K_0"])
    assert [item.item_id for item in selected] == ["ergo_GSM8K_2", "ergo_GSM8K_0"]


def test_select_items_by_id_raises_on_unknown_id(resources_dir):
    bank = load_ergo_math_bank(resources_dir)
    with pytest.raises(KeyError):
        select_items_by_id(bank, ["not_a_real_item_id"])


def test_vendored_resource_loads_and_has_the_expected_shape():
    bank = load_ergo_math_bank()
    assert len(bank) == 103
    for item in bank:
        assert item.gold_answer
        assert 4 <= len(item.shards) <= 12
