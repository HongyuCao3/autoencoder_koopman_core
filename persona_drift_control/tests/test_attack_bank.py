import pytest

from persona_drift.attack_bank import load_attack_bank, select_attacks_by_id, select_screening_attacks


def test_load_attack_bank_has_expected_categories():
    bank = load_attack_bank()
    assert set(bank.keys()) == {
        "chemical_biological",
        "cybercrime_intrusion",
        "harassment_bullying",
        "harmful",
        "illegal",
        "misinformation_disinformation",
    }
    assert sum(len(v) for v in bank.values()) == 600


def test_every_entry_has_at_least_three_turns_and_a_plain_query():
    bank = load_attack_bank()
    for entries in bank.values():
        for entry in entries:
            assert len(entry.multi_turn_queries) >= 3
            assert entry.plain_query
            assert entry.attack_id.startswith("safemtdata_")


def test_select_screening_attacks_is_stratified_and_deterministic():
    bank = load_attack_bank()
    selected_a = select_screening_attacks(bank, num_attacks=20, rng_seed=0)
    selected_b = select_screening_attacks(bank, num_attacks=20, rng_seed=0)
    assert len(selected_a) == 20
    assert [e.attack_id for e in selected_a] == [e.attack_id for e in selected_b]
    categories = {e.category for e in selected_a}
    assert categories == set(bank.keys())


def test_select_screening_attacks_differs_across_rng_seeds():
    bank = load_attack_bank()
    selected_a = select_screening_attacks(bank, num_attacks=20, rng_seed=0)
    selected_b = select_screening_attacks(bank, num_attacks=20, rng_seed=1)
    assert [e.attack_id for e in selected_a] != [e.attack_id for e in selected_b]


def test_select_attacks_by_id_preserves_requested_order():
    bank = load_attack_bank()
    sample = select_screening_attacks(bank, num_attacks=20, rng_seed=0)
    requested_ids = [sample[3].attack_id, sample[0].attack_id, sample[7].attack_id]
    selected = select_attacks_by_id(bank, requested_ids)
    assert [e.attack_id for e in selected] == requested_ids


def test_select_attacks_by_id_raises_on_unknown_id():
    bank = load_attack_bank()
    with pytest.raises(KeyError):
        select_attacks_by_id(bank, ["not_a_real_attack_id"])
