from persona_drift.benign_bank import ENTRIES_PER_SESSION, all_benign_sessions, load_benign_bank

EXPECTED_CATEGORIES = {
    "writing",
    "roleplay",
    "reasoning",
    "math",
    "coding",
    "extraction",
    "stem",
    "humanities",
}


def test_load_benign_bank_covers_every_mtbench_category():
    bank = load_benign_bank()
    assert set(bank.keys()) == EXPECTED_CATEGORIES
    for category, entries in bank.items():
        assert len(entries) == 1
        assert entries[0].category == category


def test_each_session_chains_entries_per_session_mtbench_questions():
    bank = load_benign_bank()
    for entries in bank.values():
        (entry,) = entries
        assert len(entry.multi_turn_queries) == ENTRIES_PER_SESSION * 2
        assert all(isinstance(q, str) and q for q in entry.multi_turn_queries)


def test_session_content_is_deterministic_across_loads():
    bank_a = load_benign_bank()
    bank_b = load_benign_bank()
    for category in EXPECTED_CATEGORIES:
        assert bank_a[category][0].multi_turn_queries == bank_b[category][0].multi_turn_queries


def test_all_benign_sessions_returns_one_per_category_sorted():
    bank = load_benign_bank()
    sessions = all_benign_sessions(bank)
    assert len(sessions) == len(EXPECTED_CATEGORIES)
    assert [s.category for s in sessions] == sorted(EXPECTED_CATEGORIES)
    assert len({s.benign_id for s in sessions}) == len(EXPECTED_CATEGORIES)
