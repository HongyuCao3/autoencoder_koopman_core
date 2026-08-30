from persona_drift.prompt_bank import (
    KNOWN_SATURATED_PROMPT_IDS,
    classify_scorer_screening_safety,
    load_prompt_bank,
    score_response,
    select_screening_prompts,
)


def test_load_prompt_bank_categories_and_counts():
    bank = load_prompt_bank()
    assert set(bank.keys()) == {"character_traits", "language_constraints"}
    assert len(bank["character_traits"]) == 13
    assert len(bank["language_constraints"]) == 28


def test_known_saturated_prompts_are_excluded():
    bank = load_prompt_bank()
    all_ids = {e.prompt_id for entries in bank.values() for e in entries}
    assert all_ids.isdisjoint(KNOWN_SATURATED_PROMPT_IDS)


def test_no_entry_has_unresolved_random_probe():
    bank = load_prompt_bank()
    for entries in bank.values():
        for entry in entries:
            assert entry.probe_question != "random"
            assert entry.probe_question


def test_random_probe_resolution_is_deterministic_across_loads():
    bank_a = load_prompt_bank()
    bank_b = load_prompt_bank()
    for label in bank_a:
        probes_a = [e.probe_question for e in bank_a[label]]
        probes_b = [e.probe_question for e in bank_b[label]]
        assert probes_a == probes_b


def test_select_screening_prompts_is_stratified_and_deterministic():
    bank = load_prompt_bank()
    selected_a = select_screening_prompts(bank, num_prompts=5, rng_seed=0)
    selected_b = select_screening_prompts(bank, num_prompts=5, rng_seed=0)
    assert len(selected_a) == 5
    assert [e.prompt_id for e in selected_a] == [e.prompt_id for e in selected_b]
    categories = {e.prompt_category for e in selected_a}
    assert categories == {"character_traits", "language_constraints"}


def test_score_response_handles_scoring_exceptions():
    bank = load_prompt_bank()
    entry = next(e for e in bank["character_traits"] if "tennis" in e.system_prompt.lower())
    score, failure = score_response(entry, "I love playing tennis on weekends.")
    assert failure is False
    assert score == 1.0


def test_classify_scorer_screening_safety_flags_boolean_predicates():
    is_safe, reason = classify_scorer_screening_safety(lambda x: 1.0 if "thank" in x.lower() else 0.0)
    assert is_safe is False
    assert reason == "binary_across_battery"


def test_classify_scorer_screening_safety_flags_unbounded_scores():
    is_safe, reason = classify_scorer_screening_safety(lambda x: float(len(x.split())))
    assert is_safe is False
    assert reason == "out_of_unit_range"


def test_classify_scorer_screening_safety_accepts_a_real_fraction_scorer():
    is_safe, reason = classify_scorer_screening_safety(lambda x: x.lower().count("a") / max(len(x), 1))
    assert is_safe is True
    assert reason == "ok"


def test_classify_scorer_screening_safety_is_inconclusive_when_every_eval_fails():
    def always_raises(_: str) -> float:
        raise ValueError("boom")

    is_safe, reason = classify_scorer_screening_safety(always_raises)
    assert is_safe is False
    assert reason == "all_battery_evals_failed"


def test_select_screening_prompts_avoids_known_saturated_scorer_shapes():
    bank = load_prompt_bank()
    take_per_label = 5  # num_prompts=10 // 2 labels
    safe_pool_sizes = {
        label: sum(classify_scorer_screening_safety(e.score_fn)[0] for e in entries)
        for label, entries in bank.items()
    }
    assert all(size >= take_per_label for size in safe_pool_sizes.values()), (
        f"test assumption violated, not enough screening-safe scorers per category: {safe_pool_sizes}"
    )
    for seed in range(10):
        selected = select_screening_prompts(bank, num_prompts=10, rng_seed=seed)
        for entry in selected:
            is_safe, reason = classify_scorer_screening_safety(entry.score_fn)
            assert is_safe, f"{entry.prompt_id} selected despite unsafe scorer ({reason})"


def test_score_response_never_raises_on_empty_text():
    # A real self-chat response can legitimately be an empty string
    # (refusal, truncation, max_new_tokens=0 edge case). Every third-party
    # scoring lambda must degrade to (nan, True) instead of propagating.
    bank = load_prompt_bank()
    for entries in bank.values():
        for entry in entries:
            score, failure = score_response(entry, "")
            if failure:
                assert score != score  # nan

