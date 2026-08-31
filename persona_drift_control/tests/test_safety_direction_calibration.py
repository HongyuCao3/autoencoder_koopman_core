from persona_drift.safety_direction_calibration import build_calibration_texts


def test_build_calibration_texts_returns_matched_harmful_and_harmless_pairs():
    harmful_texts, harmless_texts, attack_ids = build_calibration_texts(num_attacks=10, rng_seed=100)

    assert len(harmful_texts) == 10
    assert len(harmless_texts) == 10
    assert len(attack_ids) == 10
    assert all(text for text in harmful_texts)
    assert all(text for text in harmless_texts)
    # harmless texts are turn-1 (context-building) queries, distinct from
    # the harmful plain_query goal for the same attack.
    assert harmful_texts != harmless_texts


def test_build_calibration_texts_is_deterministic_given_rng_seed():
    a = build_calibration_texts(num_attacks=10, rng_seed=100)
    b = build_calibration_texts(num_attacks=10, rng_seed=100)
    assert a == b


def test_build_calibration_texts_differs_across_rng_seeds():
    a = build_calibration_texts(num_attacks=10, rng_seed=100)
    b = build_calibration_texts(num_attacks=10, rng_seed=200)
    assert a[2] != b[2]  # attack_ids differ
