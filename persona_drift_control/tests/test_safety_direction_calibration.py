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


def test_excluded_attacks_are_dropped_before_sampling():
    # Disqualifying overlap for a READOUT: the harmless pole is the sampled
    # attacks' own turn-1 queries, so a direction calibrated on the attacks it
    # will later score is partly fit to call their opening turns safe. The
    # 2026-08-31 artifact overlapped all 8 Phase J held-out attacks.
    excluded = {"safemtdata_0074", "safemtdata_0169", "safemtdata_0257", "safemtdata_0289"}
    _harmful, _harmless, attack_ids = build_calibration_texts(40, 100, exclude_attack_ids=excluded)
    assert len(attack_ids) == 40
    assert not (set(attack_ids) & excluded)


def test_excluding_nothing_matches_the_unfiltered_sample():
    # Guard for every direction calibrated before this parameter existed.
    baseline = build_calibration_texts(40, 100)
    for empty in (None, [], set()):
        assert build_calibration_texts(40, 100, exclude_attack_ids=empty) == baseline
