from persona_drift.surface_features import SURFACE_FEATURE_NAMES, extract_surface_features


def test_returns_all_expected_keys():
    features = extract_surface_features("The quick brown fox jumps over the lazy dog.")
    assert set(features.keys()) == set(SURFACE_FEATURE_NAMES)


def test_empty_text_is_nan_safe_not_a_crash():
    features = extract_surface_features("")
    assert features["num_tokens"] == 0.0
    assert features["num_sents"] == 0.0
    assert features["avg_word_len"] != features["avg_word_len"]  # NaN != NaN


def test_adjective_heavy_text_has_higher_adj_ratio_than_plain_text():
    adj_heavy = extract_surface_features("The big red beautiful wonderful house.")
    plain = extract_surface_features("The house is there.")
    assert adj_heavy["adj_ratio"] > plain["adj_ratio"]


def test_positive_and_negative_sentiment_are_distinguishable():
    positive = extract_surface_features("I love this, it's absolutely wonderful and amazing!")
    negative = extract_surface_features("I hate this, it's terrible and awful.")
    assert positive["vader_sentiment"] > 0
    assert negative["vader_sentiment"] < 0


def test_repeated_words_lower_type_token_ratio():
    repetitive = extract_surface_features("go go go go go go go go")
    varied = extract_surface_features("go run jump swim climb dance sing fly")
    assert repetitive["ttr"] < varied["ttr"]
