from persona_drift.chat_model import GenerationConfig


def test_generation_config_defaults_are_transformers_no_ops():
    # Existing callers (selfchat.py's live loop, validated by the completed
    # signal-screening run) must see byte-for-byte identical .generate()
    # kwargs unless they explicitly opt into anti-repetition settings.
    config = GenerationConfig()
    assert config.repetition_penalty == 1.0
    assert config.no_repeat_ngram_size == 0


def test_generation_config_accepts_anti_repetition_overrides():
    config = GenerationConfig(repetition_penalty=1.15, no_repeat_ngram_size=4)
    assert config.repetition_penalty == 1.15
    assert config.no_repeat_ngram_size == 4
