from persona_drift.chat_model import GenerationConfig, _split_at_think_end


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


def test_split_at_think_end_with_no_think_token_returns_everything_as_content():
    # enable_thinking=False (or a non-Qwen3 tokenizer): no </think> token is
    # ever emitted, so the whole generation is content, byte-for-byte the
    # pre-thinking-mode behavior.
    thinking_ids, content_ids = _split_at_think_end([10, 11, 12], think_end_id=None)
    assert thinking_ids == []
    assert content_ids == [10, 11, 12]


def test_split_at_think_end_with_think_end_id_not_present_returns_everything_as_content():
    thinking_ids, content_ids = _split_at_think_end([10, 11, 12], think_end_id=99)
    assert thinking_ids == []
    assert content_ids == [10, 11, 12]


def test_split_at_think_end_splits_after_the_think_end_token_inclusive():
    # [reasoning..., </think>, answer...] -> ([reasoning..., </think>], [answer...])
    thinking_ids, content_ids = _split_at_think_end([1, 2, 99, 3, 4], think_end_id=99)
    assert thinking_ids == [1, 2, 99]
    assert content_ids == [3, 4]


def test_split_at_think_end_with_no_content_after_think_end():
    # A generation truncated right at </think>, no final answer tokens left.
    thinking_ids, content_ids = _split_at_think_end([1, 2, 99], think_end_id=99)
    assert thinking_ids == [1, 2, 99]
    assert content_ids == []
