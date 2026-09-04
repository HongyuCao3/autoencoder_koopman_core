from persona_drift.chat_model import GenerationConfig, _build_prompt_text, _split_at_think_end


class _ThinkingAwareTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        return f"thinking={enable_thinking}"


class _LegacyTokenizer:
    """Tokenizer whose chat template predates thinking-mode support."""

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking=None):
        if enable_thinking is not None:
            raise TypeError("unexpected keyword argument 'enable_thinking'")
        return "legacy prompt"


def test_build_prompt_text_passes_enable_thinking_through():
    assert _build_prompt_text(_ThinkingAwareTokenizer(), [{"role": "user", "content": "hi"}], True) == "thinking=True"
    assert _build_prompt_text(_ThinkingAwareTokenizer(), [{"role": "user", "content": "hi"}], False) == "thinking=False"


def test_build_prompt_text_falls_back_for_tokenizers_without_thinking_support():
    assert _build_prompt_text(_LegacyTokenizer(), [{"role": "user", "content": "hi"}], False) == "legacy prompt"


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
