"""Offline (no GPU, no real model) checks for
scripts/analyze_ergo_entropy_readout.py's index arithmetic -- the one place
an off-by-one would silently misattribute entropy to the wrong token and
burn GPU budget before anyone noticed."""

import importlib.util
import pathlib

import torch

MODULE_PATH = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "analyze_ergo_entropy_readout.py"
spec = importlib.util.spec_from_file_location("analyze_ergo_entropy_readout", MODULE_PATH)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class _FakeTokenizer:
    """Whitespace tokenizer: each id is the word's index in a fixed
    vocabulary, offsets are the word's exact character span. Deterministic
    and good enough to test slicing, not real tokenization."""

    def __init__(self, vocab: dict[str, int]):
        self.vocab = vocab

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
        # Single-user-message prompts only, good enough for these tests:
        # concatenates content verbatim so word-level tokenization below
        # lands on known vocabulary words.
        return " ".join(m["content"] for m in messages)

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        ids, offsets = [], []
        pos = 0
        for word in text.split(" "):
            start = text.index(word, pos)
            end = start + len(word)
            pos = end
            ids.append(self.vocab[word])
            offsets.append((start, end))
        out = {"input_ids": ids}
        if return_offsets_mapping:
            out["offset_mapping"] = offsets
        return out


class _FakeInnerModel:
    def __init__(self, seq_len: int, vocab_size: int = 5):
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __call__(self, full_ids):
        n = full_ids.shape[1]
        logits = torch.zeros(1, n, self.vocab_size)
        return type("Out", (), {"logits": logits})()


class _FakeChatModel:
    def __init__(self, tokenizer, seq_len, device="cpu"):
        self.tokenizer = tokenizer
        self.model = _FakeInnerModel(seq_len)
        self.device = device


def test_token_entropies_length_matches_completion_token_count():
    vocab = {"prompt": 0, "has": 1, "three": 2, "words": 3, "one": 4, "two": 5}
    tokenizer = _FakeTokenizer(vocab)
    model = _FakeChatModel(tokenizer, seq_len=None)
    entropies = mod._token_entropies(model, [{"role": "user", "content": "prompt has three words"}], "one two")
    assert entropies.numel() == 2  # completion "one two" -> 2 tokens


def test_token_entropies_uniform_logits_give_max_entropy():
    # Fake inner model always returns all-zero logits -> uniform distribution
    # over vocab_size=5 -> entropy = ln(5) at every position, a hand-checkable value.
    vocab = {"a": 0, "b": 1}
    tokenizer = _FakeTokenizer(vocab)
    model = _FakeChatModel(tokenizer, seq_len=None)
    entropies = mod._token_entropies(model, [{"role": "user", "content": "a"}], "b")
    assert entropies.numel() == 1
    assert abs(float(entropies[0]) - torch.log(torch.tensor(5.0))) < 1e-5


def test_answer_span_entropy_starts_at_the_marker_token():
    vocab = {"Current": 0, "answer:": 1, "42": 2, "extra": 3}
    tokenizer = _FakeTokenizer(vocab)
    model = _FakeChatModel(tokenizer, seq_len=None)
    completion = "extra Current answer: 42"
    entropies = torch.tensor([1.0, 2.0, 3.0, 4.0])  # one per token: extra, Current, answer:, 42
    span_mean = mod._answer_span_entropy(model, completion, entropies)
    # marker starts at "Current" (index 1) -> span is tokens [Current, answer:, 42] -> mean(2,3,4)
    assert span_mean == 3.0


def test_answer_span_entropy_is_none_when_marker_missing():
    vocab = {"no": 0, "marker": 1, "here": 2}
    tokenizer = _FakeTokenizer(vocab)
    model = _FakeChatModel(tokenizer, seq_len=None)
    entropies = torch.tensor([1.0, 2.0, 3.0])
    assert mod._answer_span_entropy(model, "no marker here", entropies) is None


def test_answer_span_entropy_is_none_on_empty_completion():
    vocab = {"Current": 0, "answer:": 1}
    tokenizer = _FakeTokenizer(vocab)
    model = _FakeChatModel(tokenizer, seq_len=None)
    assert mod._answer_span_entropy(model, "Current answer:", torch.zeros(0)) is None
