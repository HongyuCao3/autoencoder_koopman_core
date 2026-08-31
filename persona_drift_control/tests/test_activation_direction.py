import numpy as np
import pytest

from persona_drift.activation_direction import compute_safety_direction


class FakeActivationChatModel:
    """Duck-typed stand-in for ChatModel.hidden_state_at_layer -- returns a
    deterministic activation vector per message text so the diff-in-means
    math is testable without torch or a real model (same pattern as
    tests/test_attack_trajectory.py's FakeChatModel for .generate)."""

    def __init__(self, activation_by_text: dict[str, np.ndarray]):
        self._activation_by_text = activation_by_text
        self.calls: list[tuple[str, int]] = []

    def hidden_state_at_layer(self, messages: list[dict[str, str]], layer: int) -> np.ndarray:
        text = messages[0]["content"]
        self.calls.append((text, layer))
        return self._activation_by_text[text]


def test_direction_points_from_harmful_toward_harmless_mean():
    harmful_texts = ["h1", "h2"]
    harmless_texts = ["s1", "s2"]
    activations = {
        "h1": np.array([1.0, 0.0]),
        "h2": np.array([3.0, 0.0]),
        "s1": np.array([1.0, 10.0]),
        "s2": np.array([3.0, 10.0]),
    }
    chat_model = FakeActivationChatModel(activations)

    direction, stats = compute_safety_direction(chat_model, harmful_texts, harmless_texts, layer=5)

    # harmful mean = [2, 0], harmless mean = [2, 10] -> direction = [0, 10]
    assert direction == pytest.approx(np.array([0.0, 10.0]))
    assert stats["layer"] == 5
    assert stats["n_harmful"] == 2
    assert stats["n_harmless"] == 2
    assert stats["direction_norm"] == pytest.approx(10.0)


def test_projection_stats_separate_the_two_poles_when_direction_is_informative():
    activations = {
        "h1": np.array([0.0, 0.0]),
        "h2": np.array([0.0, 0.0]),
        "s1": np.array([0.0, 5.0]),
        "s2": np.array([0.0, 5.0]),
    }
    chat_model = FakeActivationChatModel(activations)

    _direction, stats = compute_safety_direction(chat_model, ["h1", "h2"], ["s1", "s2"], layer=1)

    assert stats["harmless_mean_projection"] > stats["harmful_mean_projection"]


def test_hidden_state_at_layer_called_once_per_text_with_requested_layer():
    activations = {"h1": np.zeros(3), "h2": np.zeros(3), "s1": np.ones(3), "s2": np.ones(3)}
    chat_model = FakeActivationChatModel(activations)

    compute_safety_direction(chat_model, ["h1", "h2"], ["s1", "s2"], layer=7)

    assert len(chat_model.calls) == 4
    assert all(layer == 7 for _text, layer in chat_model.calls)


def test_raises_on_too_few_texts_per_pole():
    chat_model = FakeActivationChatModel({"h1": np.zeros(2), "s1": np.zeros(2)})
    with pytest.raises(ValueError):
        compute_safety_direction(chat_model, ["h1"], ["s1"], layer=1)
