"""Tests for the `core` protocol reconstruction used to drive a second backbone.

The scorer rules were recovered by fitting against the landed corpus, so the tests
pin the *discriminating* cases -- the ones that separate the recovered rule from
the obvious wrong rule -- not just the easy ones.
"""

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

from verify_core_protocol_reconstruction import (  # noqa: E402
    average_word_length,
    character_count,
    check_turn_structure,
    comma_count,
    first_integer,
    word_count,
    word_tokens,
)


class TestWordCount:
    def test_plain_sentence(self):
        assert word_count("Public transport connects cities daily.") == 5

    def test_slash_splits(self):
        """`24/7` is two words -- naive .split() gets this wrong."""
        assert word_count("around 24/7 access") == 4
        assert len("around 24/7 access".split()) == 3

    def test_curly_apostrophe_splits(self):
        """`nature's` with U+2019 is two words."""
        assert word_count("a whisper of nature’s song") == 6

    def test_straight_apostrophe_does_not_split(self):
        assert word_count("residents' lives") == 2

    def test_single_hyphen_is_one_word(self):
        assert word_count("well-being matters") == 2

    def test_double_hyphen_is_two_words(self):
        """`up-to-date` counts 2 -- not 1 (no split) and not 3 (full split)."""
        assert word_count("up-to-date information") == 3

    def test_empty(self):
        assert word_count("") == 0
        assert word_count(None) == 0


class TestOtherScorers:
    def test_character_count_is_unstripped_len(self):
        assert character_count("abc") == 3
        assert character_count(" abc ") == 5

    def test_average_word_length_strips_punctuation(self):
        assert average_word_length("ab, cd.") == 2.0

    def test_average_word_length_empty(self):
        assert average_word_length("") == 0.0

    def test_comma_count(self):
        assert comma_count("a, b, c") == 2

    def test_first_integer_signed(self):
        assert first_integer("the answer is -7 today") == -7

    def test_first_integer_missing(self):
        assert first_integer("no digits") is None

    def test_word_tokens_drops_empties(self):
        assert word_tokens("  a   b  ") == ["a", "b"]


class TestTurnStructure:
    def _traj(self, feedback="fb1", assistant="gen1", append_only=True):
        p1 = [{"role": "user", "content": "start"}]
        tail = [{"role": "assistant", "content": assistant},
                {"role": "user", "content": feedback}]
        p2 = (p1 if append_only else [{"role": "user", "content": "DIFFERENT"}]) + tail
        return [
            {"trajectory_id": "t", "turn": 1, "prompt_messages": p1,
             "raw_generation": "gen1", "feedback_text": "fb1"},
            {"trajectory_id": "t", "turn": 2, "prompt_messages": p2,
             "raw_generation": "gen2", "feedback_text": "fb2"},
        ]

    def test_all_three_conventions_hold(self):
        got = check_turn_structure(self._traj())
        assert got["pairs"] == 1
        assert got["history_is_append_only"] == 1
        assert got["assistant_turn_is_raw_generation"] == 1
        assert got["user_turn_is_feedback_of_previous_row"] == 1

    def test_detects_non_append_only_history(self):
        got = check_turn_structure(self._traj(append_only=False))
        assert got["history_is_append_only"] == 0

    def test_detects_wrong_assistant_source(self):
        got = check_turn_structure(self._traj(assistant="NOT-THE-GENERATION"))
        assert got["assistant_turn_is_raw_generation"] == 0

    def test_detects_wrong_feedback_row(self):
        """Guards the off-by-one that fb[t] vs fb[t+1] would introduce."""
        got = check_turn_structure(self._traj(feedback="fb2"))
        assert got["user_turn_is_feedback_of_previous_row"] == 0

    def test_single_turn_trajectory_yields_no_pairs(self):
        rows = [{"trajectory_id": "t", "turn": 1, "prompt_messages": [],
                 "raw_generation": "g", "feedback_text": "f"}]
        assert check_turn_structure(rows)["pairs"] == 0


@pytest.mark.parametrize("task,field,expected", [
    ("sentence_length_t10", "measured_raw_count", 1.0),
    ("character_length_t5", "measured_raw_count", 1.0),
    ("average_word_length_t5", "measured_raw_value", 1.0),
    ("even_odd_t5", "integer_value", 1.0),
    ("vector_count_stage1_t10", "word_count", 1.0),
])
def test_scorer_fidelity_on_landed_corpus(task, field, expected):
    """Regression: these five reproduce the landed corpus exactly. If a change
    drops one below 1.0, the replay harness is no longer faithful."""
    from verify_core_protocol_reconstruction import check_scorers, load

    rows = load(task)
    assert check_scorers(task, rows)[field]["fidelity"] == expected
