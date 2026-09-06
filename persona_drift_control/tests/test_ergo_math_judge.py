from persona_drift.ergo_math_bank import GSM8KShardedItem
from persona_drift.ergo_math_judge import extract_answer_by_regex, judge_math_answer


def _entry(gold="3360"):
    return GSM8KShardedItem(item_id="ergo_GSM8K_1246", gold_answer=gold, shards=("a", "b", "c", "d"))


def test_extract_answer_by_regex_anchored_format():
    assert extract_answer_by_regex("Some reasoning.\nCurrent answer: 3360") == "3360"
    assert extract_answer_by_regex("Current answer: -12.5") == "-12.5"


def test_extract_answer_by_regex_falls_back_to_permissive_search():
    assert extract_answer_by_regex("I think the total is 3360 points.") == "3360"


def test_extract_answer_by_regex_returns_none_when_nothing_found():
    assert extract_answer_by_regex("I'm not sure yet, I need more information.") is None


def test_judge_math_answer_correct():
    score, parse_failure, raw = judge_math_answer(
        None, _entry(gold="3360"), [], 4, "clue", "Current answer: 3360", 0, None
    )
    assert score == 1.0
    assert parse_failure is False
    assert raw == "3360"


def test_judge_math_answer_incorrect():
    score, parse_failure, raw = judge_math_answer(
        None, _entry(gold="3360"), [], 4, "clue", "Current answer: 100", 0, None
    )
    assert score == 0.0
    assert parse_failure is False


def test_judge_math_answer_parse_failure_scores_zero():
    score, parse_failure, raw = judge_math_answer(
        None, _entry(gold="3360"), [], 1, "clue", "I don't have enough information yet.", 0, None
    )
    assert score == 0.0
    assert parse_failure is True
    assert raw == ""


def test_judge_math_answer_normalizes_currency_and_commas():
    score, _parse_failure, _raw = judge_math_answer(
        None, _entry(gold="3360"), [], 4, "clue", "Current answer: $3,360.00", 0, None
    )
    assert score == 1.0
