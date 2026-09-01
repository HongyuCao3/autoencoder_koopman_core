from persona_drift.helpfulness_judge import judge_helpfulness_score


class FakeJudge:
    def __init__(self, reply="3"):
        self.model_id = "fake-judge"
        self._reply = reply
        self.calls = 0
        self.enable_thinking_calls = []

    def generate(self, messages, seed, config=None, enable_thinking=None, return_thinking=False):
        self.calls += 1
        self.enable_thinking_calls.append(enable_thinking)
        return self._reply


def test_judge_helpfulness_score_parses_digit_and_normalizes():
    y_help, failure, raw = judge_helpfulness_score(
        FakeJudge("5"), "how does photosynthesis work?", "a complete, correct explanation", seed=0
    )
    assert failure is False
    assert y_help == 1.0
    assert raw == "5"


def test_judge_helpfulness_score_normalizes_low_end():
    y_help, failure, raw = judge_helpfulness_score(
        FakeJudge("1"), "how does photosynthesis work?", "I'd rather not get into that.", seed=0
    )
    assert failure is False
    assert y_help == 0.0


def test_judge_helpfulness_score_parses_digit_embedded_in_extra_text():
    y_help, failure, raw = judge_helpfulness_score(FakeJudge("The score is 4 out of 5."), "q", "y", seed=0)
    assert failure is False
    assert y_help == 0.75


def test_judge_helpfulness_score_flags_unparseable_output():
    y_help, failure, raw = judge_helpfulness_score(FakeJudge("no digit here"), "q", "y", seed=0)
    assert failure is True
    assert y_help != y_help  # nan
    assert raw == "no digit here"


def test_judge_helpfulness_score_includes_the_question_in_the_prompt():
    judge = FakeJudge("5")
    captured = {}

    def capturing_generate(messages, seed, config=None, enable_thinking=None, return_thinking=False):
        captured["prompt"] = messages[0]["content"]
        captured["enable_thinking"] = enable_thinking
        return "5"

    judge.generate = capturing_generate
    judge_helpfulness_score(judge, "the current question", "a reply", seed=0)
    assert "the current question" in captured["prompt"]


def test_judge_helpfulness_score_always_pins_enable_thinking_false():
    judge = FakeJudge("5")
    judge_helpfulness_score(judge, "q", "y", seed=0)
    assert judge.enable_thinking_calls == [False]
