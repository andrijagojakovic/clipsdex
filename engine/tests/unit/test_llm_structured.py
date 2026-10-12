import pytest
from pydantic import BaseModel, Field

from clipsdex_engine.llm import LLMOutputError, Message, ask_structured
from mocks.fake_llm import FakeLLM


class Guess(BaseModel):
    category: str
    confidence: float = Field(ge=0, le=1)


MESSAGES = [Message("user", "Categorize: 3 pricing mistakes freelancers make")]


def test_valid_first_answer() -> None:
    fake = FakeLLM(replies=['{"category": "business", "confidence": 0.9}'])
    guess = ask_structured(fake, "gemma3:4b", MESSAGES, Guess)
    assert guess == Guess(category="business", confidence=0.9)
    assert len(fake.calls) == 1
    assert fake.calls[0].json_schema == Guess.model_json_schema()


def test_retries_once_with_the_problem_explained() -> None:
    fake = FakeLLM(
        replies=[
            '{"category": "business", "confidence": 7}',  # out of range
            '{"category": "business", "confidence": 0.7}',
        ]
    )
    guess = ask_structured(fake, "gemma3:4b", MESSAGES, Guess)
    assert guess.confidence == 0.7
    retry = fake.calls[1].messages
    assert retry[: len(MESSAGES)] == MESSAGES
    assert retry[-2].role == "assistant"
    assert "not valid" in retry[-1].content and "confidence" in retry[-1].content


def test_gives_up_after_two_bad_answers() -> None:
    fake = FakeLLM(replies=["not json", '{"category": 1}'])
    with pytest.raises(LLMOutputError, match="did not return a usable answer"):
        ask_structured(fake, "gemma3:4b", MESSAGES, Guess)
    assert len(fake.calls) == 2


def test_fake_llm_reply_functions_and_events() -> None:
    fake = FakeLLM(replies=[lambda msgs: f'{{"category": "{msgs[-1].content}", "confidence": 1}}'])
    assert ask_structured(fake, "gemma3:4b", [Message("user", "edit")], Guess).category == "edit"
    fake.load("gemma3:4b")
    fake.unload("gemma3:4b")
    assert fake.events == ["load gemma3:4b", "unload gemma3:4b"]
