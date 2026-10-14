"""Structured (JSON) answers: ask with a schema, validate with Pydantic, retry once."""

from __future__ import annotations

import logging

from pydantic import BaseModel, ValidationError

from clipsdex_engine.llm.base import LLMOutputError, LLMProvider, Message

log = logging.getLogger(__name__)

DEFAULT_MAX_TOKENS = 400  # our JSON answers are short; a stuck model stops in seconds
RETRY_MIN_TEMPERATURE = 0.3  # greedy decoding is what usually loops; vary the retry a bit


def ask_structured[T: BaseModel](
    provider: LLMProvider,
    model: str,
    messages: list[Message],
    schema: type[T],
    *,
    temperature: float = 0.0,
    keep_alive: str | int | None = None,
    max_tokens: int | None = DEFAULT_MAX_TOKENS,
) -> T:
    """Return the model's answer parsed as ``schema``; one retry, then ``LLMOutputError``.

    Retried: invalid JSON / schema (the model is told what was wrong) and a model that got
    stuck repeating itself (asked again with a little randomness).
    """
    json_schema = schema.model_json_schema()
    attempt_messages = list(messages)
    for attempt in (1, 2):
        temp = temperature if attempt == 1 else max(temperature, RETRY_MIN_TEMPERATURE)
        try:
            result = provider.chat(
                model,
                attempt_messages,
                json_schema=json_schema,
                temperature=temp,
                keep_alive=keep_alive,
                max_tokens=max_tokens,
            )
        except LLMOutputError as e:
            log.warning("Unusable output from %s (attempt %d): %s", model, attempt, e)
            attempt_messages = list(messages)
            continue
        try:
            return schema.model_validate_json(result.text)
        except ValidationError as e:
            log.warning("Invalid structured output from %s (attempt %d): %s", model, attempt, e)
            attempt_messages = [
                *messages,
                Message("assistant", result.text),
                Message(
                    "user",
                    "That answer was not valid. Reply again with only JSON matching the "
                    f"schema. Problems: {e.errors(include_url=False)}",
                ),
            ]
    raise LLMOutputError(
        f"The model {model} did not return a usable answer after 2 tries. "
        "A larger model (Detailed preset) may work better."
    )
