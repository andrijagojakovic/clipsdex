"""Structured (JSON) answers: ask with a schema, validate with Pydantic, retry once."""

from __future__ import annotations

import logging

from pydantic import BaseModel, ValidationError

from clipsdex_engine.llm.base import LLMOutputError, LLMProvider, Message

log = logging.getLogger(__name__)


def ask_structured[T: BaseModel](
    provider: LLMProvider,
    model: str,
    messages: list[Message],
    schema: type[T],
    *,
    temperature: float = 0.0,
    keep_alive: str | int | None = None,
) -> T:
    """Return the model's answer parsed as ``schema``; one retry with the error, then give up."""
    json_schema = schema.model_json_schema()
    attempt_messages = list(messages)
    for attempt in (1, 2):
        result = provider.chat(
            model,
            attempt_messages,
            json_schema=json_schema,
            temperature=temperature,
            keep_alive=keep_alive,
        )
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
