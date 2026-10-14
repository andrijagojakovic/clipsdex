"""In-memory LLMProvider for unit tests: no Ollama, no network."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from clipsdex_engine.llm import ChatResult, Health, Message, ModelMissingError
from clipsdex_engine.llm.base import ProviderUnavailableError, same_model

# A reply text, a function of the messages, or an exception to raise (e.g. LLMOutputError).
Reply = str | Callable[[list[Message]], str] | Exception


@dataclass
class ChatCall:
    model: str
    messages: list[Message]
    json_schema: dict[str, Any] | None
    keep_alive: str | int | None
    temperature: float = 0.0
    max_tokens: int | None = None


@dataclass
class FakeLLM:
    """Replies come from ``replies`` in order (see ``Reply``);
    when they run out, ``default_reply`` is used."""

    replies: list[Reply] = field(default_factory=list)
    default_reply: str = "{}"
    models: list[str] = field(default_factory=lambda: ["gemma3:4b", "nomic-embed-text:latest"])
    running: bool = True
    name: str = "fake"
    calls: list[ChatCall] = field(default_factory=list)
    events: list[str] = field(default_factory=list)  # "load gemma3:4b", "unload ..."

    def _check(self, model: str) -> None:
        if not self.running:
            raise ProviderUnavailableError("Fake provider is not running.")
        if not self.has_model(model):
            raise ModelMissingError(f"The model {model} is not downloaded yet.")

    def health(self) -> Health:
        if not self.running:
            return Health(False, None, "Fake provider is not running.")
        return Health(True, "0.0-fake", "Fake provider is running.")

    def list_models(self) -> list[str]:
        if not self.running:
            raise ProviderUnavailableError("Fake provider is not running.")
        return list(self.models)

    def has_model(self, model: str) -> bool:
        return any(same_model(model, m) for m in self.list_models())

    def chat(
        self,
        model: str,
        messages: list[Message],
        *,
        json_schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
        keep_alive: str | int | None = None,
        max_tokens: int | None = None,
    ) -> ChatResult:
        self._check(model)
        self.calls.append(
            ChatCall(model, list(messages), json_schema, keep_alive, temperature, max_tokens)
        )
        reply = self.replies.pop(0) if self.replies else self.default_reply
        if isinstance(reply, Exception):
            raise reply
        text = reply(messages) if callable(reply) else reply
        return ChatResult(text=text, model=model, seconds=0.0)

    def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        """Deterministic 8-dim vectors from a hash of each text."""
        self._check(model)
        return [[b / 255 for b in hashlib.sha256(t.encode()).digest()[:8]] for t in texts]

    def load(self, model: str, keep_alive: str | int = "10m") -> None:
        self._check(model)
        self.events.append(f"load {model}")

    def unload(self, model: str) -> None:
        self._check(model)
        self.events.append(f"unload {model}")
