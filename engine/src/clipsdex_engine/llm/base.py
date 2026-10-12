"""The provider interface every LLM/vision/embedding call goes through (CLAUDE.md §8).

Ollama is the only provider now; LiteLLM (cloud, bring-your-own-key) comes later and must
implement the same interface. No module outside ``llm/`` talks to a model server directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class Message:
    role: Role
    content: str
    images: list[bytes] = field(default_factory=list)  # encoded image files (PNG/JPEG)


@dataclass(frozen=True)
class ChatResult:
    text: str
    model: str
    seconds: float | None = None  # total time reported by the provider, if any


@dataclass(frozen=True)
class Health:
    ok: bool
    version: str | None
    message: str  # human-readable, shown to users


class LLMError(RuntimeError):
    """Base class for provider errors. Messages are safe to show to users."""


class ProviderUnavailableError(LLMError):
    """The model server is not running or not reachable."""


class ModelMissingError(LLMError):
    """The requested model is not downloaded."""


class LLMOutputError(LLMError):
    """The model kept returning output we could not use (e.g. invalid JSON)."""


class LLMProvider(Protocol):
    name: str

    def health(self) -> Health: ...

    def list_models(self) -> list[str]:
        """Names of downloaded models, e.g. ``["gemma3:4b", "nomic-embed-text:latest"]``."""
        ...

    def has_model(self, model: str) -> bool: ...

    def chat(
        self,
        model: str,
        messages: list[Message],
        *,
        json_schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
        keep_alive: str | int | None = None,
    ) -> ChatResult:
        """One non-streaming reply. With ``json_schema`` the reply must be JSON matching it."""
        ...

    def embed(self, model: str, texts: list[str]) -> list[list[float]]: ...

    def load(self, model: str, keep_alive: str | int = "10m") -> None:
        """Load a model now and keep it in memory (the scheduler groups work by model, D16)."""
        ...

    def unload(self, model: str) -> None:
        """Free the model's memory right away."""
        ...


def same_model(a: str, b: str) -> bool:
    """``nomic-embed-text`` and ``nomic-embed-text:latest`` are the same model."""

    def norm(name: str) -> str:
        return name if ":" in name else f"{name}:latest"

    return norm(a) == norm(b)
