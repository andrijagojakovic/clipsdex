"""LLM access. Everything goes through an ``LLMProvider`` (see ``base.py``)."""

from __future__ import annotations

import os

from clipsdex_engine.llm.base import (
    ChatResult,
    Health,
    LLMError,
    LLMOutputError,
    LLMProvider,
    Message,
    ModelMissingError,
    ProviderUnavailableError,
)
from clipsdex_engine.llm.ollama import DEFAULT_URL, OllamaProvider
from clipsdex_engine.llm.structured import ask_structured

OLLAMA_URL_ENV = "CLIPSDEX_OLLAMA_URL"

__all__ = [
    "ChatResult",
    "Health",
    "LLMError",
    "LLMOutputError",
    "LLMProvider",
    "Message",
    "ModelMissingError",
    "ProviderUnavailableError",
    "ask_structured",
    "default_provider",
]


def default_provider(base_url: str | None = None) -> LLMProvider:
    """Ollama at ``base_url``, else ``CLIPSDEX_OLLAMA_URL``, else the standard local port.

    (Step 8.2 adds an app-managed runtime on its own port.)
    """
    return OllamaProvider(base_url or os.environ.get(OLLAMA_URL_ENV) or DEFAULT_URL)
