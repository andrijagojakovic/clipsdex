"""Unit-test guard rails."""

import pytest

from clipsdex_engine import llm
from mocks.fake_llm import FakeLLM


@pytest.fixture(autouse=True)
def _no_real_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unit tests never talk to a real Ollama. Tests that need an LLM pass
    a FakeLLM explicitly or patch ``llm.default_provider`` themselves."""
    monkeypatch.setattr(llm, "default_provider", lambda *_a, **_k: FakeLLM(running=False))
