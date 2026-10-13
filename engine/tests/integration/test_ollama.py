"""Real Ollama. Run with: uv run pytest -m integration

Needs a running Ollama with gemma3:4b and nomic-embed-text.
"""

import pytest
from pydantic import BaseModel, Field

from clipsdex_engine import llm
from clipsdex_engine.llm import Message

TEXT_MODEL = "gemma3:4b"
EMBED_MODEL = "nomic-embed-text"

_provider = llm.default_provider()
pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True, scope="module")
def _needs_ollama() -> None:  # checked lazily: collecting this module must not touch the network
    if not _provider.health().ok:
        pytest.skip("Ollama is not running")


class Guess(BaseModel):
    category: str
    confidence: float = Field(ge=0, le=1)


def test_models_present() -> None:
    assert _provider.has_model(TEXT_MODEL)
    assert _provider.has_model(EMBED_MODEL)


def test_structured_answer() -> None:
    guess = llm.ask_structured(
        _provider,
        TEXT_MODEL,
        [
            # Categories need descriptions: with bare names, gemma3:4b once read "edit"
            # as "editing advice" and picked it for this caption.
            Message(
                "user",
                "Categories: business = advice about work, money, clients; "
                "edit = a music video edit made from clips; movie = a scene from a film. "
                "Pick one category for a video captioned '3 pricing mistakes freelancers "
                "make'. Give a confidence from 0 to 1.",
            )
        ],
        Guess,
        keep_alive="1m",
    )
    assert guess.category == "business"


def test_embeddings() -> None:
    vectors = _provider.embed(EMBED_MODEL, ["negotiating with clients", "a dog on a beach"])
    assert len(vectors) == 2 and len(vectors[0]) == 768


def test_unload_frees_the_model() -> None:
    _provider.load(TEXT_MODEL, keep_alive="1m")
    _provider.unload(TEXT_MODEL)
