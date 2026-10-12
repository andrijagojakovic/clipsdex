import base64
import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from clipsdex_engine.llm import Message, ModelMissingError, ProviderUnavailableError
from clipsdex_engine.llm.base import LLMError, same_model
from clipsdex_engine.llm.ollama import OllamaProvider

Handler = Callable[[httpx.Request], httpx.Response]


def provider(handler: Handler) -> tuple[OllamaProvider, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    return OllamaProvider("http://ollama.test:11434/", transport=httpx.MockTransport(record)), seen


def body(request: httpx.Request) -> dict[str, Any]:
    return json.loads(request.content)


def test_chat_request_and_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "gemma3:4b",
                "message": {"role": "assistant", "content": '{"a": 1}'},
                "done": True,
                "total_duration": 2_500_000_000,
            },
        )

    p, seen = provider(handler)
    schema = {"type": "object", "properties": {"a": {"type": "integer"}}}
    result = p.chat(
        "gemma3:4b",
        [Message("system", "be brief"), Message("user", "describe", images=[b"\x89PNG"])],
        json_schema=schema,
        keep_alive="5m",
    )

    assert (result.text, result.model, result.seconds) == ('{"a": 1}', "gemma3:4b", 2.5)
    sent = body(seen[0])
    assert seen[0].url.path == "/api/chat"
    assert sent["stream"] is False and sent["format"] == schema and sent["keep_alive"] == "5m"
    assert sent["options"] == {"temperature": 0.0}
    assert sent["messages"][0] == {"role": "system", "content": "be brief"}
    assert sent["messages"][1]["images"] == [base64.b64encode(b"\x89PNG").decode()]
    assert seen[0].headers["User-Agent"].startswith("ClipsDex/")


def test_missing_model_is_a_clear_error() -> None:
    p, _ = provider(lambda _r: httpx.Response(404, json={"error": "model 'x:1b' not found"}))
    with pytest.raises(ModelMissingError, match="ollama pull x:1b"):
        p.chat("x:1b", [Message("user", "hi")])


def test_server_not_running() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    p, _ = provider(refuse)
    with pytest.raises(ProviderUnavailableError, match="Is the Ollama app running"):
        p.list_models()
    health = p.health()
    assert not health.ok and "Ollama" in health.message


def test_other_errors_carry_ollama_message() -> None:
    p, _ = provider(lambda _r: httpx.Response(500, json={"error": "out of memory"}))
    with pytest.raises(LLMError, match="out of memory"):
        p.chat("gemma3:4b", [Message("user", "hi")])


def test_health_and_models() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "0.35.1"})
        return httpx.Response(
            200, json={"models": [{"name": "gemma3:4b"}, {"name": "nomic-embed-text:latest"}]}
        )

    p, _ = provider(handler)
    assert p.health().ok and p.health().version == "0.35.1"
    assert p.has_model("nomic-embed-text")  # ":latest" is implied
    assert p.has_model("gemma3:4b")
    assert not p.has_model("gemma3:12b")


def test_embed() -> None:
    p, seen = provider(
        lambda _r: httpx.Response(200, json={"embeddings": [[0.1, 0.2], [0.3, 0.4]]})
    )
    assert p.embed("nomic-embed-text", ["a", "b"]) == [[0.1, 0.2], [0.3, 0.4]]
    assert body(seen[0]) == {"model": "nomic-embed-text", "input": ["a", "b"]}
    assert p.embed("nomic-embed-text", []) == []
    assert len(seen) == 1  # no request for empty input


def test_load_and_unload_use_keep_alive() -> None:
    p, seen = provider(lambda _r: httpx.Response(200, json={"done": True}))
    p.load("gemma3:4b", keep_alive="30m")
    p.unload("gemma3:4b")
    assert [body(r)["keep_alive"] for r in seen] == ["30m", 0]
    assert all(r.url.path == "/api/generate" for r in seen)


@pytest.mark.parametrize(
    ("a", "b", "same"),
    [
        ("x", "x:latest", True),
        ("x:latest", "x", True),
        ("x:4b", "x", False),
        ("x:4b", "x:4b", True),
    ],
)
def test_same_model(a: str, b: str, same: bool) -> None:
    assert same_model(a, b) is same
