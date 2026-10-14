"""Ollama provider, the local default. API: https://github.com/ollama/ollama/blob/main/docs/api.md"""

from __future__ import annotations

import base64
from typing import Any

import httpx

from clipsdex_engine import net
from clipsdex_engine.llm.base import (
    ChatResult,
    Health,
    LLMError,
    LLMOutputError,
    Message,
    ModelMissingError,
    ProviderUnavailableError,
    same_model,
)

DEFAULT_URL = "http://127.0.0.1:11434"
# Loading a model from disk on CPU can take 15+ s and a long answer much more.
TIMEOUT = httpx.Timeout(connect=5.0, read=600.0, write=60.0, pool=5.0)


class OllamaProvider:
    name = "ollama"

    def __init__(self, base_url: str = DEFAULT_URL, transport: httpx.BaseTransport | None = None):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(
            base_url=self.base_url,
            headers={"User-Agent": net.USER_AGENT},
            timeout=TIMEOUT,
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def _request(self, method: str, path: str, json: dict[str, Any] | None = None) -> Any:
        try:
            response = self._client.request(method, path, json=json)
        except httpx.ConnectError as e:
            raise ProviderUnavailableError(
                f"Can't reach Ollama at {self.base_url}. Is the Ollama app running?"
            ) from e
        except httpx.TimeoutException as e:
            raise LLMError(
                "Ollama took too long to answer. Try again, or use a faster preset."
            ) from e
        except httpx.HTTPError as e:
            raise LLMError(f"Talking to Ollama failed: {e}") from e

        if response.status_code == 404 and json and "model" in json:
            raise ModelMissingError(
                f"The model {json['model']} is not downloaded yet. "
                f"Download it with: ollama pull {json['model']}"
            )
        if response.is_error:
            detail = _error_detail(response)
            if "repeat limit" in detail:
                # The model got stuck repeating itself: a bad answer, worth another try.
                raise LLMOutputError("The model got stuck repeating itself.")
            raise LLMError(f"Ollama returned an error ({response.status_code}): {detail}")
        return response.json()

    def health(self) -> Health:
        try:
            version = self._request("GET", "/api/version").get("version")
        except LLMError as e:
            return Health(ok=False, version=None, message=str(e))
        return Health(ok=True, version=version, message=f"Ollama {version} is running.")

    def list_models(self) -> list[str]:
        return [m["name"] for m in self._request("GET", "/api/tags").get("models", [])]

    def has_model(self, model: str) -> bool:
        return any(same_model(model, name) for name in self.list_models())

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
        options: dict[str, Any] = {"temperature": temperature}
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        body: dict[str, Any] = {
            "model": model,
            "messages": [_message_json(m) for m in messages],
            "stream": False,
            "options": options,
        }
        if json_schema is not None:
            body["format"] = json_schema
        if keep_alive is not None:
            body["keep_alive"] = keep_alive
        data = self._request("POST", "/api/chat", body)
        total_ns = data.get("total_duration")
        return ChatResult(
            text=data.get("message", {}).get("content", ""),
            model=data.get("model", model),
            seconds=total_ns / 1e9 if total_ns else None,
        )

    def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        data = self._request("POST", "/api/embed", {"model": model, "input": texts})
        return data["embeddings"]

    def load(self, model: str, keep_alive: str | int = "10m") -> None:
        self._request("POST", "/api/generate", {"model": model, "keep_alive": keep_alive})

    def unload(self, model: str) -> None:
        self._request("POST", "/api/generate", {"model": model, "keep_alive": 0})


def _message_json(message: Message) -> dict[str, Any]:
    out: dict[str, Any] = {"role": message.role, "content": message.content}
    if message.images:
        out["images"] = [base64.b64encode(img).decode("ascii") for img in message.images]
    return out


def _error_detail(response: httpx.Response) -> str:
    try:
        return str(response.json().get("error", response.text))
    except ValueError:
        return response.text[:200]
