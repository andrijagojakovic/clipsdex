"""Frame descriptions (step 3.6): one short, factual description per key frame.

One vision-model call per frame, never many images in one call (small local models mix
them up). The model never names or guesses who a person is: movie identification later
works from text, dialogue and scene descriptions only.

Stored inline in ``artifacts.data``:
``{"model", "frames": [{t, description, shot_type} | {t, error}]}``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, Field
from sqlalchemy import Engine

from clipsdex_engine import artifacts, llm
from clipsdex_engine.artifacts import frames
from clipsdex_engine.config import DataPaths
from clipsdex_engine.db import session_scope
from clipsdex_engine.db.models import Artifact
from clipsdex_engine.llm import LLMProvider, Message

TYPE = "frame_descriptions"
PROMPT_VERSION = 2
# On the reference CPU laptop gemma3:4b spends ~65-70 s encoding each new image (it is
# resized to 896x896 inside the model, so sending smaller frames does not help) plus
# ~1 s per 8 output tokens. Keep answers short.
MAX_TOKENS = 120

ShotType = Literal[
    "person talking to camera",
    "film or TV scene",
    "text or slide",
    "screen recording",
    "sports",
    "food",
    "nature or travel",
    "music performance",
    "animation or game",
    "other",
]

PROMPT = """\
This is one frame from a short social media video. Describe what is visible, factually.
- One sentence, at most 25 words: the setting, what people are doing, notable objects,
  clothing or era if it stands out, the look of the image (film, phone video, animation).
- Do not name, identify or guess who any person is, even if they look famous.
- If there is text in the image, just say there is text; do not copy it.
- Pick the shot type that fits best."""


class NoFramesError(Exception):
    """The frames artifact this step needs does not exist (yet)."""


class FrameDescription(BaseModel):
    description: str = Field(max_length=300)
    shot_type: ShotType


def params_for(model: str, budget: int) -> dict[str, Any]:
    return {"model": model, "frames": frames.params_for(budget), "prompt": PROMPT_VERSION}


def describe_one(provider: LLMProvider, model: str, jpeg: bytes) -> FrameDescription:
    return llm.ask_structured(
        provider,
        model,
        [Message("user", PROMPT, images=[jpeg])],
        FrameDescription,
        keep_alive="10m",
        max_tokens=MAX_TOKENS,
    )


def build_descriptions(
    engine: Engine,
    paths: DataPaths,
    video_id: int,
    budget: int,
    provider: LLMProvider,
    model: str,
    progress: Callable[[float], None] | None = None,
) -> dict[str, Any]:
    """Describe every key frame. A frame the model can't handle is recorded with its error;
    the job fails only if no frame could be described."""
    with session_scope(engine) as session:
        found = artifacts.find(session, video_id, frames.TYPE, frames.params_for(budget))
        if found is None or not isinstance(found.data, dict):
            raise NoFramesError("Key frames have not been selected for this video yet.")
        entries: list[dict[str, Any]] = list(found.data.get("frames", []))

    described: list[dict[str, Any]] = []
    last_error: llm.LLMError | None = None
    for n, entry in enumerate(entries, start=1):
        jpeg = (paths.root / entry["path"]).read_bytes()
        try:
            answer = describe_one(provider, model, jpeg)
        except llm.ProviderUnavailableError:
            raise  # the server is gone: retry the whole job later
        except llm.LLMError as e:
            last_error = e
            described.append({"t": entry["t"], "error": str(e)})
        else:
            described.append(
                {
                    "t": entry["t"],
                    "description": answer.description.strip(),
                    "shot_type": answer.shot_type,
                }
            )
        if progress is not None:
            progress(n / max(1, len(entries)))

    if entries and last_error is not None and all("error" in d for d in described):
        raise last_error

    data = {"model": model, "frames": described}
    params = params_for(model, budget)
    with session_scope(engine) as session:
        artifact = artifacts.find(session, video_id, TYPE, params) or Artifact(
            video_id=video_id,
            type=TYPE,
            params_hash=artifacts.params_hash(params),
            params=params,
        )
        artifact.data = data
        session.add(artifact)
    return data
