"""Frame descriptions (step 3.6): one short, factual description per key frame.

One vision-model call per frame, never many images in one call (small local models mix
them up). The model never names or guesses who a person is: movie identification later
works from text, dialogue and scene descriptions only.

A vision call costs ~45-55 s per frame on a CPU-only laptop, so not every key frame is
described: plain frames (one background color, maybe with text on it) are skipped, because
the model can't add anything the OCR has not read already, and at most
``max_described(budget)`` of the rest are described, spread over the video.

Stored inline in ``artifacts.data``:
``{"model", "frames": [{t, description, shot_type} | {t, error}], "skipped": [{t, reason}]}``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import cv2
import numpy as np
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
SELECTION_VERSION = 1  # bump when the frame choice below changes
# A frame is "plain" when this share of its pixels is within PLAIN_TOLERANCE (0-255, per
# channel) of its median color: a text slide, a black screen, a solid color.
PLAIN_SHARE = 0.9
PLAIN_TOLERANCE = 12
PLAIN_REASON = "plain frame (one color, maybe text): the on-screen text covers it"

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
    return {
        "model": model,
        "frames": frames.params_for(budget),
        "prompt": PROMPT_VERSION,
        "max": max_described(budget),
        "select": SELECTION_VERSION,
    }


def max_described(budget: int) -> int:
    """Frames described per video: 2 for the Fast and Balanced budgets (2 and 4 key
    frames), half the key frames for bigger budgets."""
    return min(budget, max(2, budget // 2))


def is_plain(jpeg: bytes) -> bool:
    image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return False
    small = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA).reshape(-1, 3)
    median = np.median(small, axis=0)
    distance = np.abs(small.astype(np.int16) - median).max(axis=1)
    return float((distance <= PLAIN_TOLERANCE).mean()) >= PLAIN_SHARE


def spread(count: int, wanted: int) -> list[int]:
    """``wanted`` indices out of ``count``, evenly spread (the middle of each part)."""
    if count <= wanted:
        return list(range(count))
    return [int((i + 0.5) * count / wanted) for i in range(wanted)]


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

    skipped: list[dict[str, Any]] = []
    candidates: list[tuple[dict[str, Any], bytes]] = []
    for entry in entries:
        jpeg = (paths.root / entry["path"]).read_bytes()
        if is_plain(jpeg):
            skipped.append({"t": entry["t"], "reason": PLAIN_REASON})
        else:
            candidates.append((entry, jpeg))
    chosen = [candidates[i] for i in spread(len(candidates), max_described(budget))]

    described: list[dict[str, Any]] = []
    last_error: llm.LLMError | None = None
    for n, (entry, jpeg) in enumerate(chosen, start=1):
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
            progress(n / max(1, len(chosen)))

    if chosen and last_error is not None and all("error" in d for d in described):
        raise last_error

    data = {"model": model, "frames": described, "skipped": skipped}
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
