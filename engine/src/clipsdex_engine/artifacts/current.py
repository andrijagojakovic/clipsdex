"""Which version of each artifact the current settings ask for.

An artifact is reused only when its params match, so "is the transcript there?" always
means "is there a transcript made with the current Whisper model?".
"""

from __future__ import annotations

from typing import Any

from clipsdex_engine.artifacts import frame_desc, frames, ocr, transcript
from clipsdex_engine.config import ModelChoice

# Artifacts that analyzers can ask for, in the order they are computed.
NAMES = (frames.TYPE, transcript.TYPE, ocr.TYPE, frame_desc.TYPE)
# Artifacts that are computed from the key frames.
NEEDS_FRAMES = frozenset({ocr.TYPE, frame_desc.TYPE})


def params(name: str, models: ModelChoice) -> dict[str, Any]:
    budget = models.max_frames
    if name == transcript.TYPE:
        return transcript.params_for(models.whisper)
    if name == frames.TYPE:
        return frames.params_for(budget)
    if name == ocr.TYPE:
        return ocr.params_for(budget)
    if name == frame_desc.TYPE:
        return frame_desc.params_for(models.vision, budget)
    raise ValueError(f"Unknown artifact {name!r}. Known: {', '.join(NAMES)}")
