"""OCR artifact (step 3.5): on-screen text from every selected key frame, deduplicated.

The same caption usually shows up on several frames, and OCR reads it slightly differently
each time ("MIDNIGHT CITY - M83" / "MIDNIGHT CITY -M83"). Lines that are near-identical are
merged; the most confident reading wins.

Stored inline in ``artifacts.data``:
``{"frames": [{t, lines: [{text, score}]}], "lines": [{text, score, first_t, seen}], "text"}``.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

import cv2
import numpy as np
from sqlalchemy import Engine

from clipsdex_engine import artifacts
from clipsdex_engine.artifacts import frames
from clipsdex_engine.config import DataPaths
from clipsdex_engine.db import session_scope
from clipsdex_engine.db.models import Artifact
from clipsdex_engine.profile.quick_ocr import MIN_CHARS, MIN_SCORE, TextReader

TYPE = "ocr"
VERSION = 1
SAME_LINE_RATIO = 0.85  # similarity above which two readings are the same text


class NoFramesError(Exception):
    """The frames artifact this OCR needs does not exist (yet)."""


@dataclass
class _Line:
    text: str
    score: float
    first_t: float
    seen: int


def params_for(budget: int) -> dict[str, Any]:
    return {"frames": frames.params_for(budget), "min_score": MIN_SCORE, "v": VERSION}


def _key(text: str) -> str:
    """Compare text without case, spacing or punctuation differences."""
    return re.sub(r"[\W_]+", "", text.casefold())


def same_text(a: str, b: str) -> bool:
    """Near-identical readings of one line. Numbers must match exactly: "Part 1" and
    "Part 2" (or a film and its sequel) are different text, even though they look alike."""
    if re.findall(r"\d+", a) != re.findall(r"\d+", b):
        return False
    return SequenceMatcher(None, _key(a), _key(b)).ratio() >= SAME_LINE_RATIO


def merge_lines(per_frame: list[tuple[float, list[tuple[str, float]]]]) -> list[_Line]:
    """Merge readings from all frames into unique lines, in order of first appearance."""
    merged: list[_Line] = []
    for t, lines in per_frame:
        for text, score in lines:
            key = _key(text)
            if not key:
                continue
            match = next((m for m in merged if same_text(text, m.text)), None)
            if match is None:
                merged.append(_Line(text, score, t, 1))
                continue
            match.seen += 1
            if score > match.score:  # keep the most confident reading
                match.text, match.score = text, score
    return merged


def read_frames(
    paths: DataPaths,
    frame_entries: list[dict[str, Any]],
    reader: TextReader,
    progress: Callable[[float], None] | None = None,
) -> list[tuple[float, list[tuple[str, float]]]]:
    results: list[tuple[float, list[tuple[str, float]]]] = []
    for n, entry in enumerate(frame_entries, start=1):
        data = (paths.root / entry["path"]).read_bytes()  # not cv2.imread: non-ASCII paths
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        lines = reader.read(image) if image is not None else []
        normalized = [(" ".join(text.split()), score) for text, score in lines]
        kept = [
            (text, score)
            for text, score in normalized
            if score >= MIN_SCORE and sum(c.isalnum() for c in text) >= MIN_CHARS
        ]
        results.append((float(entry["t"]), kept))
        if progress is not None:
            progress(n / max(1, len(frame_entries)))
    return results


def build_ocr(
    engine: Engine,
    paths: DataPaths,
    video_id: int,
    budget: int,
    reader: TextReader,
    progress: Callable[[float], None] | None = None,
) -> dict[str, Any]:
    with session_scope(engine) as session:
        frames_artifact = artifacts.find(session, video_id, frames.TYPE, frames.params_for(budget))
        if frames_artifact is None or not isinstance(frames_artifact.data, dict):
            raise NoFramesError("Key frames have not been selected for this video yet.")
        entries: list[dict[str, Any]] = list(frames_artifact.data.get("frames", []))

    per_frame = read_frames(paths, entries, reader, progress)
    merged = merge_lines(per_frame)
    data = {
        "frames": [
            {"t": t, "lines": [{"text": text, "score": round(s, 3)} for text, s in lines]}
            for t, lines in per_frame
        ],
        "lines": [
            {"text": m.text, "score": round(m.score, 3), "first_t": m.first_t, "seen": m.seen}
            for m in merged
        ],
        "text": "\n".join(m.text for m in merged),
    }

    params = params_for(budget)
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
