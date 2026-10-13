"""Quick on-screen text check (step 2.4): OCR on 1-2 frames.

Only answers "is there text, and what does it say". The full OCR artifact over all
selected frames comes in step 3.5.

RapidOCR ships its ONNX models inside the package (PP-OCRv6, Apache-2.0), so nothing is
downloaded at run time.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from clipsdex_engine import media

OCR_WIDTH = 720  # big enough for caption-sized text, small enough to stay fast on CPU
MIN_SCORE = 0.6
MIN_CHARS = 2


class TextReader(Protocol):
    def read(self, image: np.ndarray) -> list[tuple[str, float]]:
        """BGR image -> [(text line, confidence 0-1)]."""
        ...


@dataclass(frozen=True)
class QuickOcr:
    has_text: bool
    text: str | None


class RapidOcrReader:
    def __init__(self) -> None:
        logging.getLogger("RapidOCR").setLevel(logging.ERROR)
        from rapidocr import RapidOCR  # imported lazily: loads onnxruntime

        self._engine: Any = RapidOCR(params={"Global.log_level": "error"})

    def read(self, image: np.ndarray) -> list[tuple[str, float]]:
        result = self._engine(image)
        if result.txts is None:
            return []
        return [(str(t), float(s)) for t, s in zip(result.txts, result.scores, strict=True)]


def sample_times(duration_s: float | None) -> list[float]:
    """Early frame (titles often appear first) and the middle; one frame for short clips."""
    if not duration_s or duration_s <= 0:
        return [0.0]
    early = min(1.0, duration_s * 0.1)
    middle = duration_s * 0.5
    return [early] if middle - early < 1.0 else [early, middle]


def clean_lines(lines: list[tuple[str, float]]) -> list[str]:
    """Confident lines with real content, deduplicated, in reading order."""
    kept: list[str] = []
    seen: set[str] = set()
    for text, score in lines:
        text = " ".join(text.split())
        key = text.casefold()
        if score < MIN_SCORE or sum(c.isalnum() for c in text) < MIN_CHARS or key in seen:
            continue
        seen.add(key)
        kept.append(text)
    return kept


def quick_ocr(path: Path, info: media.MediaInfo, reader: TextReader) -> QuickOcr:
    if not (info.has_video and info.width and info.height):
        return QuickOcr(has_text=False, text=None)
    w, h = media.analysis_size(info.width, info.height, OCR_WIDTH)
    lines: list[tuple[str, float]] = []
    for t in sample_times(info.duration_s):
        frame = media.frame_at(path, t, w, h)
        if frame is not None:
            lines.extend(reader.read(frame))
    kept = clean_lines(lines)
    return QuickOcr(has_text=bool(kept), text="\n".join(kept) or None)
