"""Scene cut count (step 2.3) with PySceneDetect's ContentDetector.

Frames come from ffmpeg at low resolution and a reduced frame rate (fast, and safe for
non-ASCII paths); PySceneDetect only does the detection.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
from scenedetect import ContentDetector
from scenedetect.common import FrameTimecode

from clipsdex_engine import media

ANALYSIS_FPS = 15.0
ANALYSIS_WIDTH = 256
THRESHOLD = 27.0  # PySceneDetect's default for ContentDetector
MIN_SCENE_S = 0.5  # ignore "cuts" closer than this (flashes, fast motion)


def count_cuts_in_frames(frames: Iterable[np.ndarray], fps: float = ANALYSIS_FPS) -> int:
    detector = ContentDetector(threshold=THRESHOLD, min_scene_len=max(1, round(MIN_SCENE_S * fps)))
    cuts = 0
    last: FrameTimecode | None = None
    for index, frame in enumerate(frames):
        last = FrameTimecode(index, fps=fps)
        cuts += len(detector.process_frame(last, frame))
    if last is not None:
        cuts += len(detector.post_process(last))
    return cuts


def count_cuts(path: Path, width: int, height: int) -> int:
    """Number of hard cuts in the video (a single continuous shot has 0)."""
    w, h = media.analysis_size(width, height, ANALYSIS_WIDTH)
    return count_cuts_in_frames(media.iter_frames(path, w, h, ANALYSIS_FPS))
