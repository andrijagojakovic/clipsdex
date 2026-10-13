"""Frames artifact (step 3.4): a few representative key frames per video.

1. Candidates: the middle of every scene (PySceneDetect on small ffmpeg-decoded frames),
   plus one frame every ``LONG_SCENE_STEP_S`` inside long scenes.
2. Near-duplicates are dropped with a perceptual hash, so a static talking-head video ends
   up with 1-3 frames while a fast-cut edit keeps many.
3. At most ``budget`` frames remain (the preset's frame budget), spread over the video.

Selected frames are saved as JPEG (720 px wide) for OCR and the vision model.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scenedetect import ContentDetector
from scenedetect.common import FrameTimecode
from sqlalchemy import Engine

from clipsdex_engine import artifacts, media
from clipsdex_engine.config import DataPaths
from clipsdex_engine.db import session_scope
from clipsdex_engine.db.models import Artifact, Video

TYPE = "frames"
VERSION = 1
SCAN_WIDTH = 256  # scene detection runs on small frames
SCAN_FPS = 10.0
SAVE_WIDTH = 720  # saved frames: readable text, small enough for CPU vision models
LONG_SCENE_STEP_S = 10.0
DUPLICATE_DISTANCE = 10  # max Hamming distance (of 64 bits) for "the same picture"
JPEG_QUALITY = 90


class NoVideoFileError(Exception):
    """Frames need the video itself; it was deleted or moved."""


@dataclass(frozen=True)
class Candidate:
    t: float  # seconds
    scene: int
    thumb: np.ndarray  # 32x32 grayscale, enough for the perceptual hash


def thumbnail(image: np.ndarray) -> np.ndarray:
    """32x32 grayscale. Scan frames are reduced to this at once (~1 KB each, not ~350 KB)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)


def phash(image: np.ndarray) -> int:
    """64-bit perceptual hash: low-frequency DCT of a 32x32 grayscale image vs. its median."""
    small = thumbnail(image).astype(np.float32)
    low = cv2.dct(small)[:8, :8].flatten()
    bits = low[1:] > np.median(low[1:])  # skip the DC term (overall brightness)
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def params_for(budget: int) -> dict[str, Any]:
    return {"budget": budget, "width": SAVE_WIDTH, "dedupe": DUPLICATE_DISTANCE, "v": VERSION}


def scan(
    frames: Iterable[np.ndarray],
    fps: float = SCAN_FPS,
    progress: Callable[[int], None] | None = None,
) -> list[Candidate]:
    """One pass over the scan frames: find scene cuts and keep candidate frames."""
    detector = ContentDetector(min_scene_len=max(1, round(0.5 * fps)))
    step = max(1, round(LONG_SCENE_STEP_S * fps))
    scenes: list[list[tuple[int, np.ndarray]]] = [[]]
    last: FrameTimecode | None = None
    for index, frame in enumerate(frames):
        last = FrameTimecode(index, fps=fps)
        for cut in detector.process_frame(last, frame):
            cut_index = cut.frame_num
            # Frames already stored at/after the cut belong to the new scene.
            moved = [(i, f) for i, f in scenes[-1] if i >= cut_index]
            scenes[-1] = [(i, f) for i, f in scenes[-1] if i < cut_index]
            scenes.append(moved)
        scenes[-1].append((index, thumbnail(frame)))
        if progress is not None and index % 50 == 0:
            progress(index)
    if last is not None:
        detector.post_process(last)  # flushes a pending cut; it can't start a frame scene

    candidates: list[Candidate] = []
    for scene_no, scene in enumerate(s for s in scenes if s):
        picks = {len(scene) // 2}  # the middle: away from cuts and transitions
        picks.update(range(step // 2, len(scene), step))  # extra frames in long scenes
        for pos in sorted(picks):
            index, thumb = scene[pos]
            candidates.append(Candidate(index / fps, scene_no, thumb))
    return candidates


def select(candidates: list[Candidate], budget: int) -> list[Candidate]:
    """Drop near-duplicates (keep the first), then spread at most ``budget`` over time."""
    kept: list[Candidate] = []
    hashes: list[int] = []
    for c in sorted(candidates, key=lambda c: c.t):
        h = phash(c.thumb)
        if all(hamming(h, other) > DUPLICATE_DISTANCE for other in hashes):
            kept.append(c)
            hashes.append(h)
    if len(kept) <= budget:
        return kept
    # Evenly spaced picks keep the beginning, the end and everything in between covered.
    positions = np.linspace(0, len(kept) - 1, budget).round().astype(int)
    return [kept[i] for i in dict.fromkeys(positions.tolist())]


def _source(engine: Engine, video_id: int) -> Path:
    with session_scope(engine) as session:
        video = session.get_one(Video, video_id)
        path = None if video.file_deleted or not video.file_path else Path(video.file_path)
    if path is None or not path.is_file():
        raise NoVideoFileError(
            "Frames need the video file, but it was deleted or moved. Import it again."
        )
    return path


def build_frames(
    engine: Engine,
    paths: DataPaths,
    video_id: int,
    budget: int,
    progress: Callable[[float], None] | None = None,
) -> dict[str, Any]:
    """Select and save key frames for one video; returns the artifact data."""
    source = _source(engine, video_id)
    info = media.probe(source)
    if not (info.has_video and info.width and info.height):
        raise media.MediaError("This file has no video picture to take frames from.")

    total = max(1.0, (info.duration_s or 1.0) * SCAN_FPS)
    scan_w, scan_h = media.analysis_size(info.width, info.height, SCAN_WIDTH)
    candidates = scan(
        media.iter_frames(source, scan_w, scan_h, SCAN_FPS),
        progress=(lambda i: progress(min(0.8, i / total * 0.8))) if progress else None,
    )
    chosen = select(candidates, budget)

    params = params_for(budget)
    digest = artifacts.params_hash(params)
    folder = artifacts.artifact_dir(paths, video_id) / f"{TYPE}-{digest[:8]}"
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob("*.jpg"):
        old.unlink()  # a redo replaces the previous selection
    save_w, save_h = media.analysis_size(info.width, info.height, SAVE_WIDTH)
    saved: list[dict[str, Any]] = []
    for n, c in enumerate(chosen, start=1):
        image = media.frame_at(source, c.t, save_w, save_h)
        if image is None:
            continue
        ok, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok:
            continue
        file = folder / f"frame_{n:02d}_{c.t:07.2f}s.jpg"
        file.write_bytes(jpeg.tobytes())  # cv2.imwrite fails on non-ASCII paths
        saved.append(
            {
                "t": round(c.t, 2),
                "scene": c.scene,
                "path": artifacts.to_stored_path(paths, file),
                "phash": f"{phash(image):016x}",
            }
        )
        if progress is not None:
            progress(0.8 + 0.2 * n / max(1, len(chosen)))

    data = {
        "frames": saved,
        "scenes": len({c.scene for c in candidates}),
        "candidates": len(candidates),
        "budget": budget,
    }
    with session_scope(engine) as session:
        artifact = artifacts.find(session, video_id, TYPE, params) or Artifact(
            video_id=video_id, type=TYPE, params_hash=digest, params=params
        )
        artifact.path = artifacts.to_stored_path(paths, folder)
        artifact.data = data
        session.add(artifact)
    return data
