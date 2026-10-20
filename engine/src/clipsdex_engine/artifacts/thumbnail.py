"""A small JPEG per video for the library view.

Made once and kept in the video's artifact folder: from the first saved key frame if
there is one (no video decoding needed), else from the video file itself (a frame
early in the clip, where titles often are). A video with neither has no thumbnail.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from sqlalchemy import Engine
from sqlmodel import Session, col, select

from clipsdex_engine import artifacts, media
from clipsdex_engine.config import DataPaths
from clipsdex_engine.db.models import Artifact, Profile, Video

WIDTH = 360
QUALITY = 82
FILE_NAME = "thumb.jpg"


def _resized(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    if w <= WIDTH:
        return image
    return cv2.resize(image, (WIDTH, round(h * WIDTH / w)), interpolation=cv2.INTER_AREA)


def _first_key_frame(session: Session, paths: DataPaths, video_id: int) -> np.ndarray | None:
    found = session.exec(
        select(Artifact)
        .where(Artifact.video_id == video_id, Artifact.type == "frames")
        .order_by(col(Artifact.created_at).desc())
    ).first()
    if found is None or not isinstance(found.data, dict) or not found.data.get("frames"):
        return None
    jpeg = paths.root / found.data["frames"][0]["path"]
    if not jpeg.is_file():
        return None
    # Bytes + imdecode, not cv2.imread: imread fails on non-ASCII Windows paths.
    return cv2.imdecode(np.frombuffer(jpeg.read_bytes(), np.uint8), cv2.IMREAD_COLOR)


def _from_video(video: Video, profile: Profile | None) -> np.ndarray | None:
    if not video.file_path or video.file_deleted:
        return None
    path = Path(video.file_path)
    if not path.is_file():
        return None
    if profile is not None and profile.width and profile.height:
        width, height, duration = profile.width, profile.height, profile.duration_s
    else:
        info = media.probe(path)
        if not (info.has_video and info.width and info.height):
            return None
        width, height, duration = info.width, info.height, info.duration_s
    w, h = media.analysis_size(width, height, WIDTH)
    at = min(1.0, (duration or 0.0) * 0.1)
    return media.frame_at(path, at, w, h)


def thumbnail(engine: Engine, paths: DataPaths, video_id: int) -> Path | None:
    """The thumbnail file, made on first use; None if the video has nothing to show."""
    out = artifacts.artifact_dir(paths, video_id) / FILE_NAME
    if out.is_file():
        return out
    with Session(engine) as session:
        video = session.get(Video, video_id)
        if video is None:
            return None
        image = _first_key_frame(session, paths, video_id)
        if image is None:
            try:
                image = _from_video(video, session.get(Profile, video_id))
            except media.MediaError:
                image = None
    if image is None:
        return None
    ok, buf = cv2.imencode(".jpg", _resized(image), [cv2.IMWRITE_JPEG_QUALITY, QUALITY])
    if not ok:
        return None
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_name(out.name + ".part")
    part.write_bytes(buf.tobytes())
    part.replace(out)
    return out
