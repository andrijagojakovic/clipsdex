"""Level 1 quick profile (CLAUDE.md §5.1): cheap, automatic facts about every video.

Implemented so far: ffprobe metadata and the audio artifact (2.1).
Later steps add audio classes (2.2), cut count (2.3), quick OCR (2.4), category guess (2.6).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy import Engine
from sqlmodel import Session, col, select

from clipsdex_engine import media
from clipsdex_engine.artifacts import audio
from clipsdex_engine.config import DataPaths
from clipsdex_engine.db import session_scope
from clipsdex_engine.db.models import Profile, Video, utcnow

PROFILE_VERSION = 1


@dataclass
class ProfileReport:
    profiled: list[int] = field(default_factory=list)
    failed: list[tuple[int, str]] = field(default_factory=list)


def local_file(video: Video) -> Path | None:
    if video.file_path is None or video.file_deleted:
        return None
    return Path(video.file_path)


def profile_video(session: Session, paths: DataPaths, video: Video) -> Profile:
    """Build (or rebuild) the quick profile of one video. Raises MediaError on bad files."""
    assert video.id is not None
    path = local_file(video)
    if path is None:
        raise media.MediaError("This video has no local file to analyze.")
    if not path.is_file():
        raise media.MediaError(f"The file is no longer at {path}. Was it moved or deleted?")

    info = media.probe(path)
    if info.has_audio:
        audio.ensure_audio(session, paths, video, path)

    profile = session.get(Profile, video.id) or Profile(video_id=video.id)
    profile.duration_s = info.duration_s
    profile.width = info.width
    profile.height = info.height
    profile.has_audio = info.has_audio
    profile.profile_version = PROFILE_VERSION
    profile.created_at = utcnow()
    session.add(profile)
    session.flush()
    return profile


def videos_to_profile(
    session: Session, video_ids: Iterable[int] | None = None, redo: bool = False
) -> list[Video]:
    """Videos with a local file that have no profile yet (or all of them with ``redo``)."""
    query = select(Video).where(
        col(Video.file_path).is_not(None), col(Video.file_deleted).is_(False)
    )
    if video_ids is not None:
        query = query.where(col(Video.id).in_(list(video_ids)))
    if not redo:
        query = query.where(col(Video.id).not_in(select(Profile.video_id)))
    return list(session.exec(query.order_by(col(Video.id))))


def run_quick_profile(
    engine: Engine,
    paths: DataPaths,
    video_ids: Iterable[int] | None = None,
    redo: bool = False,
) -> ProfileReport:
    report = ProfileReport()
    with Session(engine) as session:
        ids = [v.id for v in videos_to_profile(session, video_ids, redo) if v.id is not None]

    for video_id in ids:
        # One transaction per video: a broken file never undoes the others.
        try:
            with session_scope(engine) as session:
                profile_video(session, paths, session.get_one(Video, video_id))
        except media.MediaError as e:
            report.failed.append((video_id, str(e)))
        else:
            report.profiled.append(video_id)
    return report
