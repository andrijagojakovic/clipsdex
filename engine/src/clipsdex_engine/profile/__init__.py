"""Level 1 quick profile (CLAUDE.md §5.1): cheap, automatic facts about every video.

Implemented so far: ffprobe metadata and the audio artifact (2.1), speech/music/silence
ratios and the vocals flag (2.2), cut count (2.3). Later steps add quick OCR (2.4) and the
category guess (2.6).

Heavy models are loaded at most once per run, and only when needed (D16).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy import Engine
from sqlmodel import Session, col, select

from clipsdex_engine import media, net
from clipsdex_engine.artifacts import audio
from clipsdex_engine.config import DataPaths
from clipsdex_engine.db import session_scope
from clipsdex_engine.db.models import Profile, Video, utcnow
from clipsdex_engine.models import panns
from clipsdex_engine.profile import audio_classes, cuts
from clipsdex_engine.profile.audio_classes import AudioTagger

# Bump when a step adds or changes profile fields: older profiles are then rebuilt.
PROFILE_VERSION = 3


@dataclass
class ProfileReport:
    profiled: list[int] = field(default_factory=list)
    failed: list[tuple[int, str]] = field(default_factory=list)


class LazyTagger:
    """Creates the audio tagger on first use, so runs without audio never load the model."""

    def __init__(self, factory: Callable[[], AudioTagger]) -> None:
        self._factory = factory
        self._tagger: AudioTagger | None = None

    def get(self) -> AudioTagger:
        if self._tagger is None:
            self._tagger = self._factory()
        return self._tagger


def default_tagger(paths: DataPaths, progress: net.Progress | None = None) -> LazyTagger:
    def load() -> AudioTagger:
        return panns.PannsTagger(panns.ensure_checkpoint(paths.models, progress))

    return LazyTagger(load)


def local_file(video: Video) -> Path | None:
    if video.file_path is None or video.file_deleted:
        return None
    return Path(video.file_path)


def profile_video(session: Session, paths: DataPaths, video: Video, tagger: LazyTagger) -> Profile:
    """Build (or rebuild) the quick profile of one video. Raises MediaError on bad files."""
    assert video.id is not None
    path = local_file(video)
    if path is None:
        raise media.MediaError("This video has no local file to analyze.")
    if not path.is_file():
        raise media.MediaError(f"The file is no longer at {path}. Was it moved or deleted?")

    info = media.probe(path)
    profile = session.get(Profile, video.id) or Profile(video_id=video.id)
    profile.duration_s = info.duration_s
    profile.width = info.width
    profile.height = info.height
    profile.has_audio = info.has_audio
    profile.cut_count = (
        cuts.count_cuts(path, info.width, info.height)
        if info.has_video and info.width and info.height
        else None
    )

    if info.has_audio:
        artifact = audio.ensure_audio(session, paths, video, path)
        samples = audio_classes.read_wav(audio.audio_path(paths, artifact))
        classes = audio_classes.classify(samples, tagger.get())
        profile.speech_ratio = classes.speech_ratio
        profile.music_ratio = classes.music_ratio
        profile.silence_ratio = classes.silence_ratio
        profile.has_vocals = classes.has_vocals
    else:
        profile.speech_ratio = 0.0
        profile.music_ratio = 0.0
        profile.silence_ratio = 1.0
        profile.has_vocals = None

    profile.profile_version = PROFILE_VERSION
    profile.created_at = utcnow()
    session.add(profile)
    session.flush()
    return profile


def videos_to_profile(
    session: Session, video_ids: Iterable[int] | None = None, redo: bool = False
) -> list[Video]:
    """Videos with a local file and no up-to-date profile (or all of them with ``redo``)."""
    query = select(Video).where(
        col(Video.file_path).is_not(None), col(Video.file_deleted).is_(False)
    )
    if video_ids is not None:
        query = query.where(col(Video.id).in_(list(video_ids)))
    if not redo:
        current = select(Profile.video_id).where(Profile.profile_version >= PROFILE_VERSION)
        query = query.where(col(Video.id).not_in(current))
    return list(session.exec(query.order_by(col(Video.id))))


def run_quick_profile(
    engine: Engine,
    paths: DataPaths,
    video_ids: Iterable[int] | None = None,
    redo: bool = False,
    tagger: LazyTagger | None = None,
) -> ProfileReport:
    """Profile videos one by one. Raises ``net.DownloadError`` if a model can't be fetched."""
    tagger = tagger or default_tagger(paths)
    report = ProfileReport()
    with Session(engine) as session:
        ids = [v.id for v in videos_to_profile(session, video_ids, redo) if v.id is not None]

    for video_id in ids:
        # One transaction per video: a broken file never undoes the others.
        try:
            with session_scope(engine) as session:
                profile_video(session, paths, session.get_one(Video, video_id), tagger)
        except media.MediaError as e:
            report.failed.append((video_id, str(e)))
        else:
            report.profiled.append(video_id)
    return report
