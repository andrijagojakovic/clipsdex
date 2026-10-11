"""Audio artifact: the video's audio as 16 kHz mono WAV, extracted once and reused."""

from __future__ import annotations

from pathlib import Path

from sqlmodel import Session

from clipsdex_engine import artifacts, media
from clipsdex_engine.config import DataPaths
from clipsdex_engine.db.models import Artifact, Video

TYPE = "audio"
SAMPLE_RATE = 16_000
PARAMS = {"sample_rate": SAMPLE_RATE, "channels": 1, "codec": "pcm_s16le"}


def audio_path(paths: DataPaths, artifact: Artifact) -> Path:
    assert artifact.path is not None
    return artifacts.from_stored_path(paths, artifact.path)


def ensure_audio(session: Session, paths: DataPaths, video: Video, source: Path) -> Artifact:
    """Return the cached audio artifact, extracting it from ``source`` if needed.

    The caller checks that the video has audio (see the profile's ``has_audio``).
    """
    assert video.id is not None
    existing = artifacts.find(session, video.id, TYPE, PARAMS)
    if existing is not None and audio_path(paths, existing).is_file():
        return existing

    digest = artifacts.params_hash(PARAMS)
    dst = artifacts.artifact_dir(paths, video.id) / f"{TYPE}-{digest[:8]}.wav"
    media.extract_audio(source, dst, sample_rate=SAMPLE_RATE)

    artifact = existing or Artifact(video_id=video.id, type=TYPE, params_hash=digest, params=PARAMS)
    artifact.path = artifacts.to_stored_path(paths, dst)
    session.add(artifact)
    session.flush()
    return artifact
