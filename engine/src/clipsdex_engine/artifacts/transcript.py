"""Transcript artifact: speech-to-text with timestamps and detected language.

Stored inline in ``artifacts.data``:
``{"language", "language_probability", "duration", "segments": [{start, end, text}], "text"}``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from sqlalchemy import Engine
from sqlmodel import Session

from clipsdex_engine import artifacts
from clipsdex_engine.artifacts import audio
from clipsdex_engine.config import DataPaths
from clipsdex_engine.db import session_scope
from clipsdex_engine.db.models import Artifact, Profile, Video
from clipsdex_engine.models.whisper import COMPUTE_TYPE, Segment, TranscriptInfo
from clipsdex_engine.profile import audio_classes

TYPE = "transcript"


class TranscriptSource(Protocol):
    def transcribe(
        self, samples: np.ndarray, on_info: Callable[[TranscriptInfo], None] | None = None
    ) -> Any:  # Iterator[Segment]
        ...


class NoAudioToTranscribeError(Exception):
    """Neither a local video file nor extracted audio is available."""


def params_for(model: str) -> dict[str, Any]:
    return {"model": model, "compute_type": COMPUTE_TYPE, "vad": True}


def find(session: Session, video_id: int, model: str) -> Artifact | None:
    return artifacts.find(session, video_id, TYPE, params_for(model))


def empty_transcript() -> dict[str, Any]:
    return {
        "language": None,
        "language_probability": None,
        "duration": 0.0,
        "segments": [],
        "text": "",
    }


def _audio_samples(engine: Engine, paths: DataPaths, video_id: int) -> np.ndarray | None:
    """The cached 16 kHz audio (extracted now if needed), or None if the video has no audio."""
    with session_scope(engine) as session:
        video = session.get_one(Video, video_id)
        profile = session.get(Profile, video_id)
        if profile is not None and profile.has_audio is False:
            return None
        cached = artifacts.find(session, video_id, audio.TYPE, audio.PARAMS)
        if cached is not None and audio.audio_path(paths, cached).is_file():
            wav = audio.audio_path(paths, cached)
        else:
            source = None if video.file_deleted or not video.file_path else Path(video.file_path)
            if source is None or not source.is_file():
                raise NoAudioToTranscribeError(
                    "This video has no local file and no extracted audio to transcribe."
                )
            wav = audio.audio_path(paths, audio.ensure_audio(session, paths, video, source))
    return audio_classes.read_wav(wav)


def build_transcript(
    engine: Engine,
    paths: DataPaths,
    video_id: int,
    model: str,
    transcriber: TranscriptSource,
    progress: Callable[[float], None] | None = None,
) -> dict[str, Any]:
    """Transcribe a video and save the artifact. The slow part runs outside any DB transaction."""
    samples = _audio_samples(engine, paths, video_id)
    data = empty_transcript()
    if samples is not None and len(samples) > 0:
        info_box: list[TranscriptInfo] = []
        segments: list[Segment] = []
        for segment in transcriber.transcribe(samples, on_info=info_box.append):
            segments.append(segment)
            if progress is not None and info_box and info_box[0].duration > 0:
                progress(min(0.99, segment.end / info_box[0].duration))
        info = info_box[0] if info_box else None
        # Whisper always "detects" a language; without any speech that guess means nothing.
        spoken = info if info is not None and any(s.text for s in segments) else None
        data = {
            "language": spoken.language if spoken else None,
            "language_probability": (
                round(spoken.language_probability, 3)
                if spoken and spoken.language_probability is not None
                else None
            ),
            "duration": round(info.duration, 3) if info else len(samples) / audio.SAMPLE_RATE,
            "segments": [asdict(s) for s in segments],
            "text": " ".join(s.text for s in segments if s.text),
        }

    params = params_for(model)
    with session_scope(engine) as session:
        artifact = find(session, video_id, model) or Artifact(
            video_id=video_id,
            type=TYPE,
            params_hash=artifacts.params_hash(params),
            params=params,
        )
        artifact.data = data
        session.add(artifact)
    return data
