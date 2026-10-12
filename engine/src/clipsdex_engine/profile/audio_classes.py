"""Speech / music / silence ratios and a best-effort vocals flag (CLAUDE.md §5.1, step 2.2).

The audio is cut into 2 s segments (1 s was too short to recognize music reliably).
Quiet segments count as silence (energy check, no model); the rest go through the tagger.
A segment can contain both speech and music (e.g. talking over a song), so
``speech_ratio + music_ratio + silence_ratio`` may exceed 1.
"""

from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from clipsdex_engine.models import audioset

SEGMENT_S = 2.0
MIN_LAST_SEGMENT_S = 1.0  # a shorter tail is ignored
SILENCE_DBFS = -50.0
SPEECH_THRESHOLD = 0.3
MUSIC_THRESHOLD = 0.3
VOCALS_THRESHOLD = 0.2
VOCALS_MIN_SHARE = 0.25  # share of music segments with vocals to call the music "with vocals"


class AudioTagger(Protocol):
    def predict(self, segments: np.ndarray) -> np.ndarray:
        """(n, samples) float32 at 16 kHz -> (n, 527) AudioSet probabilities."""
        ...


@dataclass(frozen=True)
class AudioClasses:
    speech_ratio: float
    music_ratio: float
    silence_ratio: float
    has_vocals: bool | None  # None when there is (almost) no music
    segments: int


def read_wav(path: Path) -> np.ndarray:
    """16-bit mono WAV (our audio artifact) -> float32 samples in [-1, 1]."""
    with wave.open(str(path)) as w:
        if w.getsampwidth() != 2 or w.getnchannels() != 1:
            raise ValueError(f"Expected 16-bit mono WAV, got {path}")
        frames = w.readframes(w.getnframes())
    return np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0


def split_segments(samples: np.ndarray, sample_rate: int = audioset.SAMPLE_RATE) -> np.ndarray:
    seg = int(SEGMENT_S * sample_rate)
    count, tail = divmod(len(samples), seg)
    parts = [samples[: count * seg].reshape(count, seg)] if count else []
    if tail >= MIN_LAST_SEGMENT_S * sample_rate:
        parts.append(np.pad(samples[count * seg :], (0, seg - tail))[None, :])
    if not parts:
        return np.zeros((0, seg), dtype=np.float32)
    return np.concatenate(parts).astype(np.float32, copy=False)


def dbfs(segments: np.ndarray) -> np.ndarray:
    rms = np.sqrt(np.mean(np.square(segments, dtype=np.float64), axis=1))
    return 20.0 * np.log10(np.maximum(rms, 1e-10))


def summarize(probs: np.ndarray, silent: np.ndarray) -> AudioClasses:
    """``probs``: (non-silent segments, 527); ``silent``: bool mask over all segments."""
    total = len(silent)
    if total == 0:
        return AudioClasses(0.0, 0.0, 0.0, None, 0)

    speech = probs[:, audioset.SPEECH] >= SPEECH_THRESHOLD
    music = probs[:, audioset.MUSIC] >= MUSIC_THRESHOLD
    vocals = probs[:, list(audioset.VOCAL_CLASSES)].max(axis=1) >= VOCALS_THRESHOLD

    has_vocals: bool | None = None
    if music.sum() > 0:
        has_vocals = bool((vocals & music).sum() / music.sum() >= VOCALS_MIN_SHARE)

    return AudioClasses(
        speech_ratio=round(float(speech.sum()) / total, 3),
        music_ratio=round(float(music.sum()) / total, 3),
        silence_ratio=round(float(silent.sum()) / total, 3),
        has_vocals=has_vocals,
        segments=total,
    )


def classify(samples: np.ndarray, tagger: AudioTagger) -> AudioClasses:
    segments = split_segments(samples)
    silent = dbfs(segments) < SILENCE_DBFS
    probs = tagger.predict(segments[~silent])
    return summarize(probs, silent)
