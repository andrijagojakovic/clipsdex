"""Whisper speech-to-text via faster-whisper (CTranslate2, int8 on CPU).

Model files are the Systran CTranslate2 conversions on Hugging Face (MIT, like OpenAI's
Whisper weights). We download them ourselves, pinned to a revision and checksum-verified,
into ``<data dir>/models/whisper/<size>/`` instead of letting huggingface_hub fill a cache in
the user's home folder.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from clipsdex_engine import net
from clipsdex_engine.net import Checksum

COMPUTE_TYPE = "int8"

FileSpec = tuple[str, int, Checksum]  # (name, size in bytes, checksum)


def _git(name: str, size: int, blob: str) -> FileSpec:
    return (name, size, Checksum("git-sha1", blob, size))


def _lfs(name: str, size: int, sha256: str) -> FileSpec:
    return (name, size, Checksum("sha256", sha256))


_TOKENIZER = _git("tokenizer.json", 2_203_239, "7818adb6de9fa3064d3ff81226fdd675be1f6344")
_VOCAB = _git("vocabulary.txt", 459_861, "c9074644d9d1205686f16d411564729461324b75")


@dataclass(frozen=True)
class WhisperFiles:
    repo: str
    revision: str
    files: tuple[FileSpec, ...]

    @property
    def total_bytes(self) -> int:
        return sum(size for _, size, _ in self.files)


# Verified against the Hugging Face API on 2026-10-12.
MODELS: dict[str, WhisperFiles] = {
    "base": WhisperFiles(
        "Systran/faster-whisper-base",
        "ebe41f70d5b6dfa9166e2c581c45c9c0cfc57b66",
        (
            _git("config.json", 2_309, "867cf1a0fece1394e01d55e287ba2f09a577c046"),
            _lfs(
                "model.bin",
                145_217_532,
                "d01c3014881c9c6f3133c182f3d2887eb6ca1c789a7538c5c007196857a0a6a9",
            ),
            _TOKENIZER,
            _VOCAB,
        ),
    ),
    "small": WhisperFiles(
        "Systran/faster-whisper-small",
        "536b0662742c02347bc0e980a01041f333bce120",
        (
            _git("config.json", 2_370, "e5047537059bd8f182d9ca64c470201585015187"),
            _lfs(
                "model.bin",
                483_546_902,
                "3e305921506d8872816023e4c273e75d2419fb89b24da97b4fe7bce14170d671",
            ),
            _TOKENIZER,
            _VOCAB,
        ),
    ),
    "medium": WhisperFiles(
        "Systran/faster-whisper-medium",
        "08e178d48790749d25932bbc082711ddcfdfbc4f",
        (
            _git("config.json", 2_257, "242aa06a480a7b5509375c645097e87af5136774"),
            _lfs(
                "model.bin",
                1_527_906_378,
                "9b45e1009dcc4ab601eff815b61d80e60ce3fd8c74c1a14f4a282258286b51ae",
            ),
            _TOKENIZER,
            _VOCAB,
        ),
    ),
}


class UnknownModelError(ValueError):
    pass


def model_dir(models_dir: Path, size: str) -> Path:
    return models_dir / "whisper" / size


def is_downloaded(models_dir: Path, size: str) -> bool:
    spec = MODELS.get(size)
    if spec is None:
        return False
    folder = model_dir(models_dir, size)
    return all(
        (folder / name).is_file() and (folder / name).stat().st_size == n
        for name, n, _ in spec.files
    )


def ensure_model(models_dir: Path, size: str, progress: net.Progress | None = None) -> Path:
    """Download the model files that are missing; returns the model folder."""
    spec = MODELS.get(size)
    if spec is None:
        raise UnknownModelError(f"Unknown Whisper model {size!r}. Use one of: {', '.join(MODELS)}")
    folder = model_dir(models_dir, size)
    done_before = 0
    for name, n, checksum in spec.files:
        path = folder / name
        if not (path.is_file() and path.stat().st_size == n):
            url = f"https://huggingface.co/{spec.repo}/resolve/{spec.revision}/{name}"

            def file_progress(done: int, _total: int | None, base: int = done_before) -> None:
                if progress is not None:
                    progress(base + done, spec.total_bytes)

            net.download(url, path, checksum=checksum, progress=file_progress)
        done_before += n
    return folder


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class TranscriptInfo:
    language: str | None
    language_probability: float | None
    duration: float


class Transcriber:
    """Holds one loaded Whisper model. ``transcribe`` yields segments as they are decoded."""

    def __init__(self, folder: Path, cpu_threads: int = 0) -> None:
        from faster_whisper import WhisperModel  # heavy import, only when a model is used

        self._model: Any = WhisperModel(
            str(folder), device="cpu", compute_type=COMPUTE_TYPE, cpu_threads=cpu_threads
        )

    def transcribe(
        self, samples: np.ndarray, on_info: Callable[[TranscriptInfo], None] | None = None
    ) -> Iterator[Segment]:
        segments, info = self._model.transcribe(
            samples,
            beam_size=5,
            vad_filter=True,  # skips music/silence: fewer hallucinated "lyrics" and outros
            condition_on_previous_text=False,  # stops one hallucination from repeating
        )
        if on_info is not None:
            on_info(TranscriptInfo(info.language, info.language_probability, info.duration))
        for s in segments:
            yield Segment(float(s.start), float(s.end), s.text.strip())
