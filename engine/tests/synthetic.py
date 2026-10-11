"""Synthetic test media, generated on the fly with ffmpeg (never real clips; see CLAUDE.md §8)."""

import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

from clipsdex_engine import binaries

needs_ffmpeg = pytest.mark.skipif(
    binaries.resolve("ffmpeg").path is None or binaries.resolve("ffprobe").path is None,
    reason="ffmpeg/ffprobe not installed",
)


@dataclass(frozen=True)
class SyntheticMedia:
    with_audio: Path  # 2 s test pattern, 320x240, 440 Hz tone
    silent: Path  # 1 s blue frame, no audio stream
    not_video: Path  # random bytes with an .mp4 extension


def _ffmpeg(*args: str) -> None:
    ffmpeg = binaries.require("ffmpeg")
    subprocess.run([str(ffmpeg), "-nostdin", "-v", "error", "-y", *args], check=True)


def make_synthetic_media(tmp_path_factory: pytest.TempPathFactory) -> SyntheticMedia:
    if binaries.resolve("ffmpeg").path is None:
        pytest.skip("ffmpeg not installed")
    # Cyrillic and spaces on purpose: Windows usernames and folders can contain them.
    folder = tmp_path_factory.mktemp("media") / "тест медији"
    folder.mkdir()
    media = SyntheticMedia(
        with_audio=folder / "tone.mp4",
        silent=folder / "blue.mp4",
        not_video=folder / "broken.mp4",
    )
    _ffmpeg(
        "-f", "lavfi", "-i", "testsrc=duration=2:size=320x240:rate=25",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
        "-shortest", "-pix_fmt", "yuv420p", str(media.with_audio),
    )  # fmt: skip
    _ffmpeg(
        "-f", "lavfi", "-i", "color=c=blue:duration=1:size=160x120:rate=25",
        "-pix_fmt", "yuv420p", str(media.silent),
    )  # fmt: skip
    media.not_video.write_bytes(b"this is not a video" * 100)
    return media
