"""Thin wrappers around ffprobe/ffmpeg. All media tool calls go through here."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from clipsdex_engine import binaries

# The engine runs as a sidecar of a GUI app: never flash a console window on Windows.
_NO_WINDOW = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0


class MediaError(RuntimeError):
    """A media file could not be read or converted. The message is safe to show to users."""


@dataclass(frozen=True)
class MediaInfo:
    duration_s: float | None
    width: int | None
    height: int | None
    has_video: bool
    has_audio: bool


def _run(args: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            args,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
            creationflags=_NO_WINDOW,
        )
    except subprocess.TimeoutExpired as e:
        raise MediaError(f"{Path(args[0]).stem} took longer than {timeout:.0f} s.") from e
    except OSError as e:
        raise MediaError(f"Could not start {Path(args[0]).stem}: {e}") from e


def _last_line(text: str) -> str:
    lines = [line for line in text.strip().splitlines() if line.strip()]
    return lines[-1] if lines else "unknown error"


def _rotation(stream: dict[str, Any]) -> int:
    for side_data in stream.get("side_data_list", []):
        if "rotation" in side_data:
            return int(float(side_data["rotation"]))
    rotate = stream.get("tags", {}).get("rotate")
    return int(float(rotate)) if rotate else 0


def parse_probe(data: dict[str, Any]) -> MediaInfo:
    """Turn ffprobe's JSON (-show_format -show_streams) into MediaInfo."""
    streams: list[dict[str, Any]] = data.get("streams", [])
    # Cover art in audio files shows up as a video stream with attached_pic; ignore it.
    video = next(
        (
            s
            for s in streams
            if s.get("codec_type") == "video" and not s.get("disposition", {}).get("attached_pic")
        ),
        None,
    )
    has_audio = any(s.get("codec_type") == "audio" for s in streams)

    duration = data.get("format", {}).get("duration")
    if duration is None and video is not None:
        duration = video.get("duration")

    width = height = None
    if video is not None:
        width, height = video.get("width"), video.get("height")
        if width and height and abs(_rotation(video)) % 180 == 90:
            width, height = height, width  # phone videos are often stored rotated

    return MediaInfo(
        duration_s=round(float(duration), 3) if duration not in (None, "N/A") else None,
        width=int(width) if width else None,
        height=int(height) if height else None,
        has_video=video is not None,
        has_audio=has_audio,
    )


def probe(path: Path, timeout: float = 60.0) -> MediaInfo:
    ffprobe = binaries.require("ffprobe")
    result = _run(
        [
            str(ffprobe),
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ],
        timeout,
    )
    if result.returncode != 0:
        raise MediaError(f"Not a readable video file ({_last_line(result.stderr)}).")
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        raise MediaError("ffprobe returned unreadable output.") from e
    info = parse_probe(data)
    if not info.has_video and not info.has_audio:
        raise MediaError("The file has no video or audio.")
    return info


def extract_audio(src: Path, dst: Path, sample_rate: int = 16_000, timeout: float = 600.0) -> None:
    """Write the audio of ``src`` as mono 16-bit PCM WAV. Atomic: ``dst`` is complete or absent."""
    ffmpeg = binaries.require("ffmpeg")
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    result = _run(
        [
            str(ffmpeg),
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-i",
            str(src),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-c:a",
            "pcm_s16le",
            "-f",
            "wav",
            str(tmp),
        ],
        timeout,
    )
    if result.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise MediaError(f"Could not extract audio ({_last_line(result.stderr)}).")
    tmp.replace(dst)
