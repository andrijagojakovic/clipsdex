"""Synthetic test media, generated on the fly with ffmpeg (never real clips)."""

import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pytest

from clipsdex_engine import binaries
from clipsdex_engine.models import audioset

needs_ffmpeg = pytest.mark.skipif(
    binaries.resolve("ffmpeg").path is None or binaries.resolve("ffprobe").path is None,
    reason="ffmpeg/ffprobe not installed",
)


@dataclass(frozen=True)
class SyntheticMedia:
    with_audio: Path  # 2 s test pattern, 320x240, 440 Hz tone
    silent: Path  # 1 s blue frame, no audio stream
    not_video: Path  # random bytes with an .mp4 extension
    # 3 x 1.5 s: red, test pattern, dark gradient -> exactly 2 hard cuts. Shots must also
    # differ in brightness: ContentDetector averages hue/sat/lum change (threshold 27).
    three_shots: Path


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
        three_shots=folder / "three shots.mp4",
    )
    _ffmpeg(
        "-f", "lavfi", "-i", "color=c=red:size=320x568:rate=30:duration=1.5",
        "-f", "lavfi", "-i", "testsrc2=size=320x568:rate=30:duration=1.5",
        "-f", "lavfi", "-i", "gradients=size=320x568:rate=30:duration=1.5:c0=black:c1=navy",
        "-filter_complex", "[0:v][1:v][2:v]concat=n=3:v=1:a=0",
        "-pix_fmt", "yuv420p", str(media.three_shots),
    )  # fmt: skip
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


class FakeTagger:
    """Stands in for PANNs in unit tests: every non-silent segment gets the same probabilities."""

    def __init__(self, **class_probs: float) -> None:
        self.probs = np.zeros(audioset.NUM_CLASSES, dtype=np.float32)
        names = {
            "speech": audioset.SPEECH,
            "music": audioset.MUSIC,
            "singing": audioset.VOCAL_CLASSES[0],
        }
        for name, p in class_probs.items():
            self.probs[names[name]] = p
        self.calls = 0

    def predict(self, segments: np.ndarray) -> np.ndarray:
        self.calls += 1
        return np.tile(self.probs, (len(segments), 1))


class FakeReader:
    """Stands in for RapidOCR in unit tests: returns the same lines for every frame."""

    def __init__(self, lines: list[tuple[str, float]] | None = None) -> None:
        self.lines = lines or []
        self.frames: list[np.ndarray] = []

    def read(self, image: np.ndarray) -> list[tuple[str, float]]:
        self.frames.append(image)
        return list(self.lines)


def make_text_video(folder: Path, lines: list[str], seconds: float = 4.0) -> Path:
    """A still frame with big white text on dark grey, as a short H.264 video."""
    folder.mkdir(parents=True, exist_ok=True)
    image = np.full((1280, 720, 3), 40, np.uint8)
    for i, line in enumerate(lines):
        cv2.putText(
            image, line, (40, 400 + 120 * i), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (255, 255, 255), 5
        )
    png = folder / "frame.png"
    png.write_bytes(
        cv2.imencode(".png", image)[1].tobytes()
    )  # cv2.imwrite fails on non-ASCII paths
    video = folder / "text video.mp4"
    _ffmpeg(
        "-loop",
        "1",
        "-i",
        str(png),
        "-t",
        str(seconds),
        "-r",
        "25",
        "-pix_fmt",
        "yuv420p",
        str(video),
    )
    return video


@dataclass(frozen=True)
class SyntheticAudio:
    speech: Path | None  # ffmpeg's flite voice; None if this ffmpeg build lacks flite
    music: Path  # chords with a beat
    sine: Path  # a pure tone: PANNs calls it "Sine wave", not music
    silence: Path


MUSIC_EXPR = (
    "0.3*(sin(2*PI*220*t)+0.5*sin(4*PI*220*t)+sin(2*PI*277.18*t)+sin(2*PI*329.63*t))"
    "*(0.6+0.4*sin(2*PI*2*t))*if(lt(mod(t,2),1),1,0)"
    "+0.3*(sin(2*PI*196*t)+sin(2*PI*246.94*t)+sin(2*PI*293.66*t)+0.5*sin(4*PI*196*t))"
    "*(0.6+0.4*sin(2*PI*2*t))*if(lt(mod(t,2),1),0,1)"
    "+0.4*exp(-30*mod(t,0.5))*sin(2*PI*60*t)"
)
SPEECH_TEXT = (
    "Hello and welcome. Today I will explain how to price your services, how to talk to "
    "clients, and why you should always follow up after a meeting. Let us begin."
)


def make_synthetic_audio(folder: Path) -> SyntheticAudio:
    folder.mkdir(parents=True, exist_ok=True)
    wav = ("-ar", "16000", "-ac", "1")
    audio = SyntheticAudio(
        speech=folder / "speech.wav",
        music=folder / "music.wav",
        sine=folder / "sine.wav",
        silence=folder / "silence.wav",
    )
    _ffmpeg(
        "-f",
        "lavfi",
        "-i",
        f"aevalsrc='{MUSIC_EXPR}':s=16000:d=8",
        "-af",
        "volume=0.5",
        *wav,
        str(audio.music),
    )
    _ffmpeg("-f", "lavfi", "-i", "sine=frequency=440:duration=6", *wav, str(audio.sine))
    _ffmpeg("-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono", "-t", "6", *wav, str(audio.silence))
    try:
        _ffmpeg(
            "-f", "lavfi", "-i", f"flite=text='{SPEECH_TEXT}':voice=slt", *wav, str(audio.speech)
        )
    except subprocess.CalledProcessError:
        return SyntheticAudio(None, audio.music, audio.sine, audio.silence)
    return audio
