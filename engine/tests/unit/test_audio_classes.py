import wave
from pathlib import Path

import numpy as np
import pytest

from clipsdex_engine.models import panns
from clipsdex_engine.profile import audio_classes as ac
from synthetic import FakeTagger

SR = panns.SAMPLE_RATE
SEG = int(ac.SEGMENT_S * SR)


def tone(seconds: float, amplitude: float = 0.5) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (amplitude * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def test_split_keeps_long_tail_and_drops_short_one() -> None:
    assert ac.split_segments(np.zeros(SEG * 3 + SEG // 2, np.float32)).shape == (4, SEG)
    assert ac.split_segments(np.zeros(SEG * 3 + SEG // 4, np.float32)).shape == (3, SEG)
    assert ac.split_segments(np.zeros(10, np.float32)).shape == (0, SEG)


def test_dbfs_separates_silence_from_sound() -> None:
    levels = ac.dbfs(np.stack([np.zeros(SEG, np.float32), tone(ac.SEGMENT_S)]))
    assert levels[0] < ac.SILENCE_DBFS < levels[1]


def test_silent_segments_skip_the_model() -> None:
    samples = np.concatenate([tone(4), np.zeros(SEG * 2, np.float32)])
    tagger = FakeTagger(speech=0.9)
    result = ac.classify(samples, tagger)
    assert (result.speech_ratio, result.silence_ratio, result.segments) == (0.5, 0.5, 4)


def test_speech_over_music_counts_for_both() -> None:
    result = ac.classify(tone(6), FakeTagger(speech=0.8, music=0.7))
    assert (result.speech_ratio, result.music_ratio) == (1.0, 1.0)
    assert result.has_vocals is False


def test_vocals_flag() -> None:
    assert ac.classify(tone(6), FakeTagger(music=0.8, singing=0.6)).has_vocals is True
    assert ac.classify(tone(6), FakeTagger(speech=0.9)).has_vocals is None  # no music


def test_below_thresholds_is_neither_speech_nor_music() -> None:
    result = ac.classify(tone(6), FakeTagger(speech=0.1, music=0.1))
    assert (result.speech_ratio, result.music_ratio, result.silence_ratio) == (0, 0, 0)


def test_empty_audio() -> None:
    result = ac.classify(np.zeros(0, np.float32), FakeTagger())
    assert result.segments == 0 and result.has_vocals is None


def test_read_wav_roundtrip(tmp_path: Path) -> None:
    samples = tone(1, amplitude=0.25)
    path = tmp_path / "a.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((samples * 32767).astype("<i2").tobytes())
    assert np.allclose(ac.read_wav(path), samples, atol=1e-3)


def test_read_wav_rejects_stereo(tmp_path: Path) -> None:
    path = tmp_path / "stereo.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(b"\x00\x00" * 200)
    with pytest.raises(ValueError, match="mono"):
        ac.read_wav(path)
