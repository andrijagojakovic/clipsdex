"""Real PANNs model on synthetic audio. Run with: uv run pytest -m integration

Needs ffmpeg and the downloaded checkpoint (run ``clipsdex profile`` once); never downloads.
"""

from pathlib import Path

import pytest

from clipsdex_engine.config import data_paths
from clipsdex_engine.models import panns
from clipsdex_engine.profile import audio_classes as ac
from synthetic import SyntheticAudio, make_synthetic_audio, needs_ffmpeg

CHECKPOINT = panns.checkpoint_path(data_paths().models)

pytestmark = [
    pytest.mark.integration,
    needs_ffmpeg,
    pytest.mark.skipif(not CHECKPOINT.is_file(), reason="PANNs checkpoint not downloaded"),
]


@pytest.fixture(scope="module")
def tagger() -> panns.PannsTagger:
    return panns.PannsTagger(CHECKPOINT)


@pytest.fixture(scope="module")
def clips(tmp_path_factory: pytest.TempPathFactory) -> SyntheticAudio:
    return make_synthetic_audio(tmp_path_factory.mktemp("audio") / "звук")


def classify(path: Path, tagger: panns.PannsTagger) -> ac.AudioClasses:
    return ac.classify(ac.read_wav(path), tagger)


def test_speech(clips: SyntheticAudio, tagger: panns.PannsTagger) -> None:
    if clips.speech is None:
        pytest.skip("this ffmpeg build has no flite voice")
    result = classify(clips.speech, tagger)
    assert result.speech_ratio >= 0.8
    assert result.music_ratio <= 0.2


def test_music(clips: SyntheticAudio, tagger: panns.PannsTagger) -> None:
    result = classify(clips.music, tagger)
    assert result.music_ratio >= 0.75
    assert result.speech_ratio <= 0.2
    assert result.has_vocals is False


def test_pure_tone_is_not_music_or_speech(clips: SyntheticAudio, tagger: panns.PannsTagger) -> None:
    result = classify(clips.sine, tagger)
    assert result.music_ratio <= 0.2 and result.speech_ratio <= 0.2


def test_silence(clips: SyntheticAudio, tagger: panns.PannsTagger) -> None:
    result = classify(clips.silence, tagger)
    assert result.silence_ratio == 1.0
    assert result.has_vocals is None
