"""Real faster-whisper on synthetic speech. uv run pytest -m integration

Uses the Balanced model ("small") if it was downloaded already (``clipsdex transcribe`` +
``clipsdex worker`` once); never downloads in tests.
"""

from pathlib import Path

import pytest

from clipsdex_engine.config import data_paths
from clipsdex_engine.models import whisper
from clipsdex_engine.profile import audio_classes
from synthetic import make_synthetic_audio, needs_ffmpeg

MODELS = data_paths().models
pytestmark = [
    pytest.mark.integration,
    needs_ffmpeg,
    pytest.mark.skipif(not whisper.is_downloaded(MODELS, "small"), reason="Whisper small missing"),
]


def test_transcribes_english_speech(tmp_path: Path) -> None:
    clips = make_synthetic_audio(tmp_path / "говор")
    if clips.speech is None:
        pytest.skip("this ffmpeg build has no flite voice")
    transcriber = whisper.Transcriber(whisper.model_dir(MODELS, "small"))
    infos: list[whisper.TranscriptInfo] = []
    segments = list(
        transcriber.transcribe(audio_classes.read_wav(clips.speech), on_info=infos.append)
    )
    text = " ".join(s.text for s in segments).lower()
    assert infos[0].language == "en"
    assert "price" in text and "clients" in text and "follow up" in text.replace("-", " ")


def test_music_without_words_gives_no_text(tmp_path: Path) -> None:
    clips = make_synthetic_audio(tmp_path / "музика")
    transcriber = whisper.Transcriber(whisper.model_dir(MODELS, "small"))
    segments = list(transcriber.transcribe(audio_classes.read_wav(clips.music)))
    assert " ".join(s.text for s in segments).strip() == ""  # VAD keeps hallucinations out
