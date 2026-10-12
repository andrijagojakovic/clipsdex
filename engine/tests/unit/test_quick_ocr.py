from pathlib import Path

import pytest
from sqlmodel import Session
from typer.testing import CliRunner

from clipsdex_engine import config, db, media
from clipsdex_engine import profile as quick_profile
from clipsdex_engine.cli import app
from clipsdex_engine.config import DataPaths
from clipsdex_engine.db.models import Profile
from clipsdex_engine.profile import quick_ocr
from clipsdex_engine.sources import manual
from synthetic import FakeReader, SyntheticMedia, make_text_video, needs_ffmpeg


@pytest.mark.parametrize(
    ("duration", "expected"),
    [(None, [0.0]), (0.0, [0.0]), (1.5, [0.15]), (30.0, [1.0, 15.0])],
)
def test_sample_times(duration: float | None, expected: list[float]) -> None:
    assert quick_ocr.sample_times(duration) == pytest.approx(expected)


def test_clean_lines_filters_and_dedupes() -> None:
    lines = [
        ("Pulp  Fiction", 0.98),
        ("PULP FICTION", 0.95),  # same text, different case
        ("1994", 0.9),
        ("~", 0.99),  # no real content
        ("blurry", 0.3),  # not confident
    ]
    assert quick_ocr.clean_lines(lines) == ["Pulp Fiction", "1994"]


def test_no_video_stream_means_no_text() -> None:
    info = media.MediaInfo(duration_s=3, width=None, height=None, has_video=False, has_audio=True)
    reader = FakeReader([("anything", 1.0)])
    assert quick_ocr.quick_ocr(Path("x.m4a"), info, reader) == quick_ocr.QuickOcr(False, None)
    assert reader.frames == []


@needs_ffmpeg
def test_quick_ocr_reads_two_small_frames(synthetic_media: SyntheticMedia) -> None:
    info = media.probe(synthetic_media.three_shots)
    reader = FakeReader([("Song: Midnight City", 0.97)])
    result = quick_ocr.quick_ocr(synthetic_media.three_shots, info, reader)
    assert result == quick_ocr.QuickOcr(True, "Song: Midnight City")
    assert len(reader.frames) == 2
    assert reader.frames[0].shape[1] == min(quick_ocr.OCR_WIDTH, 320)


@needs_ffmpeg
def test_profile_stores_text(tmp_path: Path, synthetic_media: SyntheticMedia) -> None:
    paths = DataPaths(tmp_path / "data")
    engine = db.open_database(paths)
    manual.import_paths(engine, [synthetic_media.silent])
    models = quick_profile.ProfileModels(
        tagger=quick_profile.Lazy(lambda: pytest.fail("no audio: tagger must not load")),
        ocr=quick_profile.Lazy(lambda: FakeReader([("@some.account", 0.9)])),
    )
    quick_profile.run_quick_profile(engine, paths, models=models)
    with Session(engine) as session:
        profile = session.get_one(Profile, 1)
        assert profile.has_text is True
        assert profile.ocr_text == "@some.account"


@needs_ffmpeg
def test_real_rapidocr_on_synthetic_text(tmp_path: Path) -> None:
    """RapidOCR's models ship inside the package: no network, so this is a unit test."""
    video = make_text_video(tmp_path / "текст", ["PULP FICTION", "1994"])
    info = media.probe(video)
    result = quick_ocr.quick_ocr(video, info, quick_ocr.RapidOcrReader())
    assert result.has_text
    assert result.text is not None
    assert "PULP FICTION" in result.text.upper()
    assert "1994" in result.text


@needs_ffmpeg
def test_real_rapidocr_on_frame_without_text(synthetic_media: SyntheticMedia) -> None:
    info = media.probe(synthetic_media.silent)
    result = quick_ocr.quick_ocr(synthetic_media.silent, info, quick_ocr.RapidOcrReader())
    assert result == quick_ocr.QuickOcr(False, None)


@needs_ffmpeg
def test_cli_show_prints_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, synthetic_media: SyntheticMedia
) -> None:
    monkeypatch.setenv(config.DATA_DIR_ENV, str(tmp_path / "data"))
    monkeypatch.setattr(
        quick_profile,
        "default_models",
        lambda *_a, **_k: quick_profile.ProfileModels(
            tagger=quick_profile.Lazy(lambda: pytest.fail("no audio")),
            ocr=quick_profile.Lazy(lambda: FakeReader([("WAIT FOR IT", 0.99)])),
        ),
    )
    runner = CliRunner()
    runner.invoke(app, ["import", str(synthetic_media.silent)])
    assert runner.invoke(app, ["profile"]).exit_code == 0
    shown = runner.invoke(app, ["show", "1"]).output
    assert "text on screen: yes" in shown
    assert "| WAIT FOR IT" in shown
