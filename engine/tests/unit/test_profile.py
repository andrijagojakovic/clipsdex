import wave
from pathlib import Path

import pytest
from sqlalchemy import Engine
from sqlmodel import Session, select
from typer.testing import CliRunner

from clipsdex_engine import config, db, media
from clipsdex_engine import profile as quick_profile
from clipsdex_engine.artifacts import audio
from clipsdex_engine.cli import app
from clipsdex_engine.config import DataPaths
from clipsdex_engine.db.models import Artifact, Profile, Source, SourceType, Video
from clipsdex_engine.sources import manual
from synthetic import FakeReader, FakeTagger, SyntheticMedia, needs_ffmpeg

pytestmark = needs_ffmpeg


@pytest.fixture
def paths(tmp_path: Path) -> DataPaths:
    return DataPaths(tmp_path / "Подаци" / "ClipsDex")


@pytest.fixture
def engine(paths: DataPaths) -> Engine:
    return db.open_database(paths)


def fake_models(*lines: tuple[str, float]) -> quick_profile.ProfileModels:
    """Music with singing everywhere and the given OCR lines; no real models, no download."""
    return quick_profile.ProfileModels(
        tagger=quick_profile.Lazy(lambda: FakeTagger(music=0.9, singing=0.6)),
        ocr=quick_profile.Lazy(lambda: FakeReader(list(lines))),
    )


def test_probe_reads_synthetic_video(synthetic_media: SyntheticMedia) -> None:
    info = media.probe(synthetic_media.with_audio)
    assert info.duration_s == pytest.approx(2.0, abs=0.1)
    assert (info.width, info.height) == (320, 240)
    assert info.has_video and info.has_audio

    silent = media.probe(synthetic_media.silent)
    assert silent.has_video and not silent.has_audio


def test_probe_rejects_non_video(synthetic_media: SyntheticMedia) -> None:
    with pytest.raises(media.MediaError, match="Not a readable video file"):
        media.probe(synthetic_media.not_video)


def test_profile_with_audio_creates_wav(
    engine: Engine, paths: DataPaths, synthetic_media: SyntheticMedia
) -> None:
    manual.import_paths(engine, [synthetic_media.with_audio])

    report = quick_profile.run_quick_profile(engine, paths, models=fake_models())

    assert report.profiled == [1] and report.failed == []
    with Session(engine) as session:
        profile = session.get_one(Profile, 1)
        assert profile.has_audio is True
        assert (profile.width, profile.height) == (320, 240)
        assert (profile.music_ratio, profile.speech_ratio, profile.silence_ratio) == (1, 0, 0)
        assert profile.has_vocals is True
        assert profile.profile_version == quick_profile.PROFILE_VERSION
        artifact = session.exec(select(Artifact)).one()
        assert artifact.type == "audio"
        assert artifact.path is not None and not Path(artifact.path).is_absolute()
        wav = audio.audio_path(paths, artifact)
    with wave.open(str(wav)) as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (16_000, 1, 2)
        assert w.getnframes() / w.getframerate() == pytest.approx(2.0, abs=0.1)
    assert not list(wav.parent.glob("*.part"))


def test_video_without_audio_has_no_audio_artifact(
    engine: Engine, paths: DataPaths, synthetic_media: SyntheticMedia
) -> None:
    manual.import_paths(engine, [synthetic_media.silent])
    quick_profile.run_quick_profile(engine, paths, models=fake_models())
    with Session(engine) as session:
        profile = session.get_one(Profile, 1)
        assert profile.has_audio is False
        assert profile.silence_ratio == 1.0 and profile.has_vocals is None
        assert session.exec(select(Artifact)).all() == []


def test_model_is_not_loaded_without_audio(
    engine: Engine, paths: DataPaths, synthetic_media: SyntheticMedia
) -> None:
    loads: list[int] = []

    def factory() -> FakeTagger:
        loads.append(1)
        return FakeTagger()

    manual.import_paths(engine, [synthetic_media.silent])
    quick_profile.run_quick_profile(
        engine,
        paths,
        models=quick_profile.ProfileModels(
            tagger=quick_profile.Lazy(factory), ocr=quick_profile.Lazy(FakeReader)
        ),
    )
    assert loads == []


def test_old_profile_versions_are_rebuilt(
    engine: Engine, paths: DataPaths, synthetic_media: SyntheticMedia
) -> None:
    manual.import_paths(engine, [synthetic_media.with_audio])
    quick_profile.run_quick_profile(engine, paths, models=fake_models())
    with db.session_scope(engine) as session:
        session.get_one(Profile, 1).profile_version = 1

    report = quick_profile.run_quick_profile(engine, paths, models=fake_models())

    assert report.profiled == [1]


def test_second_run_does_nothing_and_redo_reuses_audio(
    engine: Engine, paths: DataPaths, synthetic_media: SyntheticMedia
) -> None:
    manual.import_paths(engine, [synthetic_media.with_audio])
    quick_profile.run_quick_profile(engine, paths, models=fake_models())
    with Session(engine) as session:
        wav = audio.audio_path(paths, session.exec(select(Artifact)).one())
    first_mtime = wav.stat().st_mtime_ns

    assert quick_profile.run_quick_profile(engine, paths, models=fake_models()).profiled == []
    assert quick_profile.run_quick_profile(
        engine, paths, redo=True, models=fake_models()
    ).profiled == [1]
    assert wav.stat().st_mtime_ns == first_mtime  # audio was not extracted again
    with Session(engine) as session:
        assert len(session.exec(select(Artifact)).all()) == 1


def test_broken_and_missing_files_fail_without_stopping_others(
    engine: Engine, paths: DataPaths, synthetic_media: SyntheticMedia, tmp_path: Path
) -> None:
    gone = tmp_path / "gone.mp4"
    gone.write_bytes(b"temporary")
    manual.import_paths(engine, [synthetic_media.not_video, gone, synthetic_media.with_audio])
    gone.unlink()

    report = quick_profile.run_quick_profile(engine, paths, models=fake_models())

    assert report.profiled == [3]
    reasons = dict(report.failed)
    assert "Not a readable video file" in reasons[1]
    assert "no longer at" in reasons[2]


def test_link_only_videos_are_skipped(engine: Engine, paths: DataPaths) -> None:
    with db.session_scope(engine) as session:
        source = Source(type=SourceType.INSTAGRAM_EXPORT, display_name="Instagram export")
        session.add(source)
        session.flush()
        assert source.id is not None
        session.add(Video(source_id=source.id, original_url="https://www.instagram.com/reel/x/"))
    report = quick_profile.run_quick_profile(engine, paths, models=fake_models())
    assert report.profiled == [] and report.failed == []


def test_cli_profile_and_show(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, synthetic_media: SyntheticMedia
) -> None:
    monkeypatch.setenv(config.DATA_DIR_ENV, str(tmp_path / "data"))
    monkeypatch.setattr(quick_profile, "default_models", lambda *_a, **_k: fake_models())
    runner = CliRunner()
    runner.invoke(app, ["import", str(synthetic_media.with_audio), str(synthetic_media.not_video)])

    result = runner.invoke(app, ["profile"])
    assert result.exit_code == 1  # one broken file
    assert "Profiled 1 video(s), 1 failed." in result.output

    shown = runner.invoke(app, ["show", "1"])
    assert "size 320x240, audio yes" in shown.output
    assert "Cached artifacts: audio" in shown.output

    again = runner.invoke(app, ["profile", "--video", "1"])
    assert "Nothing to profile" in again.output
