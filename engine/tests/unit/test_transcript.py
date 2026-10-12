from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import pytest
from sqlmodel import Session
from typer.testing import CliRunner

from clipsdex_engine import config, db, net
from clipsdex_engine.artifacts import transcript
from clipsdex_engine.cli import app
from clipsdex_engine.config import DataPaths
from clipsdex_engine.db.models import JobStatus, Profile, Source, SourceType, Video
from clipsdex_engine.jobs import builtin, queue
from clipsdex_engine.jobs.handlers import Handler, WorkerEnv
from clipsdex_engine.jobs.worker import Worker
from clipsdex_engine.models import whisper
from clipsdex_engine.models.whisper import Segment, TranscriptInfo
from clipsdex_engine.sources import manual
from synthetic import SyntheticMedia, needs_ffmpeg


class FakeTranscriber:
    def __init__(self, segments: list[Segment], language: str = "en") -> None:
        self.segments = segments
        self.language = language
        self.calls: list[int] = []

    def transcribe(
        self, samples: np.ndarray, on_info: Callable[[TranscriptInfo], None] | None = None
    ) -> Iterator[Segment]:
        self.calls.append(len(samples))
        if on_info is not None:
            on_info(TranscriptInfo(self.language, 0.987654, len(samples) / 16_000))
        yield from self.segments


SEGMENTS = [Segment(0.0, 0.8, "Raise your prices."), Segment(0.9, 1.9, "Then follow up.")]


@pytest.fixture
def env(tmp_path: Path) -> WorkerEnv:
    paths = DataPaths(tmp_path / "Подаци")
    return WorkerEnv(db.open_database(paths), paths)


def handlers(fake: FakeTranscriber) -> dict[str, Handler]:
    return {"transcript": builtin.TranscriptHandler(lambda _env, _model: fake)}


@needs_ffmpeg
def test_worker_transcribes_and_stores_artifact(
    env: WorkerEnv, synthetic_media: SyntheticMedia
) -> None:
    manual.import_paths(env.engine, [synthetic_media.with_audio])
    fake = FakeTranscriber(SEGMENTS)
    [job_id] = builtin.enqueue_transcripts(env)

    report = Worker(env, handlers(fake)).run()

    assert report.done == [job_id]
    assert report.events[0] == "load whisper:small"  # Balanced preset
    assert fake.calls == [32_000]  # 2 s of 16 kHz audio, extracted on demand
    with Session(env.engine) as session:
        artifact = transcript.find(session, 1, "small")
        assert artifact is not None and artifact.params["vad"] is True
        data = artifact.data
    assert data["language"] == "en" and data["language_probability"] == 0.988
    assert data["text"] == "Raise your prices. Then follow up."
    assert data["segments"][1] == {"start": 0.9, "end": 1.9, "text": "Then follow up."}
    assert builtin.enqueue_transcripts(env) == []  # already transcribed
    assert len(builtin.enqueue_transcripts(env, redo=True)) == 1


def test_video_without_audio_gets_an_empty_transcript(env: WorkerEnv) -> None:
    with db.session_scope(env.engine) as session:
        source = Source(type=SourceType.MANUAL, display_name="Video files")
        session.add(source)
        session.flush()
        assert source.id is not None
        session.add(Video(id=1, source_id=source.id, file_path="C:/missing.mp4"))
        session.flush()
        session.add(Profile(video_id=1, has_audio=False))
    fake = FakeTranscriber(SEGMENTS)
    builtin.enqueue_transcripts(env)
    assert len(Worker(env, handlers(fake)).run().done) == 1
    assert fake.calls == []
    with Session(env.engine) as session:
        artifact = transcript.find(session, 1, "small")
        assert artifact is not None and artifact.data == transcript.empty_transcript()


def test_missing_file_and_no_audio_fails_clearly(env: WorkerEnv) -> None:
    with db.session_scope(env.engine) as session:
        source = Source(type=SourceType.MANUAL, display_name="Video files")
        session.add(source)
        session.flush()
        assert source.id is not None
        session.add(Video(id=1, source_id=source.id, file_path="C:/gone/missing.mp4"))
    builtin.enqueue_transcripts(env)
    report = Worker(env, handlers(FakeTranscriber(SEGMENTS))).run()
    assert "no local file and no extracted audio" in report.failed[0][1]


@needs_ffmpeg
def test_model_download_failure_is_retried(env: WorkerEnv, synthetic_media: SyntheticMedia) -> None:
    manual.import_paths(env.engine, [synthetic_media.with_audio])

    def offline(_env: WorkerEnv, _model: str) -> FakeTranscriber:
        raise net.DownloadError("Could not download model.bin. Check your internet connection.")

    [job_id] = builtin.enqueue_transcripts(env)
    report = Worker(env, {"transcript": builtin.TranscriptHandler(offline)}).run()
    assert len(report.retried) == queue.MAX_ATTEMPTS - 1
    assert report.failed[0][0] == job_id and "internet" in report.failed[0][1]
    assert queue.counts(env.engine)[JobStatus.FAILED] == 1


@needs_ffmpeg
def test_cli_transcribe_queue_and_show(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, synthetic_media: SyntheticMedia
) -> None:
    monkeypatch.setenv(config.DATA_DIR_ENV, str(tmp_path / "data"))
    fake = FakeTranscriber(SEGMENTS)
    monkeypatch.setitem(builtin.HANDLERS, "transcript", builtin.TranscriptHandler(lambda *_: fake))
    runner = CliRunner()
    runner.invoke(app, ["import", str(synthetic_media.with_audio)])

    assert "Queued 1 transcript job(s)" in runner.invoke(app, ["transcribe"]).output
    assert runner.invoke(app, ["worker"]).exit_code == 0
    shown = runner.invoke(app, ["show", "1"]).output
    assert "Transcript (small, language: en):" in shown
    assert "Raise your prices. Then follow up." in shown


def test_whisper_manifest_is_consistent() -> None:
    for size, spec in whisper.MODELS.items():
        names = [name for name, _, _ in spec.files]
        assert names == ["config.json", "model.bin", "tokenizer.json", "vocabulary.txt"], size
        model_bin = spec.files[1][2]
        assert model_bin.kind == "sha256" and len(model_bin.value) == 64
    with pytest.raises(whisper.UnknownModelError):
        whisper.ensure_model(Path("unused"), "gigantic")


def test_git_blob_checksum_matches_git() -> None:
    # `git hash-object` of b"hello\n" is ce013625030ba8dba906f756967f9e9ca394464a
    checksum = net.Checksum("git-sha1", "ce013625030ba8dba906f756967f9e9ca394464a", 6)
    digest = checksum.new_digest()
    digest.update(b"hello\n")
    assert digest.hexdigest() == checksum.value


@needs_ffmpeg
def test_no_speech_means_no_language(env: WorkerEnv, synthetic_media: SyntheticMedia) -> None:
    manual.import_paths(env.engine, [synthetic_media.with_audio])  # a tone, no words
    builtin.enqueue_transcripts(env)
    Worker(env, handlers(FakeTranscriber([Segment(0.0, 1.0, "")], language="en"))).run()
    with Session(env.engine) as session:
        artifact = transcript.find(session, 1, "small")
        assert artifact is not None
        assert artifact.data["language"] is None and artifact.data["text"] == ""
