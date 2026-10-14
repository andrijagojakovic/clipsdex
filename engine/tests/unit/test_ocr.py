from pathlib import Path

import numpy as np
import pytest
from sqlmodel import Session
from typer.testing import CliRunner

from clipsdex_engine import artifacts, config, db
from clipsdex_engine.artifacts import ocr
from clipsdex_engine.cli import app
from clipsdex_engine.config import DataPaths
from clipsdex_engine.db.models import Job, JobStatus, Video
from clipsdex_engine.jobs import builtin
from clipsdex_engine.jobs.handlers import WorkerEnv
from clipsdex_engine.jobs.worker import Worker
from clipsdex_engine.sources import manual
from synthetic import SyntheticMedia, make_text_video, needs_ffmpeg


class ScriptedReader:
    """Returns the next scripted list of lines for each frame it reads."""

    def __init__(self, per_frame: list[list[tuple[str, float]]]) -> None:
        self.per_frame = list(per_frame)

    def read(self, image: np.ndarray) -> list[tuple[str, float]]:
        return self.per_frame.pop(0) if self.per_frame else []


def test_merge_keeps_most_confident_reading_and_first_time() -> None:
    merged = ocr.merge_lines(
        [
            (1.0, [("MIDNIGHT CITY -M83", 0.81), ("@edits.daily", 0.95)]),
            (3.0, [("MIDNIGHT CITY - M83", 0.97), ("@edits.daily", 0.94)]),
            (5.0, [("Midnight City — M83", 0.9), ("Part 2", 0.99)]),
        ]
    )
    assert [m.text for m in merged] == ["MIDNIGHT CITY - M83", "@edits.daily", "Part 2"]
    assert [m.seen for m in merged] == [3, 2, 1]
    assert [m.first_t for m in merged] == [1.0, 1.0, 5.0]


def test_numbers_keep_lookalike_lines_apart() -> None:
    # 96% similar, but the number changes the meaning (a sequel, "Part 2", "Top 5").
    merged = ocr.merge_lines([(0.0, [("PULP FICTION", 0.9), ("PULP FICTION 2", 0.9)])])
    assert [m.text for m in merged] == ["PULP FICTION", "PULP FICTION 2"]
    assert not ocr.same_text("Top 3 tips", "Top 5 tips")
    assert ocr.same_text("NOW PLAYING", "NOW PLAYlNG")  # an OCR slip is still the same line


def test_punctuation_only_is_ignored() -> None:
    assert ocr.merge_lines([(0.0, [("...", 0.99)])]) == []


@pytest.fixture
def env(tmp_path: Path) -> WorkerEnv:
    paths = DataPaths(tmp_path / "Подаци")
    return WorkerEnv(db.open_database(paths), paths)


def ocr_data(env: WorkerEnv, video_id: int = 1, budget: int = 4) -> dict[str, object]:
    with Session(env.engine) as session:
        artifact = artifacts.find(session, video_id, ocr.TYPE, ocr.params_for(budget))
        assert artifact is not None and isinstance(artifact.data, dict)
        return artifact.data


@needs_ffmpeg
def test_queue_selects_frames_first_then_reads_them(
    env: WorkerEnv, synthetic_media: SyntheticMedia
) -> None:
    manual.import_paths(env.engine, [synthetic_media.three_shots])
    reader = ScriptedReader(
        [
            [("NOW PLAYING", 0.98), ("blurry", 0.2)],
            [("NOW PLAYlNG", 0.88), ("x", 0.99)],  # an OCR slip and a one-char line
            [("Song: Midnight City", 0.95)],
        ]
    )
    handlers = {**builtin.HANDLERS, "ocr": builtin.OcrHandler(lambda: reader)}

    ids = builtin.enqueue_ocr(env)
    assert len(ids) == 2  # frames + ocr
    with Session(env.engine) as session:
        assert [session.get_one(Job, i).name for i in ids] == ["frames", "ocr"]

    report = Worker(env, handlers).run()

    assert len(report.done) == 2
    data = ocr_data(env)
    assert data["text"] == "NOW PLAYING\nSong: Midnight City"
    assert builtin.enqueue_ocr(env) == []


def test_nothing_to_read_without_frames_or_file(env: WorkerEnv, tmp_path: Path) -> None:
    gone = tmp_path / "gone.mp4"
    gone.write_bytes(b"x")
    manual.import_paths(env.engine, [gone])
    with db.session_scope(env.engine) as session:
        session.get_one(Video, 1).file_deleted = True
    assert builtin.enqueue_ocr(env) == []


@needs_ffmpeg
def test_frames_failure_skips_ocr(env: WorkerEnv, tmp_path: Path) -> None:
    gone = tmp_path / "gone.mp4"
    gone.write_bytes(b"x")
    manual.import_paths(env.engine, [gone])
    gone.unlink()
    ids = builtin.enqueue_ocr(env)
    Worker(env, builtin.HANDLERS).run()
    with Session(env.engine) as session:
        statuses = [session.get_one(Job, i).status for i in ids]
    assert statuses == [JobStatus.FAILED, JobStatus.FAILED]


@needs_ffmpeg
def test_real_rapidocr_end_to_end(env: WorkerEnv, tmp_path: Path) -> None:
    video = make_text_video(tmp_path / "текст", ["NOW PLAYING", "MIDNIGHT CITY"], seconds=4)
    manual.import_paths(env.engine, [video])
    builtin.enqueue_ocr(env)
    Worker(env, builtin.HANDLERS).run()
    text = str(ocr_data(env)["text"]).upper()
    assert "NOW PLAYING" in text and "MIDNIGHT CITY" in text


@needs_ffmpeg
def test_cli_ocr_and_show(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, synthetic_media: SyntheticMedia
) -> None:
    monkeypatch.setenv(config.DATA_DIR_ENV, str(tmp_path / "data"))
    reader = ScriptedReader([[("WAIT FOR IT", 0.99)]] * 3)
    monkeypatch.setitem(builtin.HANDLERS, "ocr", builtin.OcrHandler(lambda: reader))
    runner = CliRunner()
    runner.invoke(app, ["import", str(synthetic_media.three_shots)])
    assert "Queued 2 job(s)" in runner.invoke(app, ["ocr"]).output
    assert runner.invoke(app, ["worker"]).exit_code == 0
    shown = runner.invoke(app, ["show", "1"]).output
    assert "On-screen text (1 unique line(s)):" in shown
    assert "| WAIT FOR IT" in shown
