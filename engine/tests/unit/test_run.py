import json
from pathlib import Path

import pytest
from filelock import FileLock
from typer.testing import CliRunner

from clipsdex_engine import artifacts, config, db, llm
from clipsdex_engine.artifacts import current
from clipsdex_engine.cli import app
from clipsdex_engine.config import DataPaths
from clipsdex_engine.db.models import Artifact, JobStatus, Source, SourceType, Video
from clipsdex_engine.jobs import queue
from mocks.fake_llm import FakeLLM

ANSWER = json.dumps(
    {
        "category": "business",
        "confidence": 0.85,
        "tags": ["pricing"],
        "summary": "Pricing advice for freelancers.",
        "reason": "Talks about prices.",
    }
)


@pytest.fixture
def setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[DataPaths, FakeLLM]:
    """Video 1 has every input categorize reads, so only the analyzer itself runs."""
    paths = DataPaths(tmp_path / "data")
    engine = db.open_database(paths)
    with db.session_scope(engine) as session:
        source = Source(type=SourceType.MANUAL, display_name="Video files")
        session.add(source)
        session.flush()
        assert source.id is not None
        session.add(Video(id=1, source_id=source.id, file_path="C:/v/pricing.mp4"))
        session.flush()
        models = db.load_settings(session).models()
        for name, data in (
            ("transcript", {"language": "en", "text": "Raise your prices."}),
            ("ocr", {"text": ""}),
            ("frame_descriptions", {"frames": [], "skipped": []}),
        ):
            params = current.params(name, models)
            session.add(
                Artifact(
                    video_id=1,
                    type=name,
                    params_hash=artifacts.params_hash(params),
                    params=params,
                    data=data,
                )
            )
    fake = FakeLLM(default_reply=ANSWER)
    monkeypatch.setenv(config.DATA_DIR_ENV, str(paths.root))
    monkeypatch.setattr(llm, "default_provider", lambda *_a, **_k: fake)
    return paths, fake


def test_run_plans_asks_works_and_shows_results(setup: tuple[DataPaths, FakeLLM]) -> None:
    _paths, fake = setup
    runner = CliRunner()
    out = runner.invoke(app, ["run", "--analyzer", "categorize", "--video", "1"], input="y\n")
    assert out.exit_code == 0, out.output
    assert "Plan: categorize for the selected videos: 1 video(s)" in out.output
    assert "Start? [Y/n]" in out.output
    assert "Queued 1 job(s)" in out.output
    assert "#1    categorize (85%): business: Pricing advice for freelancers. [pricing]" in (
        out.output
    )
    assert len(fake.calls) == 1

    again = runner.invoke(app, ["run", "--analyzer", "categorize", "--video", "1", "-y"])
    assert "already done: categorize" in again.output
    assert "Everything needed is already computed." in again.output
    assert len(fake.calls) == 1


def test_saying_no_starts_nothing(setup: tuple[DataPaths, FakeLLM]) -> None:
    paths, fake = setup
    out = CliRunner().invoke(app, ["run", "--analyzer", "categorize", "--video", "1"], input="n\n")
    assert "Nothing was started." in out.output
    assert fake.calls == [] and queue.recent(db.open_database(paths)) == []


def test_no_wait_only_queues(setup: tuple[DataPaths, FakeLLM]) -> None:
    paths, fake = setup
    out = CliRunner().invoke(
        app, ["run", "--analyzer", "categorize", "--video", "1", "-y", "--no-wait"]
    )
    assert "Run them with: clipsdex worker" in out.output
    assert fake.calls == []
    assert [j.status for j in queue.recent(db.open_database(paths))] == [JobStatus.QUEUED]


def test_another_worker_does_the_jobs(setup: tuple[DataPaths, FakeLLM]) -> None:
    paths, fake = setup
    with FileLock(paths.root / "worker.lock"):
        out = CliRunner().invoke(app, ["run", "--analyzer", "categorize", "--video", "1", "-y"])
    assert out.exit_code == 0 and "Another worker is already running" in out.output
    assert fake.calls == []


def test_unknown_or_unavailable_analyzers_are_refused(setup: tuple[DataPaths, FakeLLM]) -> None:
    runner = CliRunner()
    missing = runner.invoke(app, ["run", "--analyzer", "song_id", "-y"])
    assert missing.exit_code == 1 and "Find songs (song_id)" in missing.output
    unknown = runner.invoke(app, ["run", "--analyzer", "teleport", "-y"])
    assert unknown.exit_code == 1 and "Unknown analyzer" in unknown.output
