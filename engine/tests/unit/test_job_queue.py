import threading
from pathlib import Path

import pytest
from sqlalchemy import Engine
from sqlmodel import Session
from typer.testing import CliRunner

from clipsdex_engine import config, db
from clipsdex_engine.cli import app
from clipsdex_engine.config import DataPaths
from clipsdex_engine.db.models import Job, JobKind, JobStatus, Source, SourceType, Video
from clipsdex_engine.jobs import queue
from clipsdex_engine.jobs.queue import JobSpec

TRANSCRIPT = JobSpec(JobKind.ARTIFACT, "transcript", video_id=1, params={"model": "small"})


@pytest.fixture
def paths(tmp_path: Path) -> DataPaths:
    return DataPaths(tmp_path / "data")


@pytest.fixture
def engine(paths: DataPaths) -> Engine:
    engine = db.open_database(paths)
    with db.session_scope(engine) as session:
        source = Source(type=SourceType.MANUAL, display_name="Video files")
        session.add(source)
        session.flush()
        assert source.id is not None
        for i in (1, 2, 3):
            session.add(Video(id=i, source_id=source.id, file_path=f"C:/v/{i}.mp4"))
    return engine


def status(engine: Engine, job_id: int) -> JobStatus:
    with Session(engine) as session:
        return session.get_one(Job, job_id).status


def test_enqueue_skips_identical_active_jobs(engine: Engine) -> None:
    first = queue.enqueue_many(engine, [TRANSCRIPT, TRANSCRIPT])
    assert len(set(first)) == 1
    other_params = JobSpec(JobKind.ARTIFACT, "transcript", video_id=1, params={"model": "base"})
    assert queue.enqueue_many(engine, [other_params]) != first


def test_finished_job_can_be_enqueued_again(engine: Engine) -> None:
    [job_id] = queue.enqueue_many(engine, [TRANSCRIPT])
    claimed = queue.claim_next(engine)
    assert claimed is not None and claimed.id == job_id
    queue.mark_done(engine, job_id)
    assert queue.enqueue_many(engine, [TRANSCRIPT]) != [job_id]


def test_claim_is_fifo_and_filters_by_name(engine: Engine) -> None:
    ids = queue.enqueue_many(
        engine,
        [
            JobSpec(JobKind.ARTIFACT, "transcript", 1),
            JobSpec(JobKind.ARTIFACT, "frames", 1),
            JobSpec(JobKind.ARTIFACT, "transcript", 2),
        ],
    )
    frames = queue.claim_next(engine, names=["frames"])
    assert frames is not None and frames.id == ids[1]
    first = queue.claim_next(engine)
    assert first is not None and first.id == ids[0]
    assert first.status is JobStatus.RUNNING and first.attempts == 1
    assert first.started_at is not None
    second = queue.claim_next(engine)
    assert second is not None and second.id == ids[2]
    assert queue.claim_next(engine) is None


def test_done(engine: Engine) -> None:
    [job_id] = queue.enqueue_many(engine, [TRANSCRIPT])
    queue.claim_next(engine)
    queue.set_progress(engine, job_id, 0.5)
    assert queue.mark_done(engine, job_id)
    with Session(engine) as session:
        job = session.get_one(Job, job_id)
        assert job.status is JobStatus.DONE and job.progress == 1.0 and job.finished_at


def test_retry_until_max_attempts(engine: Engine) -> None:
    [job_id] = queue.enqueue_many(engine, [TRANSCRIPT])
    for attempt in range(1, queue.MAX_ATTEMPTS + 1):
        claimed = queue.claim_next(engine)
        assert claimed is not None and claimed.attempts == attempt
        new_status = queue.mark_failed(engine, job_id, "network hiccup", retry=True)
        expected = JobStatus.QUEUED if attempt < queue.MAX_ATTEMPTS else JobStatus.FAILED
        assert new_status is expected
    with Session(engine) as session:
        assert session.get_one(Job, job_id).error == "network hiccup"


def test_permanent_failure(engine: Engine) -> None:
    [job_id] = queue.enqueue_many(engine, [TRANSCRIPT])
    queue.claim_next(engine)
    assert queue.mark_failed(engine, job_id, "not a video") is JobStatus.FAILED


def test_cancel_queued_and_running(engine: Engine) -> None:
    ids = queue.enqueue_many(
        engine, [JobSpec(JobKind.ARTIFACT, "transcript", i) for i in (1, 2, 3)]
    )
    running = queue.claim_next(engine)
    assert running is not None and running.id == ids[0]

    assert queue.cancel(engine, [ids[0], ids[1]]) == 2
    assert queue.is_cancelled(engine, ids[0])
    # The worker finishing late must not overwrite the cancel.
    assert queue.mark_done(engine, ids[0]) is False
    assert queue.mark_failed(engine, ids[0], "boom") is None
    assert not queue.set_progress(engine, ids[0], 0.9)
    assert status(engine, ids[0]) is JobStatus.CANCELLED

    assert queue.cancel(engine) == 1  # only job 3 was still active
    assert queue.claim_next(engine) is None


def test_recover_interrupted_jobs(engine: Engine, paths: DataPaths) -> None:
    ids = queue.enqueue_many(engine, [JobSpec(JobKind.ARTIFACT, "transcript", i) for i in (1, 2)])
    queue.claim_next(engine)
    queue.claim_next(engine)
    queue.mark_done(engine, ids[1])

    restarted = db.open_database(paths)  # e.g. the app was closed mid-job
    assert queue.recover_interrupted(restarted) == 1
    assert status(restarted, ids[0]) is JobStatus.QUEUED
    again = queue.claim_next(restarted)
    assert again is not None and again.id == ids[0] and again.attempts == 2


def test_counts_and_recent(engine: Engine) -> None:
    ids = queue.enqueue_many(engine, [JobSpec(JobKind.ARTIFACT, "frames", i) for i in (1, 2, 3)])
    queue.claim_next(engine)
    queue.mark_done(engine, ids[0])
    queue.cancel(engine, [ids[1]])
    counts = queue.counts(engine)
    assert counts[JobStatus.DONE] == 1 and counts[JobStatus.CANCELLED] == 1
    assert counts[JobStatus.QUEUED] == 1 and counts[JobStatus.FAILED] == 0
    assert [j.id for j in queue.recent(engine, limit=2)] == [ids[2], ids[1]]
    assert [j.id for j in queue.recent(engine, status=JobStatus.DONE)] == [ids[0]]


def test_deleting_a_video_removes_its_jobs(engine: Engine) -> None:
    queue.enqueue_many(engine, [JobSpec(JobKind.ARTIFACT, "frames", 1)])
    with db.session_scope(engine) as session:
        session.delete(session.get_one(Video, 1))
    assert sum(queue.counts(engine).values()) == 0


def test_parallel_claims_never_share_a_job(engine: Engine, paths: DataPaths) -> None:
    n_jobs, n_workers = 40, 4
    queue.enqueue_many(
        engine,
        [JobSpec(JobKind.ARTIFACT, "frames", 1, params={"i": i}) for i in range(n_jobs)],
    )
    claimed: list[int] = []
    lock = threading.Lock()

    def worker() -> None:
        own_engine = db.create_db_engine(paths.db)  # like a separate worker process
        while (job := queue.claim_next(own_engine)) is not None:
            assert job.id is not None
            with lock:
                claimed.append(job.id)
            queue.mark_done(own_engine, job.id)

    threads = [threading.Thread(target=worker) for _ in range(n_workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(claimed) == n_jobs == len(set(claimed))
    assert queue.counts(engine)[JobStatus.DONE] == n_jobs


def test_cli_jobs_overview_and_cancel(
    paths: DataPaths, engine: Engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(config.DATA_DIR_ENV, str(paths.root))
    ids = queue.enqueue_many(engine, [JobSpec(JobKind.ARTIFACT, "frames", i) for i in (1, 2)])
    runner = CliRunner()

    overview = runner.invoke(app, ["jobs"])
    assert overview.exit_code == 0, overview.output
    assert "queued: 2" in overview.output and "artifact:frames" in overview.output

    assert runner.invoke(app, ["jobs", "cancel"]).exit_code == 1  # needs IDs or --all
    assert "Cancelled 1 job(s)." in runner.invoke(app, ["jobs", "cancel", str(ids[0])]).output
    assert "Cancelled 1 job(s)." in runner.invoke(app, ["jobs", "cancel", "--all"]).output
    assert "cancelled: 2" in runner.invoke(app, ["jobs", "--status", "cancelled"]).output
