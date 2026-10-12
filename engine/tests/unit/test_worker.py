from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from filelock import FileLock
from sqlalchemy import Engine
from sqlmodel import Session

from clipsdex_engine import db
from clipsdex_engine.config import DataPaths
from clipsdex_engine.db.models import Job, JobKind, JobStatus, Source, SourceType, Video
from clipsdex_engine.jobs import queue
from clipsdex_engine.jobs.handlers import (
    JobContext,
    PermanentJobError,
    Resource,
    RetryableJobError,
    WorkerEnv,
    spec_for,
)
from clipsdex_engine.jobs.worker import Worker, WorkerBusyError
from mocks.fake_llm import FakeLLM


@dataclass
class RecordingResource(Resource):
    key: str
    log: list[str]

    def __post_init__(self) -> None:
        super().__init__(self.key)

    def load(self) -> None:
        self.log.append(f"load {self.key}")

    def unload(self) -> None:
        self.log.append(f"unload {self.key}")


@dataclass
class FakeHandler:
    """Resource comes from params["model"]; behavior is scripted per video id."""

    name: str
    kind: JobKind = JobKind.ARTIFACT
    log: list[str] = field(default_factory=list)
    behavior: dict[int, Callable[[JobContext], None]] = field(default_factory=dict)

    def resource_key(self, video_id: int | None, params: dict[str, Any]) -> str:
        return params.get("model", "cpu")

    def make_resource(self, env: WorkerEnv, key: str) -> Resource:
        return RecordingResource(key, self.log)

    def run(self, ctx: JobContext) -> None:
        self.log.append(f"{self.name} v{ctx.job.video_id}")
        action = self.behavior.get(ctx.job.video_id or 0)
        if action is not None:
            action(ctx)


@pytest.fixture
def paths(tmp_path: Path) -> DataPaths:
    return DataPaths(tmp_path / "data")


@pytest.fixture
def env(paths: DataPaths) -> WorkerEnv:
    engine = db.open_database(paths)
    with db.session_scope(engine) as session:
        source = Source(type=SourceType.MANUAL, display_name="Video files")
        session.add(source)
        session.flush()
        assert source.id is not None
        for i in range(1, 7):
            session.add(Video(id=i, source_id=source.id, file_path=f"C:/v/{i}.mp4"))
    return WorkerEnv(engine, paths, provider_factory=FakeLLM)


def status(engine: Engine, job_id: int) -> JobStatus:
    with Session(engine) as session:
        return session.get_one(Job, job_id).status


def test_groups_by_resource_and_loads_one_model_at_a_time(env: WorkerEnv) -> None:
    log: list[str] = []
    transcript = FakeHandler("transcript", log=log)
    describe = FakeHandler("describe", log=log)
    # Interleaved on purpose: whisper, vision, whisper, vision.
    queue.enqueue_many(
        env.engine,
        [
            spec_for(transcript, 1, {"model": "whisper"}),
            spec_for(describe, 1, {"model": "vision"}),
            spec_for(transcript, 2, {"model": "whisper"}),
            spec_for(describe, 2, {"model": "vision"}),
        ],
    )
    report = Worker(env, {"transcript": transcript, "describe": describe}).run()

    assert len(report.done) == 4
    assert log == [
        "load whisper",
        "transcript v1",
        "transcript v2",
        "unload whisper",
        "load vision",
        "describe v1",
        "describe v2",
        "unload vision",
    ]


def test_dependencies_run_first_even_across_resources(env: WorkerEnv) -> None:
    log: list[str] = []
    transcript = FakeHandler("transcript", log=log)
    brief = FakeHandler("brief", kind=JobKind.ANALYZER, log=log)
    with db.session_scope(env.engine) as session:
        t1 = queue.enqueue(session, spec_for(transcript, 1, {"model": "whisper"}))
        assert t1.id is not None
        # Enqueued before transcript 2, but must wait for transcript 1 only.
        queue.enqueue(session, spec_for(brief, 1, {"model": "llm"}, depends_on=(t1.id,)))
        queue.enqueue(session, spec_for(transcript, 2, {"model": "whisper"}))
    Worker(env, {"transcript": transcript, "brief": brief}).run()
    assert [e for e in log if not e.startswith(("load", "unload"))] == [
        "transcript v1",
        "transcript v2",
        "brief v1",
    ]


def test_failed_dependency_skips_the_whole_chain(env: WorkerEnv) -> None:
    def broken(_ctx: JobContext) -> None:
        raise PermanentJobError("not a video")

    transcript = FakeHandler("transcript", behavior={1: broken})
    brief = FakeHandler("brief")
    export = FakeHandler("export")
    with db.session_scope(env.engine) as session:
        t = queue.enqueue(session, spec_for(transcript, 1))
        assert t.id is not None
        b = queue.enqueue(session, spec_for(brief, 1, depends_on=(t.id,)))
        assert b.id is not None
        e = queue.enqueue(session, spec_for(export, 1, depends_on=(b.id,)))
        ids = (t.id, b.id, e.id)

    report = Worker(env, {h.name: h for h in (transcript, brief, export)}).run()

    assert [s for s in (status(env.engine, i) for i in ids if i)] == [JobStatus.FAILED] * 3
    with Session(env.engine) as session:
        error = session.get_one(Job, ids[1]).error
        assert error is not None and "transcript" in error and "failed" in error
    assert brief.log == [] and export.log == []
    assert report.failed[0] == (ids[0], "not a video")


def test_retryable_errors_retry_then_fail(env: WorkerEnv) -> None:
    def flaky(_ctx: JobContext) -> None:
        raise RetryableJobError("server busy")

    handler = FakeHandler("fetch", behavior={1: flaky})
    [job_id] = queue.enqueue_many(env.engine, [spec_for(handler, 1)])
    report = Worker(env, {"fetch": handler}).run()
    assert handler.log.count("fetch v1") == queue.MAX_ATTEMPTS
    assert len(report.retried) == queue.MAX_ATTEMPTS - 1
    assert report.failed == [(job_id, "server busy")]


def test_unexpected_crash_fails_only_that_job(env: WorkerEnv) -> None:
    def crash(_ctx: JobContext) -> None:
        raise ZeroDivisionError("oops")

    handler = FakeHandler("calc", behavior={1: crash})
    ids = queue.enqueue_many(env.engine, [spec_for(handler, 1), spec_for(handler, 2)])
    report = Worker(env, {"calc": handler}).run()
    assert report.done == [ids[1]]
    assert "Unexpected error: oops" in report.failed[0][1]


def test_cancel_while_running_stops_at_next_checkpoint(env: WorkerEnv) -> None:
    def long_job(ctx: JobContext) -> None:
        ctx.progress(0.3)
        queue.cancel(env.engine, [ctx.job_id])  # the user presses Cancel meanwhile
        ctx.progress(0.6)  # raises JobCancelled
        raise AssertionError("should not get here")

    handler = FakeHandler("long", behavior={1: long_job})
    [job_id] = queue.enqueue_many(env.engine, [spec_for(handler, 1)])
    report = Worker(env, {"long": handler}).run()
    assert report.cancelled == [job_id]
    assert status(env.engine, job_id) is JobStatus.CANCELLED


def test_unknown_job_type_fails_clearly(env: WorkerEnv) -> None:
    [job_id] = queue.enqueue_many(env.engine, [queue.JobSpec(JobKind.ARTIFACT, "teleport", 1)])
    Worker(env, {}).run()
    with Session(env.engine) as session:
        job = session.get_one(Job, job_id)
        assert job.status is JobStatus.FAILED and job.error is not None
        assert "teleport" in job.error


def test_only_one_worker_per_data_folder(env: WorkerEnv, paths: DataPaths) -> None:
    lock = FileLock(paths.root / "worker.lock")
    lock.acquire()
    try:
        with pytest.raises(WorkerBusyError):
            Worker(env, {}).run()
    finally:
        lock.release()
    Worker(env, {}).run()  # free again


def test_resumes_jobs_interrupted_by_a_crash(env: WorkerEnv) -> None:
    handler = FakeHandler("frames")
    [job_id] = queue.enqueue_many(env.engine, [spec_for(handler, 1)])
    queue.claim_next(env.engine)  # "running" when the app was killed
    report = Worker(env, {"frames": handler}).run()
    assert report.done == [job_id] and "resumed 1 interrupted job(s)" in report.events


def test_ctrl_c_puts_the_running_job_back(env: WorkerEnv) -> None:
    def interrupted(_ctx: JobContext) -> None:
        raise KeyboardInterrupt

    log: list[str] = []
    handler = FakeHandler("frames", log=log, behavior={1: interrupted})
    [job_id] = queue.enqueue_many(env.engine, [spec_for(handler, 1, {"model": "m"})])
    with pytest.raises(KeyboardInterrupt):
        Worker(env, {"frames": handler}).run()
    assert status(env.engine, job_id) is JobStatus.QUEUED
    assert log[-1] == "unload m"  # the model is still freed
