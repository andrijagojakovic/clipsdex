"""SQLite job queue (CLAUDE.md §5.4, step 3.1).

Status flow: queued -> running -> done | failed | cancelled. A failed attempt that may
succeed later goes back to queued until ``MAX_ATTEMPTS``.

Cancelling is cooperative: a queued job is cancelled at once; a running job is marked
cancelled and the worker sees it via ``is_cancelled()`` and stops. Finishing calls only
change jobs that are still ``running``, so a cancel is never overwritten by a late "done".
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from sqlalchemy import Engine, func, update
from sqlmodel import Session, col, select

from clipsdex_engine.db import session_scope
from clipsdex_engine.db.models import Job, JobKind, JobStatus, utcnow

MAX_ATTEMPTS = 3
ACTIVE = (JobStatus.QUEUED, JobStatus.RUNNING)


@dataclass(frozen=True)
class JobSpec:
    kind: JobKind
    name: str  # artifact type ("transcript") or analyzer name ("song_id")
    video_id: int | None = None
    params: dict[str, Any] | None = None


def enqueue(session: Session, spec: JobSpec) -> Job:
    """Add a job, or return the identical queued/running one (no duplicate work)."""
    params = spec.params or {}
    existing = session.exec(
        select(Job).where(
            Job.kind == spec.kind,
            Job.name == spec.name,
            Job.video_id == spec.video_id,
            col(Job.status).in_(ACTIVE),
        )
    ).all()
    for job in existing:
        if job.params == params:
            return job
    job = Job(kind=spec.kind, name=spec.name, video_id=spec.video_id, params=params)
    session.add(job)
    session.flush()
    return job


def enqueue_many(engine: Engine, specs: Iterable[JobSpec]) -> list[int]:
    with session_scope(engine) as session:
        jobs = [enqueue(session, spec) for spec in specs]
        return [j.id for j in jobs if j.id is not None]


def claim_next(engine: Engine, names: Iterable[str] | None = None) -> Job | None:
    """Atomically move the oldest queued job (optionally only of ``names``) to running."""
    candidates = select(Job.id).where(Job.status == JobStatus.QUEUED)
    if names is not None:
        candidates = candidates.where(col(Job.name).in_(list(names)))
    candidates = candidates.order_by(col(Job.id)).limit(1).scalar_subquery()
    with session_scope(engine) as session:
        claimed = session.exec(
            update(Job)
            .where(col(Job.id) == candidates, col(Job.status) == JobStatus.QUEUED)
            .values(
                status=JobStatus.RUNNING,
                started_at=utcnow(),
                attempts=col(Job.attempts) + 1,
                progress=0.0,
                error=None,
            )
            .returning(col(Job.id))
        ).first()
        if claimed is None:
            return None
        job = session.get_one(Job, claimed[0])
        session.refresh(job)
        session.expunge(job)
        return job


def _finish(engine: Engine, job_id: int, **values: Any) -> bool:
    """Apply ``values`` only if the job is still running. Returns False if it was cancelled."""
    with session_scope(engine) as session:
        result = session.exec(
            update(Job)
            .where(col(Job.id) == job_id, col(Job.status) == JobStatus.RUNNING)
            .values(**values)
        )
        return result.rowcount == 1


def mark_done(engine: Engine, job_id: int) -> bool:
    return _finish(
        engine, job_id, status=JobStatus.DONE, progress=1.0, finished_at=utcnow(), error=None
    )


def mark_failed(engine: Engine, job_id: int, error: str, retry: bool = False) -> JobStatus | None:
    """Record a failure. With ``retry`` the job is queued again until ``MAX_ATTEMPTS``.

    Returns the new status, or None if the job was no longer running (e.g. cancelled).
    """
    with session_scope(engine) as session:
        job = session.get(Job, job_id)
        if job is None or job.status != JobStatus.RUNNING:
            return None
        if retry and job.attempts < MAX_ATTEMPTS:
            job.status = JobStatus.QUEUED
            job.started_at = None
        else:
            job.status = JobStatus.FAILED
            job.finished_at = utcnow()
        job.error = error
        return job.status


def set_progress(engine: Engine, job_id: int, progress: float) -> bool:
    return _finish(engine, job_id, progress=min(1.0, max(0.0, progress)))


def is_cancelled(engine: Engine, job_id: int) -> bool:
    with Session(engine) as session:
        status = session.exec(select(Job.status).where(Job.id == job_id)).first()
        return status == JobStatus.CANCELLED


def cancel(engine: Engine, job_ids: Iterable[int] | None = None) -> int:
    """Cancel the given jobs, or every queued/running job. Returns how many were cancelled."""
    query = update(Job).where(col(Job.status).in_(ACTIVE))
    if job_ids is not None:
        query = query.where(col(Job.id).in_(list(job_ids)))
    with session_scope(engine) as session:
        result = session.exec(query.values(status=JobStatus.CANCELLED, finished_at=utcnow()))
        return result.rowcount


def recover_interrupted(engine: Engine) -> int:
    """On startup: jobs left ``running`` by a crash or a closed app go back to the queue."""
    with session_scope(engine) as session:
        result = session.exec(
            update(Job)
            .where(col(Job.status) == JobStatus.RUNNING)
            .values(status=JobStatus.QUEUED, started_at=None, progress=0.0)
        )
        return result.rowcount


def counts(engine: Engine) -> dict[JobStatus, int]:
    with Session(engine) as session:
        rows = session.exec(select(Job.status, func.count()).group_by(col(Job.status))).all()
    found = {JobStatus(status): n for status, n in rows}
    return {status: found.get(status, 0) for status in JobStatus}


def recent(engine: Engine, limit: int = 20, status: JobStatus | None = None) -> list[Job]:
    query = select(Job).order_by(col(Job.id).desc()).limit(limit)
    if status is not None:
        query = query.where(Job.status == status)
    with Session(engine) as session:
        return list(session.exec(query))
