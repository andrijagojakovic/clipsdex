"""Live progress for the desktop app (step 6.2), as Server-Sent Events.

Two sources feed the stream:

- Job changes (status, progress, error), read from the jobs table a few times per second.
  Reading the table, not only listening to our own worker, means the app also sees jobs
  run by another worker (e.g. ``clipsdex worker`` in a terminal).
- Worker messages ("load ollama:gemma3:4b", "run #12 transcript") from the worker in this
  process, pushed through ``EventHub`` the moment they happen.

Event types: ``snapshot`` (on connect: job counts and every queued or running job),
``job`` (one job changed), ``counts`` (after any job change), ``worker`` (a worker
message). A comment line is sent every 15 s so proxies and the client keep the connection.
"""

from __future__ import annotations

import json
import queue as stdlib_queue
import threading
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import Engine, or_
from sqlmodel import Session, col, select

from clipsdex_engine.db.models import Job, utcnow
from clipsdex_engine.jobs import queue

POLL_SECONDS = 0.5
HEARTBEAT_SECONDS = 15.0
SUBSCRIBER_BUFFER = 200  # worker messages kept per slow client; older ones are dropped
RECENT = timedelta(seconds=30)  # finished jobs still reported, so none is missed


class EventHub:
    """Fan-out of worker messages to every connected client (thread-safe)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subscribers: list[stdlib_queue.Queue[dict[str, Any]]] = []

    def subscribe(self) -> stdlib_queue.Queue[dict[str, Any]]:
        q: stdlib_queue.Queue[dict[str, Any]] = stdlib_queue.Queue(SUBSCRIBER_BUFFER)
        with self._lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: stdlib_queue.Queue[dict[str, Any]]) -> None:
        with self._lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    def publish(self, event: dict[str, Any]) -> None:
        with self._lock:
            subscribers = list(self._subscribers)
        for q in subscribers:
            try:
                q.put_nowait(event)
            except stdlib_queue.Full:  # a client that stopped reading: drop its oldest
                try:
                    q.get_nowait()
                    q.put_nowait(event)
                except (stdlib_queue.Empty, stdlib_queue.Full):
                    pass

    def worker_message(self, text: str) -> None:
        self.publish({"type": "worker", "message": text, "at": utcnow().isoformat()})


def job_event(job: Job) -> dict[str, Any]:
    return {
        "id": job.id,
        "name": job.name,
        "kind": str(job.kind),
        "video_id": job.video_id,
        "status": str(job.status),
        "progress": round(job.progress, 3),
        "attempts": job.attempts,
        "error": job.error,
        "resource": job.resource,
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "finished_at": job.finished_at.isoformat() if job.finished_at else None,
    }


def _key(event: dict[str, Any]) -> tuple[Any, ...]:
    return (event["status"], event["progress"], event["error"], event["attempts"])


@dataclass
class JobWatcher:
    """Remembers what a client has seen and reports what changed since."""

    engine: Engine
    seen: dict[int, tuple[Any, ...]] = field(default_factory=dict)
    since: datetime = field(default_factory=utcnow)

    def _jobs(self, ids: Iterable[int] = ()) -> list[Job]:
        with Session(self.engine) as session:
            query = select(Job).where(
                or_(
                    col(Job.status).in_(queue.ACTIVE),
                    col(Job.finished_at) >= self.since - RECENT,
                    col(Job.id).in_(list(ids)),
                )
            )
            return list(session.exec(query.order_by(col(Job.id))))

    def snapshot(self) -> dict[str, Any]:
        jobs = [job_event(j) for j in self._jobs() if j.status in queue.ACTIVE]
        for e in jobs:
            self.seen[e["id"]] = _key(e)
        return {"counts": counts(self.engine), "jobs": jobs}

    def changes(self) -> list[dict[str, Any]]:
        """Jobs that are new or changed since the last call."""
        now = utcnow()
        changed: list[dict[str, Any]] = []
        for job in self._jobs(self.seen):
            e = job_event(job)
            if self.seen.get(e["id"]) != _key(e):
                self.seen[e["id"]] = _key(e)
                changed.append(e)
            if job.finished_at is not None and job.finished_at < now - 2 * RECENT:
                # Out of the query window: it can't change and won't come back.
                self.seen.pop(e["id"], None)
        self.since = now
        return changed


def counts(engine: Engine) -> dict[str, int]:
    return {str(k): v for k, v in queue.counts(engine).items()}


def sse(event: str, data: dict[str, Any], event_id: int | None = None) -> str:
    """One Server-Sent Event; ``data`` is JSON on a single line."""
    head = f"id: {event_id}\n" if event_id is not None else ""
    return f"{head}event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
