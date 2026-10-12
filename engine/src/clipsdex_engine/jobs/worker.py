"""The single background worker and its model-grouping scheduler (CLAUDE.md §5.4, D16).

Loop: recover interrupted jobs -> fail jobs whose dependencies failed -> pick a resource
(keep the loaded one while it has ready work, else the one with the oldest ready job) ->
load it (unloading the previous one) -> run its ready jobs one by one.

Only one worker may run per data folder (an OS file lock), otherwise two heavy models
could be loaded at once.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from filelock import FileLock, Timeout

from clipsdex_engine import llm, media, net
from clipsdex_engine.db.models import JobStatus
from clipsdex_engine.jobs import queue
from clipsdex_engine.jobs.handlers import (
    Handler,
    JobCancelled,
    JobContext,
    PermanentJobError,
    Resource,
    RetryableJobError,
    WorkerEnv,
)

log = logging.getLogger(__name__)

# Errors that may go away on their own: retry. Everything else fails the job.
RETRYABLE: tuple[type[Exception], ...] = (
    RetryableJobError,
    net.DownloadError,
    llm.ProviderUnavailableError,
)
PERMANENT: tuple[type[Exception], ...] = (PermanentJobError, media.MediaError, llm.LLMError)


class WorkerBusyError(RuntimeError):
    """Another worker already runs on this data folder."""


@dataclass
class WorkerReport:
    done: list[int] = field(default_factory=list)
    failed: list[tuple[int, str]] = field(default_factory=list)
    cancelled: list[int] = field(default_factory=list)
    retried: list[int] = field(default_factory=list)
    events: list[str] = field(default_factory=list)  # "load X", "unload X", "run #id name"


class Worker:
    def __init__(
        self,
        env: WorkerEnv,
        handlers: Mapping[str, Handler],
        poll_interval: float = 2.0,
        on_event: Callable[[str], None] | None = None,
    ) -> None:
        self.env = env
        self.handlers = dict(handlers)
        self.poll_interval = poll_interval
        self.on_event = on_event
        self.current: Resource | None = None
        self._resources: dict[str, Resource] = {}
        self.report = WorkerReport()

    def _event(self, text: str) -> None:
        self.report.events.append(text)
        log.info(text)
        if self.on_event is not None:
            self.on_event(text)

    def run(
        self, until_idle: bool = True, should_stop: Callable[[], bool] = lambda: False
    ) -> WorkerReport:
        lock = FileLock(self.env.paths.root / "worker.lock")
        try:
            lock.acquire(timeout=0)
        except Timeout as e:
            raise WorkerBusyError("Another ClipsDex worker is already running.") from e
        try:
            recovered = queue.recover_interrupted(self.env.engine)
            if recovered:
                self._event(f"resumed {recovered} interrupted job(s)")
            self._loop(until_idle, should_stop)
        finally:
            self._switch_to(None)
            lock.release()
        return self.report

    def _loop(self, until_idle: bool, should_stop: Callable[[], bool]) -> None:
        engine = self.env.engine
        while not should_stop():
            queue.fail_blocked(engine)
            ready = queue.ready_resources(engine)
            if not ready:
                if until_idle:
                    return
                time.sleep(self.poll_interval)
                continue
            keys = [key for key, _count, _oldest in ready]
            key = self.current.key if self.current and self.current.key in keys else keys[0]
            job = queue.claim_next(engine, resource=key)
            if job is None:
                continue  # someone cancelled it in between
            assert job.id is not None
            handler = self.handlers.get(job.name)
            if handler is None:
                queue.mark_failed(engine, job.id, f"No handler for job type {job.name!r}.")
                self.report.failed.append((job.id, "no handler"))
                continue
            self._switch_to(key, handler)
            assert self.current is not None
            self._run_one(handler, JobContext(self.env, job, self.current))

    def _switch_to(self, key: str | None, handler: Handler | None = None) -> None:
        if self.current is not None and self.current.key == key:
            return
        if self.current is not None:
            self._event(f"unload {self.current.key}")
            self.current.unload()
            self.current = None
        if key is None or handler is None:
            return
        resource = self._resources.get(key)
        if resource is None:
            resource = self._resources[key] = handler.make_resource(self.env, key)
        self._event(f"load {key}")
        resource.load()
        self.current = resource

    def _run_one(self, handler: Handler, ctx: JobContext) -> None:
        job_id, engine = ctx.job_id, self.env.engine
        self._event(f"run #{job_id} {ctx.job.name}")
        try:
            handler.run(ctx)
        except KeyboardInterrupt:
            queue.requeue(engine, job_id)  # resume it next time
            raise
        except JobCancelled:
            self.report.cancelled.append(job_id)
        except RETRYABLE as e:
            status = queue.mark_failed(engine, job_id, str(e), retry=True)
            if status is JobStatus.QUEUED:
                self.report.retried.append(job_id)
            elif status is JobStatus.FAILED:
                self.report.failed.append((job_id, str(e)))
        except PERMANENT as e:
            if queue.mark_failed(engine, job_id, str(e)) is not None:
                self.report.failed.append((job_id, str(e)))
        except Exception as e:
            log.exception("Job #%d (%s) crashed", job_id, ctx.job.name)
            message = f"Unexpected error: {e}. See the log file for details."
            if queue.mark_failed(engine, job_id, message) is not None:
                self.report.failed.append((job_id, message))
        else:
            if queue.mark_done(engine, job_id):
                self.report.done.append(job_id)
            else:
                self.report.cancelled.append(job_id)
