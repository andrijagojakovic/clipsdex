"""What the worker needs to know about each job type.

A ``Handler`` runs one job name (e.g. "transcript"). It names the ``Resource`` its jobs need
(e.g. "ollama:gemma3:4b"); the worker keeps one resource loaded at a time and runs every
ready job of that resource before switching (D16).
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from sqlalchemy import Engine

from clipsdex_engine import llm, net
from clipsdex_engine.config import DataPaths
from clipsdex_engine.db.models import Job, JobKind
from clipsdex_engine.jobs import queue


class JobCancelled(Exception):
    """The user cancelled the job while it was running."""


class RetryableJobError(Exception):
    """A temporary problem (network, busy server): the job is queued again, up to 3 tries."""


class PermanentJobError(Exception):
    """The job cannot succeed as asked (bad file, missing model). Message is user-facing."""


@dataclass
class WorkerEnv:
    engine: Engine
    paths: DataPaths
    # None = llm.default_provider, looked up when first needed (not at import time).
    provider_factory: Callable[[], llm.LLMProvider] | None = None
    # Called while a resource downloads its model on first use (bytes done, total).
    download_progress: net.Progress | None = None
    _provider: llm.LLMProvider | None = field(default=None, init=False, repr=False)

    @property
    def provider(self) -> llm.LLMProvider:
        if self._provider is None:
            factory = self.provider_factory or llm.default_provider
            self._provider = factory()
        return self._provider


class Resource:
    """Something heavy that jobs need loaded (a model). Default: nothing to load."""

    def __init__(self, key: str) -> None:
        self.key = key

    def load(self) -> None:
        pass

    def unload(self) -> None:
        pass


class OllamaResource(Resource):
    """A model served by Ollama: loaded once for a group of jobs, then freed."""

    def __init__(self, key: str, env: WorkerEnv, model: str) -> None:
        super().__init__(key)
        self.env = env
        self.model = model

    def load(self) -> None:
        # On failure the jobs themselves report the problem, per video.
        with contextlib.suppress(llm.LLMError):
            self.env.provider.load(self.model, keep_alive="30m")

    def unload(self) -> None:
        with contextlib.suppress(llm.LLMError):
            self.env.provider.unload(self.model)


@dataclass
class JobContext:
    env: WorkerEnv
    job: Job
    resource: Resource

    @property
    def job_id(self) -> int:
        assert self.job.id is not None
        return self.job.id

    def check_cancelled(self) -> None:
        if queue.is_cancelled(self.env.engine, self.job_id):
            raise JobCancelled

    def progress(self, fraction: float) -> None:
        """Report progress; also a cancel checkpoint."""
        if not queue.set_progress(self.env.engine, self.job_id, fraction):
            self.check_cancelled()


class Handler(Protocol):
    name: str
    kind: JobKind

    def resource_key(self, video_id: int | None, params: dict[str, Any]) -> str:
        """Which resource jobs with these params need. Decided at enqueue time."""
        ...

    def make_resource(self, env: WorkerEnv, key: str) -> Resource: ...

    def run(self, ctx: JobContext) -> None:
        """Do the work. Raise RetryableJobError / PermanentJobError / JobCancelled."""
        ...


def spec_for(
    handler: Handler,
    video_id: int | None = None,
    params: dict[str, Any] | None = None,
    depends_on: tuple[int, ...] = (),
) -> queue.JobSpec:
    params = params or {}
    return queue.JobSpec(
        kind=handler.kind,
        name=handler.name,
        video_id=video_id,
        params=params,
        resource=handler.resource_key(video_id, params),
        depends_on=depends_on,
    )
