"""The engine's HTTP API for the desktop app (step 6.1).

The same functions as the CLI, as JSON over HTTP. Every route is behind the guard in
``security.py`` (loopback Host + bearer token). Long work (profiles, analyzers) is queued
as jobs; a background worker in this process does them, one heavy model at a time.
Errors come back as ``{"error": "..."}`` with a message meant for the user.
"""

from __future__ import annotations

import asyncio
import logging
import mimetypes
import queue as stdlib_queue
import threading
import time
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import asdict
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import Engine
from sqlmodel import Session, col, select
from starlette.exceptions import HTTPException as StarletteHTTPException

from clipsdex_engine import (
    __version__,
    analyzers,
    binaries,
    categories,
    db,
    export,
    library,
    llm,
    search,
)
from clipsdex_engine.analyzers import brief
from clipsdex_engine.api import events
from clipsdex_engine.api.security import (
    DEFAULT_ORIGINS,
    MEDIA_LINK_SECONDS,
    GuardMiddleware,
    MediaLinks,
)
from clipsdex_engine.artifacts import thumbnail
from clipsdex_engine.config import DataPaths, Settings
from clipsdex_engine.db.models import JobStatus, Profile, Result, Source, SourceType, Video
from clipsdex_engine.jobs import builtin, queue
from clipsdex_engine.jobs import plan as planning
from clipsdex_engine.jobs.handlers import WorkerEnv
from clipsdex_engine.jobs.worker import Worker, WorkerBusyError
from clipsdex_engine.sources import manual

log = logging.getLogger(__name__)
WORKER_RETRY_SECONDS = 10.0  # when another worker (e.g. `clipsdex worker`) holds the lock


class BackgroundWorker:
    """Runs queued jobs in a thread while the API is up. If another worker already runs
    for this data folder, it waits and tries again later."""

    def __init__(self, env: WorkerEnv, on_event: Callable[[str], None] | None = None) -> None:
        self.env = env
        self.on_event = on_event
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, name="clipsdex-worker", daemon=True)
        self.busy_elsewhere = False

    def start(self) -> None:
        self.thread.start()

    def _run(self) -> None:
        while not self.stop.is_set():
            try:
                Worker(self.env, builtin.HANDLERS, on_event=self.on_event).run(
                    until_idle=False, should_stop=self.stop.is_set
                )
                self.busy_elsewhere = False
            except WorkerBusyError:
                self.busy_elsewhere = True
                self.stop.wait(WORKER_RETRY_SECONDS)
            except Exception:
                log.exception("Background worker crashed; restarting it")
                self.stop.wait(WORKER_RETRY_SECONDS)

    def shutdown(self, timeout: float = 5.0) -> None:
        self.stop.set()
        self.thread.join(timeout)  # a long job keeps running; it is resumed next start


# --- Request bodies ----------------------------------------------------------------------


class ImportBody(BaseModel):
    paths: list[str] = Field(min_length=1)
    recursive: bool = True


class ProfileBody(BaseModel):
    video_ids: list[int] | None = None
    redo: bool = False
    guess: bool = True


class PlanBody(BaseModel):
    analyzers: list[str] = Field(default_factory=list)
    artifacts: list[str] = Field(default_factory=list)
    category: str | None = None
    video_ids: list[int] | None = None
    redo: bool = False


class CancelBody(BaseModel):
    job_ids: list[int] | None = None  # None = every queued or running job


class CategoryBody(BaseModel):
    name: str
    description: str | None = None
    color: str | None = None


class CategoryUpdateBody(BaseModel):
    new_name: str | None = None
    description: str | None = None
    color: str | None = None


class MergeBody(BaseModel):
    into: str


class MoveBody(BaseModel):
    video_ids: list[int] = Field(min_length=1)
    to: str


class BriefBody(BaseModel):
    category: str | None = None
    video_ids: list[int] | None = None
    title: str | None = None
    merge: bool = True


# --- Helpers ---------------------------------------------------------------------------


def _video_item(session: Session, v: Video, source_types: dict[int, str]) -> dict[str, Any]:
    """What a library card shows; the full record is GET /videos/{id}."""
    assert v.id is not None
    found = library.current_assignment(session, v.id)
    assignment, category = found if found else (None, None)
    profile = session.get(Profile, v.id)
    analyzed = session.exec(
        select(Result.analyzer)
        .where(Result.video_id == v.id, col(Result.superseded).is_(False))
        .distinct()
    ).all()
    return {
        "id": v.id,
        "name": library.display_name(v),
        "author": v.author,
        "caption": v.caption,
        "original_url": v.original_url,
        "source": source_types.get(v.source_id),
        "saved_at": v.saved_at.isoformat() if v.saved_at else None,
        "imported_at": v.imported_at.isoformat(),
        "duration_s": profile.duration_s if profile else None,
        "profiled": profile is not None,
        "category": category.name if category else None,
        "category_by": str(assignment.assigned_by) if assignment else None,
        "category_confidence": assignment.confidence if assignment else None,
        "results": sorted(analyzed),
        "has_file": v.file_path is not None and not v.file_deleted,
        "file_deleted": v.file_deleted,
    }


def _job(job: Any) -> dict[str, Any]:
    return {
        "id": job.id,
        "name": job.name,
        "video_id": job.video_id,
        "status": str(job.status),
        "progress": job.progress,
        "attempts": job.attempts,
        "error": job.error,
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "finished_at": job.finished_at.isoformat() if job.finished_at else None,
    }


def _plan(p: planning.Plan) -> dict[str, Any]:
    return {
        "analyzers": list(p.analyzers),
        "artifacts": list(p.artifacts),
        "videos": [asdict(v) for v in p.videos],
        "jobs": [asdict(j) for j in p.jobs],
        "load_seconds": p.load_seconds,
        "total_seconds": p.total_seconds,
        "not_available": p.not_available,
    }


def _day(value: date | None) -> datetime | None:
    return datetime(value.year, value.month, value.day, tzinfo=UTC) if value else None


# --- The app ---------------------------------------------------------------------------


def create_app(
    paths: DataPaths,
    token: str,
    provider_factory: Callable[[], llm.LLMProvider] | None = None,
    run_worker: bool = True,
    allowed_origins: tuple[str, ...] = DEFAULT_ORIGINS,
) -> FastAPI:
    """The API for one data folder. ``provider_factory`` defaults to the local Ollama."""
    engine: Engine = db.open_database(paths)
    providers = provider_factory or llm.default_provider
    env = WorkerEnv(engine, paths, provider_factory=providers)
    hub = events.EventHub()
    worker = BackgroundWorker(env, on_event=hub.worker_message) if run_worker else None

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        if worker is not None:
            worker.start()
        yield
        if worker is not None:
            worker.shutdown()

    app = FastAPI(
        title="ClipsDex engine",
        version=__version__,
        lifespan=lifespan,
        docs_url=None,  # no public pages: everything needs the token
        redoc_url=None,
        openapi_url=None,
    )
    # Order matters: the guard runs inside CORS, so preflight requests (which carry no
    # token) are answered by CORS, and every real request still needs the token.
    app.add_middleware(GuardMiddleware, token=token)
    media_links = MediaLinks()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(allowed_origins),
        allow_methods=["GET", "POST", "PATCH", "PUT", "DELETE"],
        allow_headers=["Authorization", "Content-Type"],
    )

    @contextmanager
    def reading() -> Iterator[Session]:
        with Session(engine) as session:
            yield session

    @contextmanager
    def writing() -> Iterator[Session]:
        with db.session_scope(engine) as session:
            categories.ensure_default_categories(session)
            yield session

    @app.exception_handler(categories.CategoryError)
    async def _category_error(_r: Request, e: categories.CategoryError) -> JSONResponse:
        return JSONResponse({"error": str(e)}, status_code=400)

    @app.exception_handler(llm.LLMError)
    async def _llm_error(_r: Request, e: llm.LLMError) -> JSONResponse:
        return JSONResponse({"error": str(e)}, status_code=503)

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_r: Request, e: StarletteHTTPException) -> JSONResponse:
        return JSONResponse({"error": str(e.detail)}, status_code=e.status_code)

    @app.exception_handler(RequestValidationError)
    async def _bad_request(_r: Request, e: RequestValidationError) -> JSONResponse:
        problems = [
            f"{'.'.join(str(p) for p in err.get('loc', ()) if p != 'body')}: {err.get('msg')}"
            for err in e.errors()
        ]
        return JSONResponse({"error": "Invalid request: " + "; ".join(problems)}, status_code=422)

    # Status

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "version": __version__,
            "worker": None
            if worker is None
            else ("waiting: another worker is running" if worker.busy_elsewhere else "running"),
        }

    @app.get("/doctor")
    def doctor() -> dict[str, Any]:
        tools = []
        for name in binaries.BINARIES:
            resolved = binaries.resolve(name)
            tools.append(
                {
                    "name": name,
                    "found": resolved.path is not None,
                    "path": str(resolved.path) if resolved.path else None,
                    "problem": resolved.problem,
                    "hint": None if resolved.path else binaries.INSTALL_HINTS[name],
                }
            )
        with reading() as session:
            settings = db.load_settings(session)
        provider = providers()
        status = provider.health()
        models = settings.models()
        wanted = list(dict.fromkeys([models.text, models.vision, models.embedding]))
        return {
            "data_dir": str(paths.root),
            "tools": tools,
            "runtime": {"ok": status.ok, "version": status.version, "message": status.message},
            "models": [{"name": m, "present": status.ok and provider.has_model(m)} for m in wanted],
        }

    # Settings

    @app.get("/settings")
    def get_settings() -> dict[str, Any]:
        with reading() as session:
            settings = db.load_settings(session)
        return {
            "settings": settings.model_dump(mode="json"),
            "models": settings.models().model_dump(),
        }

    @app.put("/settings")
    def put_settings(body: Settings) -> dict[str, Any]:
        with writing() as session:
            db.save_settings(session, body)
        return get_settings()

    # Library

    @app.post("/import")
    def import_videos(body: ImportBody) -> dict[str, Any]:
        report = manual.import_paths(engine, [Path(p) for p in body.paths], body.recursive)
        return {
            "added": [str(p) for p in report.added],
            "duplicates": [str(p) for p in report.duplicates],
            "failed": [{"path": str(p), "reason": r} for p, r in report.failed],
        }

    @app.get("/videos")
    def list_videos(
        category: str | None = None,
        author: str | None = None,
        source: SourceType | None = None,
        limit: int = Query(default=500, ge=1, le=10000),
    ) -> dict[str, Any]:
        with reading() as session:
            videos = library.list_videos(
                session, library.VideoFilter(source_type=source, author=author)
            )
            if category is not None:
                wanted = {v.id for v in library.select_videos(session, category=category)}
                videos = [v for v in videos if v.id in wanted]
            sources = {src.id: str(src.type) for src in session.exec(select(Source)) if src.id}
            return {"videos": [_video_item(session, v, sources) for v in videos[:limit]]}

    def _video_file(video_id: int) -> Path:
        with reading() as session:
            video = session.get(Video, video_id)
            if video is None:
                raise HTTPException(404, f"There is no video #{video_id}.")
            if not video.file_path or video.file_deleted:
                raise HTTPException(404, "The video file was removed; only the link is kept.")
            path = Path(video.file_path)
        if not path.is_file():
            raise HTTPException(404, f"The video file is no longer at {path}.")
        return path

    @app.get("/videos/{video_id}/play")
    def play_link(video_id: int) -> dict[str, Any]:
        """A link the player can use without the token (see ``security.MediaLinks``)."""
        _video_file(video_id)
        return {"url": media_links.link(video_id), "expires_in": MEDIA_LINK_SECONDS}

    @app.get("/media/{video_id}")
    def media(video_id: int, exp: int = 0, sig: str = "") -> FileResponse:
        if not media_links.valid(video_id, exp, sig):
            raise HTTPException(403, "This video link is not valid or has expired.")
        path = _video_file(video_id)
        kind = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        # Range requests are answered by FileResponse, so the player can seek.
        return FileResponse(path, media_type=kind, headers={"Cache-Control": "private"})

    @app.get("/videos/{video_id}/thumbnail")
    def video_thumbnail(video_id: int) -> FileResponse:
        path = thumbnail.thumbnail(engine, paths, video_id)
        if path is None:
            raise HTTPException(404, f"Video #{video_id} has no picture to show.")
        # The picture never changes for a video; the app may keep it.
        return FileResponse(path, media_type="image/jpeg",
                            headers={"Cache-Control": "private, max-age=86400"})  # fmt: skip

    @app.get("/videos/{video_id}")
    def get_video(video_id: int) -> dict[str, Any]:
        with reading() as session:
            video = session.get(Video, video_id)
            if video is None:
                raise HTTPException(404, f"There is no video #{video_id}.")
            return export.video_record(session, video)

    @app.post("/videos/move")
    def move_videos(body: MoveBody) -> dict[str, Any]:
        with writing() as session:
            changed = categories.move(session, body.video_ids, body.to)
        return {"changed": changed}

    # Categories

    @app.get("/categories")
    def list_categories() -> dict[str, Any]:
        with writing() as session:
            return {
                "categories": [
                    {
                        "name": c.category.name,
                        "description": c.category.description,
                        "color": c.category.color,
                        "created_by": str(c.category.created_by),
                        "videos": c.videos,
                        "by_user": c.by_user,
                    }
                    for c in categories.counts(session)
                ]
            }

    @app.post("/categories", status_code=201)
    def add_category(body: CategoryBody) -> dict[str, Any]:
        with writing() as session:
            c = categories.create(session, body.name, body.description, body.color)
            return {"name": c.name, "description": c.description, "color": c.color}

    @app.patch("/categories/{name}")
    def update_category(name: str, body: CategoryUpdateBody) -> dict[str, Any]:
        with writing() as session:
            if body.new_name is not None:
                name = categories.rename(session, name, body.new_name).name
            c = categories.describe(session, name, body.description, body.color)
            return {"name": c.name, "description": c.description, "color": c.color}

    @app.post("/categories/{name}/merge")
    def merge_category(name: str, body: MergeBody) -> dict[str, Any]:
        with writing() as session:
            return {"moved": categories.merge(session, name, body.into)}

    @app.delete("/categories/{name}")
    def delete_category(name: str) -> dict[str, Any]:
        with writing() as session:
            categories.remove(session, name)
        return {"deleted": name}

    # Work

    @app.post("/profile")
    def profile(body: ProfileBody) -> dict[str, Any]:
        ids = builtin.enqueue_profiling(env, body.video_ids, body.redo, body.guess)
        return {"queued": ids}

    def _build(body: PlanBody) -> planning.Plan:
        try:
            planning.check_names(body.analyzers, body.artifacts)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        if not (body.analyzers or body.artifacts):
            raise HTTPException(400, "Say what to plan: analyzers or artifacts.")
        return planning.build_plan(
            engine, body.analyzers, body.artifacts, body.category, body.video_ids, body.redo
        )

    @app.get("/analyzers")
    def list_analyzers() -> dict[str, Any]:
        """What the user can ask for, and whether this version can do it yet."""
        return {
            "analyzers": [
                {
                    "name": a.name,
                    "title": a.title,
                    "available": a.available,
                    "default_categories": list(a.default_categories),
                    "requires": list(a.requires),
                }
                for a in analyzers.REGISTRY.values()
            ]
        }

    @app.post("/plan")
    def plan(body: PlanBody) -> dict[str, Any]:
        return _plan(_build(body))

    @app.post("/run")
    def run(body: PlanBody) -> dict[str, Any]:
        unavailable = [a for a in body.analyzers if a in analyzers.REGISTRY
                       and not analyzers.get(a).available]  # fmt: skip
        if unavailable:
            raise HTTPException(
                400, f"Not available in this version yet: {', '.join(unavailable)}."
            )
        p = _build(body)
        return {"plan": _plan(p), "queued": planning.queue_plan(env, p)}

    @app.get("/events")
    async def live_events(
        request: Request,
        for_seconds: Annotated[float | None, Query(gt=0, le=3600)] = None,
    ) -> StreamingResponse:
        """Server-Sent Events with job and worker progress (see ``events.py``).
        ``for_seconds`` closes the stream after that long; clients then reconnect."""
        watcher = events.JobWatcher(engine)
        inbox = hub.subscribe()

        async def stream() -> AsyncIterator[str]:
            n = 0
            try:
                yield "retry: 3000\n\n"
                yield events.sse("snapshot", await run_in_threadpool(watcher.snapshot), n)
                started = beat = time.monotonic()
                while not await request.is_disconnected():
                    if for_seconds is not None and time.monotonic() - started >= for_seconds:
                        break
                    changed = await run_in_threadpool(watcher.changes)
                    for e in changed:
                        n += 1
                        yield events.sse("job", e, n)
                    if changed:
                        n += 1
                        totals = await run_in_threadpool(events.counts, engine)
                        yield events.sse("counts", totals, n)
                    while True:
                        try:
                            message = inbox.get_nowait()
                        except stdlib_queue.Empty:
                            break
                        n += 1
                        yield events.sse("worker", message, n)
                    if time.monotonic() - beat >= events.HEARTBEAT_SECONDS:
                        beat = time.monotonic()
                        yield ": ping\n\n"
                    await asyncio.sleep(events.POLL_SECONDS)
            finally:
                hub.unsubscribe(inbox)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/jobs")
    def jobs(
        status: JobStatus | None = None, limit: int = Query(default=50, ge=1, le=1000)
    ) -> dict[str, Any]:
        return {
            "counts": {str(k): v for k, v in queue.counts(engine).items()},
            "jobs": [_job(j) for j in queue.recent(engine, limit, status)],
        }

    @app.post("/jobs/cancel")
    def cancel(body: CancelBody) -> dict[str, Any]:
        return {"cancelled": queue.cancel(engine, body.job_ids)}

    # Search, briefs, exports

    @app.get("/search")
    def search_videos(
        q: str = Query(min_length=1),
        category: str | None = None,
        author: str | None = None,
        source: SourceType | None = None,
        since: date | None = None,
        until: date | None = None,
        limit: int = Query(default=10, ge=1, le=100),
        update: bool = True,
    ) -> dict[str, Any]:
        with reading() as session:
            model = db.load_settings(session).models().embedding
        provider = providers()
        if update:
            search.update_index(engine, provider, model)
        filters = search.SearchFilter(category, author, source, _day(since), _day(until))
        hits = search.search(engine, provider, model, q, filters, limit)
        return {"hits": [asdict(h) for h in hits]}

    @app.post("/brief")
    def make_brief(body: BriefBody) -> dict[str, Any]:
        try:
            doc = brief.make_brief(
                engine, providers, body.category, body.video_ids, body.title, body.merge
            )
        except brief.NothingToBriefError as e:
            raise HTTPException(400, str(e)) from e
        return asdict(doc)

    @app.get("/export/{what}", response_class=PlainTextResponse)
    def export_library(
        what: Literal["json", "csv", "songs", "movies"],
        category: str | None = None,
        video_ids: Annotated[list[int] | None, Query()] = None,
    ) -> PlainTextResponse:
        with reading() as session:
            videos = library.select_videos(session, category=category, video_ids=video_ids)
            today = datetime.now().date()
            content, media = {
                "json": lambda: (export.library_json(session, videos), "application/json"),
                "csv": lambda: (export.library_csv(session, videos), "text/csv"),
                "songs": lambda: (export.songs_markdown(session, videos, today), "text/markdown"),
                "movies": lambda: (export.movies_markdown(session, videos, today), "text/markdown"),
            }[what]()
        return PlainTextResponse(content, media_type=f"{media}; charset=utf-8")

    return app
