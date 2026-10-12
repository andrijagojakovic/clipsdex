"""Job types available today: the quick profile and the first category guess.

Later steps register their own handlers here (transcript, frames, OCR, analyzers).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from sqlalchemy import or_
from sqlmodel import Session, col, select

from clipsdex_engine import db
from clipsdex_engine import profile as quick_profile
from clipsdex_engine.artifacts import transcript
from clipsdex_engine.db.models import Artifact, JobKind, Video
from clipsdex_engine.jobs import queue
from clipsdex_engine.jobs.handlers import (
    Handler,
    JobContext,
    OllamaResource,
    PermanentJobError,
    Resource,
    WorkerEnv,
    spec_for,
)
from clipsdex_engine.profile import category_guess

PROFILE_RESOURCE = "profile-models"  # PANNs + RapidOCR, in this process


class ProfileModelsResource(Resource):
    def __init__(self, key: str, env: WorkerEnv) -> None:
        super().__init__(key)
        self.env = env
        self.models = quick_profile.default_models(env.paths)

    def unload(self) -> None:
        # Drop the loaded models; the next load creates fresh lazy holders.
        self.models = quick_profile.default_models(self.env.paths)


class ProfileHandler:
    name = "profile"
    kind = JobKind.ARTIFACT

    def resource_key(self, video_id: int | None, params: dict[str, Any]) -> str:
        return PROFILE_RESOURCE

    def make_resource(self, env: WorkerEnv, key: str) -> Resource:
        return ProfileModelsResource(key, env)

    def run(self, ctx: JobContext) -> None:
        assert ctx.job.video_id is not None and isinstance(ctx.resource, ProfileModelsResource)
        with db.session_scope(ctx.env.engine) as session:
            video = session.get_one(Video, ctx.job.video_id)
            quick_profile.profile_video(session, ctx.env.paths, video, ctx.resource.models)


class CategoryGuessHandler:
    name = category_guess.ANALYZER
    kind = JobKind.ANALYZER

    def resource_key(self, video_id: int | None, params: dict[str, Any]) -> str:
        return f"ollama:{params['model']}"

    def make_resource(self, env: WorkerEnv, key: str) -> Resource:
        return OllamaResource(key, env, model=key.removeprefix("ollama:"))

    def run(self, ctx: JobContext) -> None:
        assert ctx.job.video_id is not None
        model = ctx.job.params["model"]
        category_guess.guess_one(ctx.env.engine, ctx.env.provider, model, ctx.job.video_id)


TranscriberFactory = Callable[[WorkerEnv, str], transcript.TranscriptSource]


def real_transcriber(env: WorkerEnv, model: str) -> transcript.TranscriptSource:
    from clipsdex_engine.models import whisper  # keeps faster-whisper out of CLI startup

    folder = whisper.ensure_model(env.paths.models, model, env.download_progress)
    return whisper.Transcriber(folder)


class WhisperResource(Resource):
    def __init__(self, key: str, env: WorkerEnv, model: str, factory: TranscriberFactory):
        super().__init__(key)
        self.env, self.model, self.factory = env, model, factory
        self.transcriber: transcript.TranscriptSource | None = None

    def load(self) -> None:
        self.transcriber = self.factory(self.env, self.model)  # downloads on first use

    def unload(self) -> None:
        self.transcriber = None


class TranscriptHandler:
    name = transcript.TYPE
    kind = JobKind.ARTIFACT

    def __init__(self, factory: TranscriberFactory = real_transcriber) -> None:
        self.factory = factory

    def resource_key(self, video_id: int | None, params: dict[str, Any]) -> str:
        return f"whisper:{params['model']}"

    def make_resource(self, env: WorkerEnv, key: str) -> Resource:
        return WhisperResource(key, env, key.removeprefix("whisper:"), self.factory)

    def run(self, ctx: JobContext) -> None:
        assert ctx.job.video_id is not None and isinstance(ctx.resource, WhisperResource)
        assert ctx.resource.transcriber is not None
        try:
            transcript.build_transcript(
                ctx.env.engine,
                ctx.env.paths,
                ctx.job.video_id,
                ctx.job.params["model"],
                ctx.resource.transcriber,
                progress=ctx.progress,
            )
        except transcript.NoAudioToTranscribeError as e:
            raise PermanentJobError(str(e)) from e


PROFILE = ProfileHandler()
CATEGORY_GUESS = CategoryGuessHandler()
TRANSCRIPT = TranscriptHandler()
HANDLERS: dict[str, Handler] = {h.name: h for h in (PROFILE, CATEGORY_GUESS, TRANSCRIPT)}


def enqueue_transcripts(
    env: WorkerEnv, video_ids: Iterable[int] | None = None, redo: bool = False
) -> list[int]:
    """Queue a transcript job (preset's Whisper model) for every video that lacks one."""
    with Session(env.engine) as session:
        model = db.load_settings(session).models().whisper
        query = select(Video).where(
            or_(
                col(Video.file_path).is_not(None),
                col(Video.id).in_(select(Artifact.video_id).where(Artifact.type == "audio")),
            )
        )
        if video_ids is not None:
            query = query.where(col(Video.id).in_(list(video_ids)))
        ids = [
            v.id
            for v in session.exec(query.order_by(col(Video.id)))
            if v.id is not None and (redo or transcript.find(session, v.id, model) is None)
        ]
    params = transcript.params_for(model)
    return queue.enqueue_many(env.engine, [spec_for(TRANSCRIPT, vid, params) for vid in ids])


def enqueue_profiling(
    env: WorkerEnv,
    video_ids: Iterable[int] | None = None,
    redo: bool = False,
    guess: bool = True,
) -> list[int]:
    """Queue a profile job per video that needs one, and a category guess after it."""
    engine = env.engine
    with Session(engine) as session:
        to_profile = [
            v.id
            for v in quick_profile.videos_to_profile(session, video_ids, redo)
            if v.id is not None
        ]
        to_guess = category_guess.videos_to_guess(session, video_ids, redo) if guess else []
        model = db.load_settings(session).models().text

    with db.session_scope(engine) as session:
        profile_jobs = {
            vid: queue.enqueue(session, spec_for(PROFILE, vid)).id for vid in to_profile
        }
        ids = [jid for jid in profile_jobs.values() if jid is not None]
        if guess:
            for vid in dict.fromkeys([*to_profile, *to_guess]):
                after = profile_jobs.get(vid)
                spec = spec_for(
                    CATEGORY_GUESS,
                    vid,
                    {"model": model},
                    depends_on=(after,) if after is not None else (),
                )
                job_id = queue.enqueue(session, spec).id
                if job_id is not None:
                    ids.append(job_id)
    return ids
