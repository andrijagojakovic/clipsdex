"""Job types available today: the quick profile and the first category guess.

Later steps register their own handlers here (transcript, frames, OCR, analyzers).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from sqlmodel import Session

from clipsdex_engine import db
from clipsdex_engine import profile as quick_profile
from clipsdex_engine.db.models import JobKind, Video
from clipsdex_engine.jobs import queue
from clipsdex_engine.jobs.handlers import (
    Handler,
    JobContext,
    OllamaResource,
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


PROFILE = ProfileHandler()
CATEGORY_GUESS = CategoryGuessHandler()
HANDLERS: dict[str, Handler] = {h.name: h for h in (PROFILE, CATEGORY_GUESS)}


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
