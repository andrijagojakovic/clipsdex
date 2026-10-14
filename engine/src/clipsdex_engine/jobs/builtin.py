"""Job types: the quick profile, the first category guess, artifacts and analyzers.

Analyzer jobs are generic: one handler per registered analyzer.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from sqlalchemy import or_
from sqlmodel import Session, col, select

from clipsdex_engine import analyzers, artifacts, db
from clipsdex_engine import profile as quick_profile
from clipsdex_engine.artifacts import current, frame_desc, frames, ocr, transcript
from clipsdex_engine.config import ModelChoice
from clipsdex_engine.db.models import Artifact, Job, JobKind, Profile, Video
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
from clipsdex_engine.profile.quick_ocr import RapidOcrReader, TextReader

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


class FramesHandler:
    name = frames.TYPE
    kind = JobKind.ARTIFACT

    def resource_key(self, video_id: int | None, params: dict[str, Any]) -> str:
        return "cpu"  # ffmpeg + OpenCV only, no model to load

    def make_resource(self, env: WorkerEnv, key: str) -> Resource:
        return Resource(key)

    def run(self, ctx: JobContext) -> None:
        assert ctx.job.video_id is not None
        try:
            frames.build_frames(
                ctx.env.engine,
                ctx.env.paths,
                ctx.job.video_id,
                ctx.job.params["budget"],
                progress=ctx.progress,
            )
        except frames.NoVideoFileError as e:
            raise PermanentJobError(str(e)) from e


ReaderFactory = Callable[[], TextReader]


class OcrResource(Resource):
    """RapidOCR in this process (ONNX models ship with the package; nothing to download)."""

    def __init__(self, key: str, factory: ReaderFactory) -> None:
        super().__init__(key)
        self.factory = factory
        self.reader: TextReader | None = None

    def load(self) -> None:
        self.reader = self.factory()

    def unload(self) -> None:
        self.reader = None


class OcrHandler:
    name = ocr.TYPE
    kind = JobKind.ARTIFACT

    def __init__(self, factory: ReaderFactory = RapidOcrReader) -> None:
        self.factory = factory

    def resource_key(self, video_id: int | None, params: dict[str, Any]) -> str:
        return "rapidocr"

    def make_resource(self, env: WorkerEnv, key: str) -> Resource:
        return OcrResource(key, self.factory)

    def run(self, ctx: JobContext) -> None:
        assert ctx.job.video_id is not None and isinstance(ctx.resource, OcrResource)
        assert ctx.resource.reader is not None
        try:
            ocr.build_ocr(
                ctx.env.engine,
                ctx.env.paths,
                ctx.job.video_id,
                ctx.job.params["frames"]["budget"],
                ctx.resource.reader,
                progress=ctx.progress,
            )
        except ocr.NoFramesError as e:
            raise PermanentJobError(str(e)) from e


class FrameDescriptionsHandler:
    name = frame_desc.TYPE
    kind = JobKind.ARTIFACT

    def resource_key(self, video_id: int | None, params: dict[str, Any]) -> str:
        # Same key as other jobs on this model: in the Balanced preset the text and vision
        # model are both gemma3:4b, so guesses and descriptions share one load.
        return f"ollama:{params['model']}"

    def make_resource(self, env: WorkerEnv, key: str) -> Resource:
        return OllamaResource(key, env, model=key.removeprefix("ollama:"))

    def run(self, ctx: JobContext) -> None:
        assert ctx.job.video_id is not None
        try:
            frame_desc.build_descriptions(
                ctx.env.engine,
                ctx.env.paths,
                ctx.job.video_id,
                ctx.job.params["frames"]["budget"],
                ctx.env.provider,
                ctx.job.params["model"],
                progress=ctx.progress,
            )
        except frame_desc.NoFramesError as e:
            raise PermanentJobError(str(e)) from e


PROFILE = ProfileHandler()
CATEGORY_GUESS = CategoryGuessHandler()
TRANSCRIPT = TranscriptHandler()
FRAMES = FramesHandler()
OCR = OcrHandler()
FRAME_DESCRIPTIONS = FrameDescriptionsHandler()
HANDLERS: dict[str, Handler] = {
    h.name: h for h in (PROFILE, CATEGORY_GUESS, TRANSCRIPT, FRAMES, OCR, FRAME_DESCRIPTIONS)
}


class AnalyzerHandler:
    """Runs one analyzer on one video and stores its result.

    Job params hold the models the job was planned with, so the analyzer reads exactly the
    artifacts that were computed for it, even if the preset changes in between.
    """

    kind = JobKind.ANALYZER

    def __init__(self, analyzer: analyzers.Analyzer) -> None:
        self.analyzer = analyzer
        self.name = analyzer.name

    def resource_key(self, video_id: int | None, params: dict[str, Any]) -> str:
        return self.analyzer.resource_key(ModelChoice(**params["models"]))

    def make_resource(self, env: WorkerEnv, key: str) -> Resource:
        if key.startswith("ollama:"):
            return OllamaResource(key, env, model=key.removeprefix("ollama:"))
        return Resource(key)

    def run(self, ctx: JobContext) -> None:
        video_id = ctx.job.video_id
        assert video_id is not None
        models = ModelChoice(**ctx.job.params["models"])
        with Session(ctx.env.engine) as session:
            video = session.get_one(Video, video_id)
            profile = session.get(Profile, video_id)
            inputs: dict[str, Any] = {}
            for name in self.analyzer.requires:
                found = artifacts.find(session, video_id, name, current.params(name, models))
                if found is None:
                    raise PermanentJobError(
                        f"The {name.replace('_', ' ')} for this video is missing; "
                        "plan the analyzer again to compute it first."
                    )
                inputs[name] = found.data
        actx = analyzers.AnalyzerContext(
            video=video,
            profile=profile,
            artifacts=inputs,
            models=models,
            paths=ctx.env.paths,
            provider_factory=lambda: ctx.env.provider,
            progress=ctx.progress,
        )
        try:
            result = self.analyzer.run(actx)  # outside any transaction: may take minutes
        except analyzers.AnalyzerNotReadyError as e:
            raise PermanentJobError(str(e)) from e
        with db.session_scope(ctx.env.engine) as session:
            analyzers.save_result(session, video_id, self.name, result)


for _analyzer in analyzers.REGISTRY.values():
    HANDLERS[_analyzer.name] = AnalyzerHandler(_analyzer)


def enqueue_analyzer(
    env: WorkerEnv, analyzer: analyzers.Analyzer | str, video_ids: Iterable[int]
) -> list[int]:
    """Queue an analyzer for these videos with the current models. Each job waits for the
    artifact jobs of its video that are still queued or running."""
    if isinstance(analyzer, str):
        analyzer = analyzers.get(analyzer)
    handler = AnalyzerHandler(analyzer)
    with Session(env.engine) as session:
        models = db.load_settings(session).models()
    params: dict[str, Any] = {"models": models.model_dump()}
    needs = list(analyzer.requires)
    if current.NEEDS_FRAMES.intersection(needs):
        needs.append(frames.TYPE)
    wanted = {n: current.params(n, models) for n in needs}

    ids: list[int] = []
    with db.session_scope(env.engine) as session:
        for vid in dict.fromkeys(video_ids):
            active = session.exec(
                select(Job).where(
                    Job.video_id == vid,
                    col(Job.name).in_(list(wanted)),
                    col(Job.status).in_(queue.ACTIVE),
                )
            ).all()
            after = tuple(j.id for j in active if j.id is not None and j.params == wanted[j.name])
            job = queue.enqueue(session, spec_for(handler, vid, params, depends_on=after))
            assert job.id is not None
            ids.append(job.id)
    return ids


def _enqueue_after_frames(
    env: WorkerEnv,
    handler: Handler,
    params_for: Callable[[int], dict[str, Any]],
    video_ids: Iterable[int] | None,
    redo: bool,
) -> list[int]:
    """Queue ``handler`` for videos that lack its artifact, after frame selection where the
    frames are missing too."""
    with Session(env.engine) as session:
        budget = db.load_settings(session).models().max_frames
        frames_params, params = frames.params_for(budget), params_for(budget)
        query = select(Video)
        if video_ids is not None:
            query = query.where(col(Video.id).in_(list(video_ids)))
        todo: list[tuple[int, bool]] = []  # (video id, frames still needed)
        for v in session.exec(query.order_by(col(Video.id))):
            if v.id is None:
                continue
            if not redo and artifacts.find(session, v.id, handler.name, params) is not None:
                continue
            has_frames = artifacts.find(session, v.id, frames.TYPE, frames_params) is not None
            has_file = v.file_path is not None and not v.file_deleted
            if has_frames or has_file:
                todo.append((v.id, not has_frames))

    ids: list[int] = []
    with db.session_scope(env.engine) as session:
        for vid, needs_frames in todo:
            after: tuple[int, ...] = ()
            if needs_frames:
                frames_job = queue.enqueue(session, spec_for(FRAMES, vid, frames_params))
                assert frames_job.id is not None
                ids.append(frames_job.id)
                after = (frames_job.id,)
            job = queue.enqueue(session, spec_for(handler, vid, params, depends_on=after))
            assert job.id is not None
            ids.append(job.id)
    return ids


def enqueue_frame_descriptions(
    env: WorkerEnv, video_ids: Iterable[int] | None = None, redo: bool = False
) -> list[int]:
    """Queue vision-model descriptions of the key frames (preset's vision model)."""
    with Session(env.engine) as session:
        model = db.load_settings(session).models().vision
    return _enqueue_after_frames(
        env,
        FRAME_DESCRIPTIONS,
        lambda budget: frame_desc.params_for(model, budget),
        video_ids,
        redo,
    )


def enqueue_ocr(
    env: WorkerEnv, video_ids: Iterable[int] | None = None, redo: bool = False
) -> list[int]:
    """Queue OCR over the key frames; queues frame selection first where it is missing."""
    return _enqueue_after_frames(env, OCR, ocr.params_for, video_ids, redo)


def enqueue_frames(
    env: WorkerEnv, video_ids: Iterable[int] | None = None, redo: bool = False
) -> list[int]:
    """Queue key-frame selection (preset's frame budget) for videos with a local file."""
    with Session(env.engine) as session:
        budget = db.load_settings(session).models().max_frames
        params = frames.params_for(budget)
        query = select(Video).where(
            col(Video.file_path).is_not(None), col(Video.file_deleted).is_(False)
        )
        if video_ids is not None:
            query = query.where(col(Video.id).in_(list(video_ids)))
        ids = [
            v.id
            for v in session.exec(query.order_by(col(Video.id)))
            if v.id is not None
            and (redo or artifacts.find(session, v.id, frames.TYPE, params) is None)
        ]
    return queue.enqueue_many(env.engine, [spec_for(FRAMES, vid, params) for vid in ids])


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
