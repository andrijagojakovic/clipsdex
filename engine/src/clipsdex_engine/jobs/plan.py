"""Plans: what a request would compute, and how long it would take.

A request names analyzers (or artifacts directly) and a set of videos. Without a
category or video IDs, each analyzer looks only at the categories it is meant for
(``song_id``: edits and music), so nothing runs on the whole library by accident. Each
analyzer's applicability rule then skips videos where it makes no sense (no music for
``song_id``), with the reason. Videos picked by ID are taken as a deliberate choice: the
rule is not applied to them. The plan lists, per video, the artifacts missing for the
current preset, the videos that can't be done (e.g. the file is gone), and a time
estimate from ``estimates``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from sqlalchemy import Engine
from sqlmodel import Session, col, select

from clipsdex_engine import analyzers as registry
from clipsdex_engine import artifacts, db, library
from clipsdex_engine.analyzers import Analyzer
from clipsdex_engine.artifacts import audio, current, frames, transcript
from clipsdex_engine.config import ModelChoice
from clipsdex_engine.db.models import Profile, Video
from clipsdex_engine.jobs import builtin, estimates
from clipsdex_engine.jobs.handlers import WorkerEnv


@dataclass
class VideoPlan:
    video_id: int
    name: str
    analyzers: list[str] = field(default_factory=list)  # will run on this video
    skipped: dict[str, str] = field(default_factory=dict)  # analyzer -> why not
    missing: list[str] = field(default_factory=list)  # artifacts to compute first
    blocked: str | None = None  # why it can't be done, if so

    @property
    def has_work(self) -> bool:
        return not self.blocked and bool(self.missing or self.analyzers)


@dataclass
class JobEstimate:
    name: str
    videos: int
    seconds: float
    measured_jobs: int  # 0 = default estimate
    resource: str


@dataclass
class Plan:
    analyzers: tuple[str, ...]
    artifacts: tuple[str, ...]
    videos: list[VideoPlan]
    jobs: list[JobEstimate]
    load_seconds: float

    @property
    def total_seconds(self) -> float:
        return sum(j.seconds for j in self.jobs) + self.load_seconds

    @property
    def to_run(self) -> list[VideoPlan]:
        return [v for v in self.videos if v.has_work]

    @property
    def blocked(self) -> list[VideoPlan]:
        return [v for v in self.videos if v.blocked]

    @property
    def not_available(self) -> list[str]:
        """Requested analyzers that can't run in this version (only their inputs are)."""
        return [a for a in self.analyzers if not registry.get(a).available]


def check_names(analyzer_names: Iterable[str], artifact_names: Iterable[str]) -> None:
    """Raise ``ValueError`` with the known names if a name is unknown."""
    for name in analyzer_names:
        registry.get(name)
    for name in artifact_names:
        if name not in current.NAMES:
            raise ValueError(f"Unknown artifact {name!r}. Known: {', '.join(current.NAMES)}")


def _has(session: Session, models: ModelChoice, video_id: int, name: str) -> bool:
    return artifacts.find(session, video_id, name, current.params(name, models)) is not None


def artifact_resource(models: ModelChoice, name: str) -> str:
    return {
        transcript.TYPE: f"whisper:{models.whisper}",
        frames.TYPE: "cpu",
        "ocr": "rapidocr",
        "frame_descriptions": f"ollama:{models.vision}",
    }[name]


def select_videos(
    session: Session, category: str | None = None, video_ids: Iterable[int] | None = None
) -> list[Video]:
    query = select(Video).order_by(col(Video.id))
    if video_ids is not None:
        query = query.where(col(Video.id).in_(list(video_ids)))
    videos = list(session.exec(query))
    if category is None:
        return videos
    wanted = category.casefold()
    return [v for v in videos if _category(session, v) == wanted]


def _category(session: Session, video: Video) -> str | None:
    assert video.id is not None
    found, _by = library.current_category(session, video.id)
    return found.name.casefold() if found is not None else None


def _in_scope(analyzer: Analyzer, video_category: str | None) -> bool:
    defaults = {c.casefold() for c in analyzer.default_categories}
    return not defaults or video_category in defaults


def build_plan(
    engine: Engine,
    analyzer_names: Iterable[str] = (),
    artifact_names: Iterable[str] = (),
    category: str | None = None,
    video_ids: Iterable[int] | None = None,
) -> Plan:
    names = tuple(dict.fromkeys(analyzer_names))
    wanted_artifacts = tuple(dict.fromkeys(artifact_names))
    check_names(names, wanted_artifacts)
    chosen = [registry.get(n) for n in names]
    picked_by_id = video_ids is not None

    with Session(engine) as session:
        models = db.load_settings(session).models()
        budget = models.max_frames
        videos: list[VideoPlan] = []
        units: dict[str, float] = {}
        counts: dict[str, int] = {}
        for v in select_videos(session, category, video_ids):
            assert v.id is not None
            vp = VideoPlan(v.id, library.display_name(v))
            profile = session.get(Profile, v.id)
            video_category = None if picked_by_id or category else _category(session, v)
            for a in chosen:
                if not picked_by_id and category is None and not _in_scope(a, video_category):
                    continue
                if not picked_by_id:
                    check = a.is_applicable(v, profile)
                    if not check.applies:
                        vp.skipped[a.name] = check.reason
                        continue
                vp.analyzers.append(a.name)
            if not (vp.analyzers or vp.skipped or wanted_artifacts):
                continue  # outside every requested analyzer's categories

            needed = set(wanted_artifacts)
            for a in chosen:
                if a.name in vp.analyzers:
                    needed.update(a.requires)
            missing = {t for t in needed if not _has(session, models, v.id, t)}
            if missing & current.NEEDS_FRAMES and not _has(session, models, v.id, frames.TYPE):
                missing.add(frames.TYPE)
            vp.missing = [name for name in current.NAMES if name in missing]

            has_file = v.file_path is not None and not v.file_deleted
            has_audio = artifacts.find(session, v.id, audio.TYPE, audio.PARAMS) is not None
            if frames.TYPE in vp.missing and not has_file:
                vp.blocked = "the video file is gone, so key frames can't be taken"
            elif transcript.TYPE in vp.missing and not (has_file or has_audio):
                vp.blocked = "no video file and no extracted audio to transcribe"
            if not vp.blocked:
                runnable = [n for n in vp.analyzers if registry.get(n).available]
                for name in [*vp.missing, *runnable]:
                    units[name] = units.get(name, 0.0) + estimates.units_for(
                        session, name, v.id, budget
                    )
                    counts[name] = counts.get(name, 0) + 1
            videos.append(vp)

        jobs: list[JobEstimate] = []
        for name in [*current.NAMES, *names]:
            if name not in counts:
                continue
            if name in registry.REGISTRY:
                resource = registry.get(name).resource_key(models)
            else:
                resource = artifact_resource(models, name)
            r = estimates.rate(session, name, resource, budget)
            jobs.append(
                JobEstimate(
                    name, counts[name], r.seconds_per_unit * units[name], r.measured_jobs, resource
                )
            )
    loads = sum(estimates.load_seconds(res) for res in {j.resource for j in jobs})
    return Plan(names, wanted_artifacts, videos, jobs, loads)


def queue_plan(env: WorkerEnv, plan: Plan) -> list[int]:
    """Queue the plan's jobs: missing artifacts first, then the analyzers that can run,
    each waiting for the artifacts it reads."""
    ids: list[int] = []
    for name in current.NAMES:
        wanted = [v.video_id for v in plan.to_run if name in v.missing]
        if not wanted:
            continue
        enqueue = {
            "frames": builtin.enqueue_frames,
            "transcript": builtin.enqueue_transcripts,
            "ocr": builtin.enqueue_ocr,
            "frame_descriptions": builtin.enqueue_frame_descriptions,
        }[name]
        ids.extend(enqueue(env, video_ids=wanted))
    for name in plan.analyzers:
        if not registry.get(name).available:
            continue
        wanted = [v.video_id for v in plan.to_run if name in v.analyzers]
        if wanted:
            ids.extend(builtin.enqueue_analyzer(env, name, wanted))
    return list(dict.fromkeys(ids))
