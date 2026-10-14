"""Plans: what a request would compute, and how long it would take (step 3.7).

A request names analyzers (or artifacts directly) and a set of videos. The plan lists, per
video, the artifacts that are missing for the current preset, the videos that can't be
done (e.g. the file is gone), and a time estimate from ``estimates``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from sqlalchemy import Engine
from sqlmodel import Session, col, select

from clipsdex_engine import artifacts, db, library
from clipsdex_engine.artifacts import audio, frame_desc, frames, ocr, transcript
from clipsdex_engine.config import Settings
from clipsdex_engine.db.models import Video
from clipsdex_engine.jobs import builtin, estimates
from clipsdex_engine.jobs.handlers import WorkerEnv

# What each analyzer reads. Step 4.1 replaces this with the analyzers' own declarations.
ANALYZER_NEEDS: dict[str, tuple[str, ...]] = {
    "brief": ("transcript", "ocr"),
    "categorize": ("transcript", "ocr", "frame_descriptions"),
    "movie_id": ("ocr", "transcript", "frame_descriptions"),
    "song_id": (),  # the audio is extracted during import
}
ARTIFACTS = ("transcript", "frames", "ocr", "frame_descriptions")
# Artifacts that are computed from the key frames.
NEEDS_FRAMES = {"ocr", "frame_descriptions"}
# Run order inside a plan (dependencies first).
ORDER = ("frames", "transcript", "ocr", "frame_descriptions")


@dataclass
class VideoPlan:
    video_id: int
    name: str
    missing: list[str] = field(default_factory=list)
    blocked: str | None = None  # why it can't be done, if so


@dataclass
class JobEstimate:
    name: str
    videos: int
    seconds: float
    measured_jobs: int  # 0 = default estimate
    resource: str


@dataclass
class Plan:
    targets: tuple[str, ...]
    videos: list[VideoPlan]
    jobs: list[JobEstimate]
    load_seconds: float

    @property
    def total_seconds(self) -> float:
        return sum(j.seconds for j in self.jobs) + self.load_seconds

    @property
    def to_run(self) -> list[VideoPlan]:
        return [v for v in self.videos if v.missing and not v.blocked]

    @property
    def blocked(self) -> list[VideoPlan]:
        return [v for v in self.videos if v.blocked]


def expand_targets(analyzers: Iterable[str], artifact_names: Iterable[str]) -> tuple[str, ...]:
    wanted: list[str] = []
    for name in analyzers:
        if name not in ANALYZER_NEEDS:
            raise ValueError(f"Unknown analyzer {name!r}. Known: {', '.join(ANALYZER_NEEDS)}")
        wanted.extend(ANALYZER_NEEDS[name])
    for name in artifact_names:
        if name not in ARTIFACTS:
            raise ValueError(f"Unknown artifact {name!r}. Known: {', '.join(ARTIFACTS)}")
        wanted.append(name)
    return tuple(dict.fromkeys(wanted))


def _has(session: Session, settings: Settings, video_id: int, name: str) -> bool:
    models = settings.models()
    budget = models.max_frames
    params = {
        "transcript": transcript.params_for(models.whisper),
        "frames": frames.params_for(budget),
        "ocr": ocr.params_for(budget),
        "frame_descriptions": frame_desc.params_for(models.vision, budget),
    }[name]
    return artifacts.find(session, video_id, name, params) is not None


def _resource(settings: Settings, name: str) -> str:
    models = settings.models()
    return {
        "transcript": f"whisper:{models.whisper}",
        "frames": "cpu",
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
    picked: list[Video] = []
    for v in videos:
        assert v.id is not None
        found, _by = library.current_category(session, v.id)
        if found is not None and found.name.casefold() == wanted:
            picked.append(v)
    return picked


def build_plan(
    engine: Engine,
    targets: tuple[str, ...],
    category: str | None = None,
    video_ids: Iterable[int] | None = None,
) -> Plan:
    with Session(engine) as session:
        settings = db.load_settings(session)
        budget = settings.models().max_frames
        videos: list[VideoPlan] = []
        units: dict[str, float] = {}
        counts: dict[str, int] = {}
        for v in select_videos(session, category, video_ids):
            assert v.id is not None
            vp = VideoPlan(v.id, library.display_name(v))
            missing = [t for t in targets if not _has(session, settings, v.id, t)]
            if any(t in NEEDS_FRAMES for t in missing) and not _has(
                session, settings, v.id, "frames"
            ):
                missing.append("frames")
            vp.missing = [name for name in ORDER if name in missing]
            has_file = v.file_path is not None and not v.file_deleted
            has_audio = artifacts.find(session, v.id, audio.TYPE, audio.PARAMS) is not None
            if "frames" in vp.missing and not has_file:
                vp.blocked = "the video file is gone, so key frames can't be taken"
            elif "transcript" in vp.missing and not (has_file or has_audio):
                vp.blocked = "no video file and no extracted audio to transcribe"
            if not vp.blocked:
                for name in vp.missing:
                    units[name] = units.get(name, 0.0) + estimates.units_for(
                        session, name, v.id, budget
                    )
                    counts[name] = counts.get(name, 0) + 1
            videos.append(vp)

        jobs: list[JobEstimate] = []
        for name in ORDER:
            if name not in counts:
                continue
            resource = _resource(settings, name)
            r = estimates.rate(session, name, resource, budget)
            jobs.append(
                JobEstimate(
                    name,
                    counts[name],
                    r.seconds_per_unit * units[name],
                    r.measured_jobs,
                    resource,
                )
            )
    loads = sum(estimates.load_seconds(res) for res in {j.resource for j in jobs})
    return Plan(targets, videos, jobs, loads)


def queue_plan(env: WorkerEnv, plan: Plan) -> list[int]:
    """Queue the missing artifact jobs of a plan (dependencies are handled by the queue)."""
    ids: list[int] = []
    for name in ORDER:
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
    return list(dict.fromkeys(ids))
