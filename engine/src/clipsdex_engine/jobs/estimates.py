"""How long jobs take on this computer (step 3.7).

Each job type has a natural unit of work: seconds of video (transcript, frames, profile),
key frames (OCR, frame descriptions) or whole videos (category guess). The rate is
"seconds per unit", taken as the median over recent finished jobs on this machine, so it
adapts to the user's hardware and one stuck job does not skew it. Until there is enough
history, conservative defaults (measured on a CPU-only laptop) are used.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Literal

from sqlmodel import Session, col, select

from clipsdex_engine import artifacts
from clipsdex_engine.artifacts import frames
from clipsdex_engine.db.models import Job, JobStatus, Profile

Unit = Literal["video_seconds", "frames", "videos"]

UNITS: dict[str, Unit] = {
    "profile": "video_seconds",
    "transcript": "video_seconds",
    "frames": "video_seconds",
    "ocr": "frames",
    "frame_descriptions": "frames",
    "category_guess": "videos",
}

# Seconds per unit when this machine has no history yet: conservative, measured on a
# 4-core-class CPU laptop without a usable GPU, plugged in.
DEFAULT_RATES: dict[str, float] = {
    "profile": 0.1,  # ~5 s for a 60 s clip (+ model loading)
    "transcript": 0.5,  # Whisper small, int8, CPU
    "frames": 0.1,  # ~3 s for a 60 s 1080x1920 clip
    "ocr": 1.5,
    "frame_descriptions": 75.0,  # gemma3:4b spends ~70 s encoding each image on CPU
    "category_guess": 10.0,
}

# One-time cost to load what a group of jobs needs (by resource key prefix).
LOAD_SECONDS: dict[str, float] = {
    "ollama:": 15.0,
    "whisper:": 5.0,
    "profile-models": 3.0,
    "rapidocr": 2.0,
}

UNKNOWN_DURATION_S = 30.0  # a typical reel, when the video was never profiled
MIN_HISTORY = 3  # finished jobs needed before trusting this machine's numbers
HISTORY = 50  # most recent finished jobs considered


@dataclass(frozen=True)
class Rate:
    seconds_per_unit: float
    measured_jobs: int  # 0 = default estimate


def load_seconds(resource: str) -> float:
    return next((s for prefix, s in LOAD_SECONDS.items() if resource.startswith(prefix)), 0.0)


def units_for(session: Session, name: str, video_id: int, budget: int) -> float:
    """How much work ``name`` is for this video, in its unit."""
    unit = UNITS.get(name, "videos")
    if unit == "videos":
        return 1.0
    if unit == "video_seconds":
        profile = session.get(Profile, video_id)
        duration = profile.duration_s if profile is not None else None
        return duration if duration and duration > 0 else UNKNOWN_DURATION_S
    found = artifacts.find(session, video_id, frames.TYPE, frames.params_for(budget))
    if found is not None and isinstance(found.data, dict):
        return float(max(1, len(found.data.get("frames", []))))
    return float(budget)  # not selected yet: at most the budget


def rate(session: Session, name: str, resource: str, budget: int) -> Rate:
    """Seconds per unit for ``name``: this machine's median if known, else the default.

    Jobs on the same resource (e.g. the same Whisper size) are preferred; other resources
    of the same job type are used only if there is not enough history.
    """
    for same_resource in (True, False):
        query = (
            select(Job)
            .where(
                Job.name == name,
                Job.status == JobStatus.DONE,
                col(Job.started_at).is_not(None),
                col(Job.finished_at).is_not(None),
            )
            .order_by(col(Job.finished_at).desc())
            .limit(HISTORY)
        )
        if same_resource:
            query = query.where(Job.resource == resource)
        per_unit: list[float] = []
        for job in session.exec(query):
            if job.video_id is None or job.started_at is None or job.finished_at is None:
                continue
            seconds = (job.finished_at - job.started_at).total_seconds()
            units = units_for(session, name, job.video_id, job.params.get("budget", budget))
            if seconds >= 0 and units > 0:
                per_unit.append(seconds / units)
        if len(per_unit) >= MIN_HISTORY:
            return Rate(statistics.median(per_unit), len(per_unit))
    return Rate(DEFAULT_RATES.get(name, 10.0), 0)


def humanize(seconds: float) -> str:
    if seconds < 60:
        return f"~{max(1, round(seconds))} s"
    if seconds < 3600:
        return f"~{round(seconds / 60)} min"
    hours, minutes = divmod(round(seconds / 60), 60)
    return f"~{hours} h {minutes} min" if minutes else f"~{hours} h"
