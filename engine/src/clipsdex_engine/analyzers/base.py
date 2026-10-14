"""What every analyzer looks like.

An analyzer is an expensive, on-demand step (find the song, write a brief, ...). It
declares which artifacts it reads, which categories it is meant for, and a cheap rule,
based on the quick profile, that says whether running it on a video makes sense at all.
Analyzers never call each other: they only read artifacts, so work is shared and cached.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, ClassVar

from sqlmodel import Session, col, update

from clipsdex_engine.config import DataPaths, ModelChoice
from clipsdex_engine.db.models import Profile, Result, Video
from clipsdex_engine.llm import LLMProvider


class AnalyzerNotReadyError(Exception):
    """The analyzer is declared but can't run in this version yet. Message is user-facing."""


@dataclass(frozen=True)
class Applicability:
    applies: bool
    reason: str  # short, shown to the user ("almost no music (4% of the audio)")


@dataclass(frozen=True)
class AnalyzerResult:
    value: Any  # JSON-serializable
    confidence: float  # 0-1
    method: str  # how it was found, e.g. "on-screen text"
    evidence: str | None = None  # the short proof shown next to the result


def _no_progress(_fraction: float) -> None:
    pass


@dataclass
class AnalyzerContext:
    """Everything a run may use. The analyzer runs outside any database transaction."""

    video: Video
    profile: Profile | None
    artifacts: dict[str, Any]  # artifact name -> data, for everything in ``requires``
    models: ModelChoice
    paths: DataPaths
    provider_factory: Callable[[], LLMProvider]
    progress: Callable[[float], None] = field(default=_no_progress)

    @property
    def provider(self) -> LLMProvider:
        return self.provider_factory()


class Analyzer:
    """Base class. Subclasses set the class attributes and override ``run``."""

    name: ClassVar[str]
    title: ClassVar[str]  # what the user asks for, e.g. "Find songs"
    requires: ClassVar[tuple[str, ...]] = ()  # artifact names
    default_categories: ClassVar[tuple[str, ...]] = ()  # () = every category
    available: ClassVar[bool] = True  # False while only the declaration exists

    def resource_key(self, models: ModelChoice) -> str:
        """What the run itself needs loaded (the worker groups jobs by it)."""
        return "cpu"

    def is_applicable(self, video: Video, profile: Profile | None) -> Applicability:
        return Applicability(True, "applies to every video")

    def run(self, ctx: AnalyzerContext) -> AnalyzerResult:
        raise AnalyzerNotReadyError(f"{self.title} is not available in this version yet.")


def not_profiled() -> Applicability:
    # Without a profile nothing can be ruled out, so the analyzer runs.
    return Applicability(True, "not profiled yet, so it can't be ruled out")


def pct(ratio: float | None) -> str:
    return f"{round((ratio or 0.0) * 100)}%"


def save_result(session: Session, video_id: int, analyzer: str, result: AnalyzerResult) -> Result:
    """Store a result; earlier results of the same analyzer for the video are superseded."""
    session.exec(
        update(Result)
        .where(col(Result.video_id) == video_id, col(Result.analyzer) == analyzer)
        .values(superseded=True)
    )
    row = Result(
        video_id=video_id,
        analyzer=analyzer,
        value=result.value,
        confidence=min(1.0, max(0.0, result.confidence)),
        method=result.method,
        evidence=result.evidence,
    )
    session.add(row)
    session.flush()
    return row
