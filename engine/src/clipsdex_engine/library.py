"""Read-side queries over the video library, shared by the CLI and (later) the API."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import PurePath

from sqlalchemy import case
from sqlmodel import Session, col, select

from clipsdex_engine.db.models import (
    Artifact,
    AssignedBy,
    Category,
    CategoryAssignment,
    Profile,
    Result,
    Source,
    SourceType,
    Video,
)


@dataclass(frozen=True)
class VideoFilter:
    source_type: SourceType | None = None
    author: str | None = None
    limit: int | None = None


@dataclass
class VideoDetails:
    video: Video
    source: Source
    profile: Profile | None
    category: Category | None
    category_assigned_by: AssignedBy | None
    results: list[Result] = field(default_factory=list)
    artifacts: list[Artifact] = field(default_factory=list)


def display_name(video: Video) -> str:
    """A short human label: file name, else the original link."""
    if video.file_path:
        return PurePath(video.file_path).name
    return video.original_url or f"video {video.id}"


def list_videos(session: Session, filters: VideoFilter | None = None) -> list[Video]:
    filters = filters or VideoFilter()
    query = select(Video).order_by(col(Video.id))
    if filters.source_type is not None:
        query = query.join(Source).where(Source.type == filters.source_type)
    if filters.author is not None:
        query = query.where(col(Video.author).ilike(filters.author))
    if filters.limit is not None:
        query = query.limit(filters.limit)
    return list(session.exec(query))


def current_category(
    session: Session, video_id: int
) -> tuple[Category, AssignedBy] | tuple[None, None]:
    """The latest user assignment wins; otherwise the latest automatic one (D11)."""
    user_first = case((col(CategoryAssignment.assigned_by) == AssignedBy.USER, 0), else_=1)
    query = (
        select(CategoryAssignment, Category)
        .join(Category)
        .where(CategoryAssignment.video_id == video_id)
        .order_by(
            user_first, col(CategoryAssignment.created_at).desc(), col(CategoryAssignment.id).desc()
        )
        .limit(1)
    )
    row = session.exec(query).first()
    if row is None:
        return None, None
    assignment, category = row
    return category, assignment.assigned_by


def video_details(session: Session, video_id: int) -> VideoDetails | None:
    video = session.get(Video, video_id)
    if video is None:
        return None
    source = session.get_one(Source, video.source_id)
    category, assigned_by = current_category(session, video_id)
    results = session.exec(
        select(Result)
        .where(Result.video_id == video_id, col(Result.superseded).is_(False))
        .order_by(col(Result.analyzer), col(Result.created_at))
    )
    artifacts = session.exec(
        select(Artifact).where(Artifact.video_id == video_id).order_by(col(Artifact.type))
    )
    return VideoDetails(
        video=video,
        source=source,
        profile=session.get(Profile, video_id),
        category=category,
        category_assigned_by=assigned_by,
        results=list(results),
        artifacts=list(artifacts),
    )
