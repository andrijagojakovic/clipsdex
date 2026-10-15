"""Database tables.

Conventions:
- Datetimes are timezone-aware and stored as UTC (SQLModel rejects naive ones); use ``utcnow()``.
- Enums are stored as their string values.
- Rows that belong to a video are deleted with it (ON DELETE CASCADE).
- Every schema change needs an Alembic migration in ``db/migrations/versions``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

import sqlalchemy as sa
from sqlalchemy import JSON, UniqueConstraint
from sqlmodel import Field, SQLModel


def utcnow() -> datetime:
    return datetime.now(UTC)


def enum_type(enum_cls: type[StrEnum]) -> sa.Enum:
    """Store an enum by its value ("manual"), not its name ("MANUAL")."""
    return sa.Enum(
        enum_cls,
        values_callable=lambda members: [m.value for m in members],
        native_enum=False,
        length=32,
    )


class SourceType(StrEnum):
    MANUAL = "manual"
    INSTAGRAM_EXPORT = "instagram_export"
    DOWNLOADER = "downloader"
    TIKTOK_EXPORT = "tiktok_export"


class CreatedBy(StrEnum):
    SYSTEM = "system"
    USER = "user"


class AssignedBy(StrEnum):
    AUTO = "auto"
    USER = "user"


class JobKind(StrEnum):
    ARTIFACT = "artifact"
    ANALYZER = "analyzer"


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Source(SQLModel, table=True):
    __tablename__ = "sources"  # pyright: ignore[reportAssignmentType]

    id: int | None = Field(default=None, primary_key=True)
    type: SourceType = Field(index=True, sa_type=enum_type(SourceType))
    display_name: str
    settings: dict[str, Any] = Field(default_factory=dict, sa_type=JSON)
    created_at: datetime = Field(default_factory=utcnow)


class Video(SQLModel, table=True):
    __tablename__ = "videos"  # pyright: ignore[reportAssignmentType]

    id: int | None = Field(default=None, primary_key=True)
    source_id: int = Field(foreign_key="sources.id", index=True, ondelete="CASCADE")
    original_url: str | None = Field(default=None, unique=True)
    file_path: str | None = None  # None once the file is deleted or never downloaded
    content_hash: str | None = Field(default=None, unique=True)
    author: str | None = Field(default=None, index=True)
    caption: str | None = None
    collection_name: str | None = None
    saved_at: datetime | None = None
    imported_at: datetime = Field(default_factory=utcnow)
    file_deleted: bool = False


class Profile(SQLModel, table=True):
    """Level 1 quick profile, one per video."""

    __tablename__ = "profiles"  # pyright: ignore[reportAssignmentType]

    video_id: int = Field(foreign_key="videos.id", primary_key=True, ondelete="CASCADE")
    duration_s: float | None = None
    width: int | None = None
    height: int | None = None
    has_audio: bool | None = None
    cut_count: int | None = None
    speech_ratio: float | None = None
    music_ratio: float | None = None
    silence_ratio: float | None = None
    has_vocals: bool | None = None
    has_text: bool | None = None
    ocr_text: str | None = None
    profile_version: int = 1
    created_at: datetime = Field(default_factory=utcnow)


class Artifact(SQLModel, table=True):
    __tablename__ = "artifacts"  # pyright: ignore[reportAssignmentType]
    __table_args__ = (UniqueConstraint("video_id", "type", "params_hash"),)

    id: int | None = Field(default=None, primary_key=True)
    video_id: int = Field(foreign_key="videos.id", index=True, ondelete="CASCADE")
    type: str  # "audio", "transcript", "frames", ...
    params_hash: str
    params: dict[str, Any] = Field(default_factory=dict, sa_type=JSON)
    path: str | None = None
    data: Any = Field(default=None, sa_type=JSON, nullable=True)
    created_at: datetime = Field(default_factory=utcnow)


class Result(SQLModel, table=True):
    __tablename__ = "results"  # pyright: ignore[reportAssignmentType]

    id: int | None = Field(default=None, primary_key=True)
    video_id: int = Field(foreign_key="videos.id", index=True, ondelete="CASCADE")
    analyzer: str = Field(index=True)
    value: Any = Field(default=None, sa_type=JSON, nullable=True)
    confidence: float | None = Field(default=None, ge=0, le=1)
    method: str
    evidence: str | None = None
    created_at: datetime = Field(default_factory=utcnow)
    superseded: bool = False


class Category(SQLModel, table=True):
    __tablename__ = "categories"  # pyright: ignore[reportAssignmentType]

    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(unique=True)
    # One line for the LLM prompt; small models misread bare names.
    description: str | None = None
    created_by: CreatedBy = Field(default=CreatedBy.SYSTEM, sa_type=enum_type(CreatedBy))
    color: str | None = None
    created_at: datetime = Field(default_factory=utcnow)


class CategoryAssignment(SQLModel, table=True):
    """History of assignments; the latest user assignment wins."""

    __tablename__ = "category_assignments"  # pyright: ignore[reportAssignmentType]

    id: int | None = Field(default=None, primary_key=True)
    video_id: int = Field(foreign_key="videos.id", index=True, ondelete="CASCADE")
    category_id: int = Field(foreign_key="categories.id", index=True, ondelete="CASCADE")
    assigned_by: AssignedBy = Field(sa_type=enum_type(AssignedBy))
    confidence: float | None = Field(default=None, ge=0, le=1)
    created_at: datetime = Field(default_factory=utcnow)


class Correction(SQLModel, table=True):
    """A user moving a video between categories; feeds few-shot examples."""

    __tablename__ = "corrections"  # pyright: ignore[reportAssignmentType]

    id: int | None = Field(default=None, primary_key=True)
    video_id: int = Field(foreign_key="videos.id", index=True, ondelete="CASCADE")
    from_category_id: int | None = Field(
        default=None, foreign_key="categories.id", ondelete="SET NULL"
    )
    to_category_id: int = Field(foreign_key="categories.id", ondelete="CASCADE")
    created_at: datetime = Field(default_factory=utcnow)


class Job(SQLModel, table=True):
    __tablename__ = "jobs"  # pyright: ignore[reportAssignmentType]

    id: int | None = Field(default=None, primary_key=True)
    kind: JobKind = Field(sa_type=enum_type(JobKind))
    name: str  # artifact type or analyzer name
    video_id: int | None = Field(
        default=None, foreign_key="videos.id", index=True, ondelete="CASCADE"
    )
    params: dict[str, Any] = Field(default_factory=dict, sa_type=JSON)
    status: JobStatus = Field(default=JobStatus.QUEUED, index=True, sa_type=enum_type(JobStatus))
    # What must be loaded to run it, e.g. "ollama:gemma3:4b"; the scheduler groups by it.
    resource: str = Field(default="cpu", index=True)
    progress: float = 0.0
    error: str | None = None
    attempts: int = 0
    created_at: datetime = Field(default_factory=utcnow)
    started_at: datetime | None = None
    finished_at: datetime | None = None


class JobDependency(SQLModel, table=True):
    """``job_id`` may only start once ``depends_on_id`` is done."""

    __tablename__ = "job_dependencies"  # pyright: ignore[reportAssignmentType]

    job_id: int = Field(foreign_key="jobs.id", primary_key=True, ondelete="CASCADE")
    depends_on_id: int = Field(
        foreign_key="jobs.id", primary_key=True, index=True, ondelete="CASCADE"
    )


class SettingRow(SQLModel, table=True):
    """Non-secret settings as key/value; secrets go to keyring."""

    __tablename__ = "settings"  # pyright: ignore[reportAssignmentType]

    key: str = Field(primary_key=True)
    value: Any = Field(default=None, sa_type=JSON, nullable=True)


class EmbeddingMeta(SQLModel, table=True):
    """What each video's search vector was made from. The vectors themselves live in the
    sqlite-vec table ``video_embeddings`` (created at run time, sized for the model)."""

    __tablename__ = "embedding_meta"  # pyright: ignore[reportAssignmentType]

    video_id: int = Field(foreign_key="videos.id", primary_key=True, ondelete="CASCADE")
    model: str
    dims: int
    text_hash: str  # of the indexed text; a change means the vector is stale
    created_at: datetime = Field(default_factory=utcnow)
