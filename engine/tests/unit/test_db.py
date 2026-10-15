from datetime import UTC, datetime
from pathlib import Path

import pytest
from alembic.autogenerate import compare_metadata
from alembic.runtime.migration import MigrationContext
from sqlalchemy import Engine, exc, text
from sqlmodel import Session, SQLModel, select

from clipsdex_engine import db
from clipsdex_engine.config import DataPaths, Preset, Settings
from clipsdex_engine.db.models import (
    Artifact,
    AssignedBy,
    Category,
    CategoryAssignment,
    Job,
    JobKind,
    JobStatus,
    Profile,
    Result,
    Source,
    SourceType,
    Video,
)


@pytest.fixture
def engine(tmp_path: Path) -> Engine:
    # Cyrillic and spaces on purpose: Windows usernames and folders can contain them.
    return db.open_database(DataPaths(tmp_path / "Корисник подаци" / "ClipsDex"))


def add_video(session: Session, **kwargs: object) -> Video:
    source = Source(type=SourceType.MANUAL, display_name="Files")
    session.add(source)
    session.flush()
    assert source.id is not None
    video = Video(source_id=source.id, **kwargs)  # pyright: ignore[reportArgumentType]
    session.add(video)
    session.flush()
    return video


def test_fresh_data_dir_is_migrated_to_head(engine: Engine) -> None:
    assert db.schema_revision(engine) == db.head_revision()


def test_migrate_twice_is_harmless(engine: Engine) -> None:
    db.migrate(engine)
    assert db.schema_revision(engine) == db.head_revision()


def test_models_match_migrations(engine: Engine) -> None:
    """Fails when a model changes without a new Alembic migration."""
    with engine.connect() as connection:
        context = MigrationContext.configure(
            connection, opts={"compare_type": True, "include_name": db.include_name}
        )
        diff = compare_metadata(context, SQLModel.metadata)
    assert diff == []


def test_create_and_read_video_with_children(engine: Engine) -> None:
    saved = datetime(2025, 3, 1, 12, 0, tzinfo=UTC)
    with db.session_scope(engine) as session:
        video = add_video(
            session,
            original_url="https://www.instagram.com/reel/abc/",
            author="someone",
            saved_at=saved,
        )
        assert video.id is not None
        session.add(Profile(video_id=video.id, duration_s=12.5, cut_count=7))
        session.add(
            Result(video_id=video.id, analyzer="song_id", value={"title": "X"}, method="ocr")
        )
        session.add(Artifact(video_id=video.id, type="audio", params_hash="h1", path="a.wav"))

    with Session(engine) as session:
        video = session.exec(select(Video)).one()
        assert video.author == "someone"
        assert video.saved_at == saved
        assert video.imported_at.tzinfo is not None
        assert session.get(Profile, video.id) is not None
        result = session.exec(select(Result)).one()
        assert result.value == {"title": "X"}


def test_enums_are_stored_as_values(engine: Engine) -> None:
    with db.session_scope(engine) as session:
        session.add(Job(kind=JobKind.ANALYZER, name="brief"))
    with engine.connect() as connection:
        row = connection.execute(text("SELECT kind, status FROM jobs")).one()
    assert tuple(row) == ("analyzer", "queued")
    with Session(engine) as session:
        assert session.exec(select(Job)).one().status is JobStatus.QUEUED


def test_duplicate_content_hash_is_rejected(engine: Engine) -> None:
    with pytest.raises(exc.IntegrityError), db.session_scope(engine) as session:
        add_video(session, content_hash="abc")
        add_video(session, content_hash="abc")


def test_link_only_videos_may_share_empty_hash(engine: Engine) -> None:
    with db.session_scope(engine) as session:
        add_video(session, original_url="https://example.com/1")
        add_video(session, original_url="https://example.com/2")
    with Session(engine) as session:
        assert len(session.exec(select(Video)).all()) == 2


def test_deleting_video_cascades(engine: Engine) -> None:
    with db.session_scope(engine) as session:
        video = add_video(session)
        category = Category(name="edit")
        session.add(category)
        session.flush()
        assert video.id is not None and category.id is not None
        session.add(
            CategoryAssignment(
                video_id=video.id, category_id=category.id, assigned_by=AssignedBy.AUTO
            )
        )
        session.add(Profile(video_id=video.id))
    with db.session_scope(engine) as session:
        session.delete(session.exec(select(Video)).one())
    with Session(engine) as session:
        assert session.exec(select(CategoryAssignment)).all() == []
        assert session.exec(select(Profile)).all() == []
        assert len(session.exec(select(Category)).all()) == 1


def test_settings_defaults_when_empty(engine: Engine) -> None:
    with Session(engine) as session:
        assert db.load_settings(session) == Settings()


def test_settings_round_trip(engine: Engine) -> None:
    settings = Settings(preset=Preset.FAST, max_frames=3, delete_after_analysis=False)
    with db.session_scope(engine) as session:
        db.save_settings(session, settings)
    with Session(engine) as session:
        assert db.load_settings(session) == settings


def test_unknown_stored_settings_are_ignored(engine: Engine) -> None:
    with engine.begin() as connection:
        connection.execute(text("INSERT INTO settings (key, value) VALUES ('removed_option', '1')"))
    with Session(engine) as session:
        assert db.load_settings(session) == Settings()
