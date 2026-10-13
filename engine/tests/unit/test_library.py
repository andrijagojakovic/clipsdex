import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import Engine
from sqlmodel import Session
from typer.testing import CliRunner

from clipsdex_engine import config, db, library
from clipsdex_engine.cli import app
from clipsdex_engine.config import DataPaths
from clipsdex_engine.db.models import (
    AssignedBy,
    Category,
    CategoryAssignment,
    Profile,
    Result,
    Source,
    SourceType,
    Video,
)

T0 = datetime(2026, 1, 1, tzinfo=UTC)


def seed(engine: Engine) -> None:
    with db.session_scope(engine) as session:
        manual = Source(type=SourceType.MANUAL, display_name="Video files")
        export = Source(type=SourceType.INSTAGRAM_EXPORT, display_name="Instagram export")
        session.add_all([manual, export])
        session.flush()
        assert manual.id and export.id
        session.add_all(
            [
                Video(id=1, source_id=manual.id, file_path=r"C:\clips\Мој клип.mp4"),
                Video(
                    id=2,
                    source_id=export.id,
                    original_url="https://www.instagram.com/reel/abc/",
                    author="Coach",
                    collection_name="Business",
                ),
                Video(id=3, source_id=export.id, original_url="https://x/3", author="other"),
            ]
        )
        edit = Category(id=1, name="edit")
        business = Category(id=2, name="business")
        session.add_all([edit, business])
        session.flush()
        session.add_all(
            [
                # Newer auto guess must NOT override an older user choice.
                CategoryAssignment(
                    video_id=2, category_id=2, assigned_by=AssignedBy.USER, created_at=T0
                ),
                CategoryAssignment(
                    video_id=2,
                    category_id=1,
                    assigned_by=AssignedBy.AUTO,
                    created_at=T0 + timedelta(days=1),
                ),
                CategoryAssignment(
                    video_id=3, category_id=1, assigned_by=AssignedBy.AUTO, created_at=T0
                ),
                CategoryAssignment(
                    video_id=3,
                    category_id=2,
                    assigned_by=AssignedBy.AUTO,
                    created_at=T0 + timedelta(days=1),
                ),
                Profile(video_id=2, duration_s=30.0, cut_count=0),
                Result(
                    video_id=2,
                    analyzer="brief",
                    value={"main": "old"},
                    method="llm",
                    superseded=True,
                ),
                Result(
                    video_id=2,
                    analyzer="brief",
                    value={"main": "Raise prices"},
                    confidence=0.8,
                    method="transcript + llm",
                ),
            ]
        )


@pytest.fixture
def engine(tmp_path: Path) -> Engine:
    engine = db.open_database(DataPaths(tmp_path / "data"))
    seed(engine)
    return engine


@pytest.fixture
def cli_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(config.DATA_DIR_ENV, str(tmp_path / "data"))
    seed(db.open_database())


def test_list_all_in_id_order(engine: Engine) -> None:
    with Session(engine) as session:
        assert [v.id for v in library.list_videos(session)] == [1, 2, 3]


def test_list_filters(engine: Engine) -> None:
    with Session(engine) as session:
        by_source = library.VideoFilter(source_type=SourceType.INSTAGRAM_EXPORT)
        assert [v.id for v in library.list_videos(session, by_source)] == [2, 3]
        by_author = library.VideoFilter(author="coach")  # case-insensitive
        assert [v.id for v in library.list_videos(session, by_author)] == [2]
        assert len(library.list_videos(session, library.VideoFilter(limit=1))) == 1


def test_display_name_prefers_file_name(engine: Engine) -> None:
    with Session(engine) as session:
        assert library.display_name(session.get_one(Video, 1)) == "Мој клип.mp4"
        assert library.display_name(session.get_one(Video, 2)).startswith("https://")


def test_user_category_wins_over_newer_auto(engine: Engine) -> None:
    with Session(engine) as session:
        category, by = library.current_category(session, 2)
        assert category is not None and category.name == "business"
        assert by is AssignedBy.USER


def test_latest_auto_category_without_user_choice(engine: Engine) -> None:
    with Session(engine) as session:
        category, by = library.current_category(session, 3)
        assert category is not None and category.name == "business"
        assert by is AssignedBy.AUTO


def test_details_skip_superseded_results(engine: Engine) -> None:
    with Session(engine) as session:
        details = library.video_details(session, 2)
        assert details is not None
        assert [r.value for r in details.results] == [{"main": "Raise prices"}]
        assert details.profile is not None
        assert details.source.type is SourceType.INSTAGRAM_EXPORT
        assert library.video_details(session, 999) is None


@pytest.mark.usefixtures("cli_env")
def test_cli_list_table() -> None:
    result = CliRunner().invoke(app, ["list"])
    assert result.exit_code == 0, result.output
    assert "Мој клип.mp4" in result.output
    assert "3 video(s)." in result.output


@pytest.mark.usefixtures("cli_env")
def test_cli_list_json_with_filter() -> None:
    result = CliRunner().invoke(app, ["list", "--json", "--source", "instagram_export"])
    assert result.exit_code == 0, result.output
    assert [v["id"] for v in json.loads(result.output)] == [2, 3]


def test_cli_list_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(config.DATA_DIR_ENV, str(tmp_path / "empty"))
    result = CliRunner().invoke(app, ["list"])
    assert result.exit_code == 0
    assert "clipsdex import" in result.output


@pytest.mark.usefixtures("cli_env")
def test_cli_show() -> None:
    result = CliRunner().invoke(app, ["show", "2"])
    assert result.exit_code == 0, result.output
    assert "business (user)" in result.output
    assert "cuts 0" in result.output
    assert "Raise prices" in result.output
    assert "found by transcript + llm" in result.output
    assert "old" not in result.output


@pytest.mark.usefixtures("cli_env")
def test_cli_show_unknown_id() -> None:
    result = CliRunner().invoke(app, ["show", "999"])
    assert result.exit_code == 1
    assert "No video with ID 999" in result.output
