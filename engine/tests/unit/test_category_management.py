from pathlib import Path

import pytest
from sqlalchemy import Engine
from sqlmodel import Session, col, select
from typer.testing import CliRunner

from clipsdex_engine import categories, config, db, library
from clipsdex_engine.analyzers import categorize
from clipsdex_engine.categories import CategoryError
from clipsdex_engine.cli import app
from clipsdex_engine.config import DataPaths
from clipsdex_engine.db.models import (
    AssignedBy,
    Category,
    CategoryAssignment,
    Correction,
    CreatedBy,
    Source,
    SourceType,
    Video,
)


@pytest.fixture
def engine(tmp_path: Path) -> Engine:
    """Videos 1, 2 auto-guessed as edit, 3 auto-guessed as business."""
    engine = db.open_database(DataPaths(tmp_path / "data"))
    with db.session_scope(engine) as session:
        categories.ensure_default_categories(session)
        source = Source(type=SourceType.MANUAL, display_name="Video files")
        session.add(source)
        session.flush()
        assert source.id is not None
        for vid in (1, 2, 3):
            session.add(Video(id=vid, source_id=source.id, caption=f"video {vid}"))
        session.flush()
        for vid, name in ((1, "edit"), (2, "edit"), (3, "business")):
            session.add(
                CategoryAssignment(
                    video_id=vid,
                    category_id=categories.get(session, name).id or 0,
                    assigned_by=AssignedBy.AUTO,
                    confidence=0.6,
                )
            )
    return engine


def current(engine: Engine, video_id: int) -> tuple[str | None, AssignedBy | None]:
    with Session(engine) as session:
        category, by = library.current_category(session, video_id)
        return (category.name if category else None), by


def test_add_rename_describe(engine: Engine) -> None:
    with db.session_scope(engine) as session:
        c = categories.create(session, "  Car   Reviews ")
        assert (c.name, c.description, c.created_by) == (
            "Car Reviews",
            "Car Reviews",
            CreatedBy.USER,
        )
        with pytest.raises(CategoryError, match='"edit" already exists'):
            categories.create(session, "EDIT")
        with pytest.raises(CategoryError, match="at most 40"):
            categories.create(session, "x" * 41)
        with pytest.raises(CategoryError, match="#3a7bd5"):
            categories.create(session, "cats", color="blue")

        categories.rename(session, "car reviews", "Cars")
        categories.rename(session, "cars", "cars")  # only the case changes: allowed
        with pytest.raises(CategoryError, match='"food" already exists'):
            categories.rename(session, "cars", "Food")
        d = categories.describe(session, "cars", "Reviews and tests of cars.", "#3a7bd5")
        assert (d.name, d.description, d.color) == ("cars", "Reviews and tests of cars.", "#3a7bd5")
        with pytest.raises(CategoryError, match=r'no category "boats". Categories: edit'):
            categories.describe(session, "boats", "x")


def test_move_is_the_users_choice_and_a_correction(engine: Engine) -> None:
    with db.session_scope(engine) as session:
        assert categories.move(session, [1, 1, 3], "Movie") == [1, 3]
    assert current(engine, 1) == ("movie", AssignedBy.USER)
    with db.session_scope(engine) as session:
        assert categories.move(session, [1], "movie") == []  # already there
        # Confirming an automatic guess is a user choice, but not a correction.
        assert categories.move(session, [2], "edit") == [2]
        with pytest.raises(CategoryError, match="no video #9"):
            categories.move(session, [9], "edit")
    with Session(engine) as session:
        corrections = session.exec(select(Correction).order_by(col(Correction.id))).all()
        names = {c.id: c.name for c in categories.all_categories(session)}
    assert [(c.video_id, names[c.from_category_id or 0], names[c.to_category_id])
            for c in corrections] == [(1, "edit", "movie"), (3, "business", "movie")]  # fmt: skip
    assert current(engine, 2) == ("edit", AssignedBy.USER)


def test_moves_become_categorize_examples(engine: Engine) -> None:
    with db.session_scope(engine) as session:
        categories.move(session, [1], "comedy")
    with Session(engine) as session:
        [example] = categorize.user_examples(session, exclude=2)
    assert example.category == "comedy" and example.was == "edit"


def test_merge_keeps_history(engine: Engine) -> None:
    with db.session_scope(engine) as session:
        categories.move(session, [3], "comedy")  # business -> comedy
        assert categories.merge(session, "comedy", "Business") == 1
        assert categories.find(session, "comedy") is None
        with pytest.raises(CategoryError, match="into itself"):
            categories.merge(session, "edit", "EDIT")
    assert current(engine, 3) == ("business", AssignedBy.USER)
    with Session(engine) as session:
        # The correction business -> comedy became business -> business, which teaches
        # nothing, so it is gone.
        assert session.exec(select(Correction)).all() == []
        counts = {c.category.name: c.videos for c in categories.counts(session)}
    assert counts["edit"] == 2 and counts["business"] == 1


def test_delete_only_unused_categories(engine: Engine) -> None:
    with db.session_scope(engine) as session:
        categories.remove(session, "travel")
        with pytest.raises(CategoryError, match="Merge it into another category"):
            categories.remove(session, "edit")
    with Session(engine) as session:
        assert session.exec(select(Category).where(Category.name == "travel")).first() is None


def test_cli(engine: Engine, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(config.DATA_DIR_ENV, str(Path(str(engine.url.database)).parent))
    runner = CliRunner()

    listed = runner.invoke(app, ["categories"]).output
    assert "edit                2 video(s)" in listed
    assert "A music edit" in listed

    added = runner.invoke(app, ["categories", "add", "cars"])
    assert 'Added "cars".' in added.output and "categories describe" in added.output
    described = runner.invoke(app, ["categories", "describe", "cars", "Car reviews."])
    assert "cars: Car reviews." in described.output

    moved = runner.invoke(app, ["move", "1", "2", "--to", "cars"])
    assert 'Moved 2 video(s) to "cars".' in moved.output
    again = runner.invoke(app, ["move", "1", "--to", "Cars"])
    assert "Moved 0 video(s)" in again.output and "1 already there." in again.output
    assert "cars                2 video(s), 2 by you (yours)" in (
        runner.invoke(app, ["categories"]).output
    )

    renamed = runner.invoke(app, ["categories", "rename", "cars", "autos"])
    assert 'Renamed to "autos".' in renamed.output
    merged = runner.invoke(app, ["categories", "merge", "autos", "edit"])
    assert "(2 video(s) moved)" in merged.output

    bad = runner.invoke(app, ["categories", "delete", "edit"])
    assert bad.exit_code == 1 and "Merge it into another category" in bad.output
    assert runner.invoke(app, ["move", "1", "--to", "nope"]).exit_code == 1
