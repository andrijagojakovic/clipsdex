import csv
import io
import json
from datetime import UTC, date, datetime
from pathlib import Path

import pytest
from sqlalchemy import Engine
from sqlmodel import Session, select
from typer.testing import CliRunner

from clipsdex_engine import artifacts, categories, config, db, export
from clipsdex_engine.cli import app
from clipsdex_engine.config import DataPaths
from clipsdex_engine.db.models import (
    Artifact,
    AssignedBy,
    CategoryAssignment,
    Profile,
    Result,
    Source,
    SourceType,
    Video,
)

TODAY = date(2026, 10, 19)


@pytest.fixture
def engine(tmp_path: Path) -> Engine:
    """1 business talk with results, 2 edit with a found song, 3 edit, song not found."""
    engine = db.open_database(DataPaths(tmp_path / "data"))
    with db.session_scope(engine) as session:
        categories.ensure_default_categories(session)
        source = Source(type=SourceType.MANUAL, display_name="Video files")
        session.add(source)
        session.flush()
        assert source.id is not None
        session.add_all(
            [
                Video(id=1, source_id=source.id, file_path="C:/Видео/pricing.mp4",
                      caption="Цене, \"pricing\", and commas", author="@coach",
                      saved_at=datetime(2026, 3, 1, tzinfo=UTC)),
                Video(id=2, source_id=source.id, original_url="https://example.com/r/2"),
                Video(id=3, source_id=source.id, original_url="https://example.com/r/3"),
            ]
        )  # fmt: skip
        session.flush()
        for vid, name, by in ((1, "business", AssignedBy.USER), (2, "edit", AssignedBy.AUTO),
                              (3, "edit", AssignedBy.AUTO)):  # fmt: skip
            session.add(
                CategoryAssignment(
                    video_id=vid,
                    category_id=categories.get(session, name).id or 0,
                    assigned_by=by,
                    confidence=None if by is AssignedBy.USER else 0.7,
                )
            )
        session.add(Profile(video_id=1, duration_s=42.0, speech_ratio=0.9, has_text=True))
        session.add_all(
            [
                Result(video_id=1, analyzer="categorize", confidence=0.8, method="transcript",
                       value={"category": "business", "summary": "Price by value.",
                              "tags": ["pricing", "clients"]}),
                Result(video_id=1, analyzer="brief", confidence=0.9, method="transcript",
                       value={"has_advice": True, "main_message": "Charge for results."}),
                Result(video_id=1, analyzer="brief", superseded=True, method="old",
                       value={"has_advice": True, "main_message": "Old notes."}),
                Result(video_id=2, analyzer="song_id", confidence=0.95, method="audio fingerprint",
                       evidence="AcoustID match", value={"found": True, "title": "Song A",
                                                         "artist": "Artist A"}),
                Result(video_id=3, analyzer="song_id", confidence=0.2, method="free methods",
                       value={"found": False, "candidates": [
                           {"title": "Song B", "artist": "Artist B", "confidence": 0.2}]}),
            ]
        )  # fmt: skip
        params: dict[str, object] = {}
        session.add(
            Artifact(video_id=1, type="transcript", params_hash=artifacts.params_hash(params),
                     params=params, data={"language": "en", "text": "Charge for results."})
        )  # fmt: skip
    return engine


def videos(engine: Engine) -> tuple[Session, list[Video]]:
    session = Session(engine)
    return session, list(session.exec(select(Video).order_by(Video.id)))  # type: ignore[arg-type]


def test_json_has_everything(engine: Engine) -> None:
    session, vids = videos(engine)
    with session:
        data = json.loads(export.library_json(session, vids))
    assert data["format"] == "clipsdex-library" and data["format_version"] == 1
    assert {c["name"] for c in data["categories"]} >= {"edit", "business"}
    first = data["videos"][0]
    assert first["file_path"] == "C:/Видео/pricing.mp4" and first["name"] == "pricing.mp4"
    assert (first["category"], first["category_by"]) == ("business", "user")
    assert first["profile"]["duration_s"] == 42.0
    assert first["results"]["brief"]["value"]["main_message"] == "Charge for results."
    assert first["results"]["categorize"]["method"] == "transcript"
    assert first["transcript"] == {"language": "en", "text": "Charge for results."}
    assert data["videos"][1]["profile"] is None and data["videos"][1]["transcript"] is None


def test_csv_is_one_row_per_video(engine: Engine) -> None:
    session, vids = videos(engine)
    with session:
        text = export.library_csv(session, vids)
    assert "\r\n" in text
    rows = list(csv.DictReader(io.StringIO(text)))
    assert [r["id"] for r in rows] == ["1", "2", "3"]
    assert rows[0]["caption"] == 'Цене, "pricing", and commas'
    assert rows[0]["tags"] == "pricing; clients"
    assert rows[0]["main_message"] == "Charge for results."
    assert rows[1]["song"] == "Song A - Artist A" and rows[2]["song"] == ""
    assert rows[1]["category_confidence"] == "0.7"


def test_song_list_shows_how_and_best_guesses(engine: Engine) -> None:
    session, vids = videos(engine)
    with session:
        md = export.songs_markdown(session, vids, TODAY)
        movies = export.movies_markdown(session, vids, TODAY)
    assert md.startswith("# Songs\n\n*Exported by ClipsDex on 2026-10-19.*")
    assert "## Found (1)" in md and "- **Song A - Artist A** (95% sure)" in md
    assert "https://example.com/r/2" in md and "*found by audio fingerprint: AcoustID match*" in md
    assert "## Not identified (1)" in md
    assert "https://example.com/r/3: not identified. Best guesses: Song B - Artist B (20%)." in md
    assert "No movie results yet." in movies


def test_cli_export_writes_files(
    engine: Engine, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = Path(str(engine.url.database)).parent
    monkeypatch.setenv(config.DATA_DIR_ENV, str(root))
    runner = CliRunner()

    out = runner.invoke(app, ["export", "csv", "--category", "edit"])
    assert out.exit_code == 0, out.output
    assert "Exported 2 video(s)" in out.output
    [written] = (root / "exports").glob("library-edit-*.csv")
    assert written.read_bytes().startswith(b"\xef\xbb\xbf")  # Excel reads it as UTF-8

    target = tmp_path / "моји" / "songs.md"
    assert runner.invoke(app, ["export", "songs", "--output", str(target)]).exit_code == 0
    assert "Song A" in target.read_text(encoding="utf-8")
    assert not list(target.parent.glob("*.part"))

    js = runner.invoke(app, ["export", "json", "--video", "1"])
    assert "Exported 1 video(s)" in js.output
    assert runner.invoke(app, ["export", "pdf"]).exit_code != 0
