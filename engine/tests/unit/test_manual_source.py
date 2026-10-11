import os
from pathlib import Path

import pytest
from sqlalchemy import Engine
from sqlmodel import Session, select
from typer.testing import CliRunner

from clipsdex_engine import config, db
from clipsdex_engine.cli import app
from clipsdex_engine.config import DataPaths
from clipsdex_engine.db.models import Source, SourceType, Video
from clipsdex_engine.sources import manual


@pytest.fixture
def engine(tmp_path: Path) -> Engine:
    return db.open_database(DataPaths(tmp_path / "data"))


def write_video(path: Path, size: int = 2048) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(os.urandom(size))
    return path


def stored_videos(engine: Engine) -> list[Video]:
    with Session(engine) as session:
        return list(session.exec(select(Video)))


def test_imports_folder_recursively_and_ignores_other_files(engine: Engine, tmp_path: Path) -> None:
    # Cyrillic and spaces on purpose: Windows usernames and folders can contain them.
    folder = tmp_path / "Сачувани ривси"
    a = write_video(folder / "a.mp4")
    b = write_video(folder / "sub" / "b.MOV")
    (folder / "notes.txt").write_text("not a video")

    report = manual.import_paths(engine, [folder])

    assert sorted(report.added) == sorted([a.resolve(), b.resolve()])
    videos = stored_videos(engine)
    assert {v.file_path for v in videos} == {str(a.resolve()), str(b.resolve())}
    assert all(v.content_hash and len(v.content_hash) == 64 for v in videos)


def test_second_run_imports_nothing_new(engine: Engine, tmp_path: Path) -> None:
    write_video(tmp_path / "in" / "a.mp4")
    write_video(tmp_path / "in" / "b.webm")
    manual.import_paths(engine, [tmp_path / "in"])

    report = manual.import_paths(engine, [tmp_path / "in"])

    assert report.added == []
    assert len(report.duplicates) == 2
    assert len(stored_videos(engine)) == 2


def test_renamed_copy_is_a_duplicate(engine: Engine, tmp_path: Path) -> None:
    original = write_video(tmp_path / "a.mp4")
    copy = tmp_path / "other" / "renamed.mp4"
    copy.parent.mkdir()
    copy.write_bytes(original.read_bytes())

    report = manual.import_paths(engine, [original, copy])

    assert report.added == [original.resolve()]
    assert report.duplicates == [copy.resolve()]


def test_non_recursive_skips_subfolders(engine: Engine, tmp_path: Path) -> None:
    top = write_video(tmp_path / "in" / "top.mp4")
    write_video(tmp_path / "in" / "sub" / "deep.mp4")

    report = manual.import_paths(engine, [tmp_path / "in"], recursive=False)

    assert report.added == [top.resolve()]


def test_missing_path_is_reported(engine: Engine, tmp_path: Path) -> None:
    report = manual.import_paths(engine, [tmp_path / "nope"])
    assert report.failed == [(tmp_path / "nope", "File or folder not found.")]


def test_all_imports_share_one_manual_source(engine: Engine, tmp_path: Path) -> None:
    manual.import_paths(engine, [write_video(tmp_path / "a.mp4")])
    manual.import_paths(engine, [write_video(tmp_path / "b.mp4")])
    with Session(engine) as session:
        sources = list(session.exec(select(Source)))
    assert [s.type for s in sources] == [SourceType.MANUAL]


def test_hash_is_stable(tmp_path: Path) -> None:
    path = tmp_path / "x.mp4"
    path.write_bytes(b"hello")
    assert manual.content_hash(path) == (
        "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    )


def test_cli_import_twice(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(config.DATA_DIR_ENV, str(tmp_path / "data"))
    folder = tmp_path / "videos"
    write_video(folder / "a.mp4")
    write_video(folder / "b.mp4")
    runner = CliRunner()

    first = runner.invoke(app, ["import", str(folder)])
    second = runner.invoke(app, ["import", str(folder)])

    assert first.exit_code == 0, first.output
    assert "Imported 2 new video(s), skipped 0" in first.output
    assert second.exit_code == 0, second.output
    assert "Imported 0 new video(s), skipped 2" in second.output


def test_cli_import_missing_path_exits_1(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(config.DATA_DIR_ENV, str(tmp_path / "data"))
    result = CliRunner().invoke(app, ["import", str(tmp_path / "nope")])
    assert result.exit_code == 1
    assert "[FAILED]" in result.output
