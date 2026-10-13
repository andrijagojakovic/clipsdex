"""Manual source: import video files the user picks, or every video in a folder.

Videos are identified by a SHA-256 hash of their content, so the same file is never
imported twice, even if it was renamed or moved.

These files belong to the user: ClipsDex only reads them and never deletes them.
"Delete after analysis" applies only to files the app itself stored.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy import Engine
from sqlmodel import Session, select

from clipsdex_engine.db import session_scope
from clipsdex_engine.db.models import Source, SourceType, Video

VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi"})
_HASH_CHUNK = 1024 * 1024
_COMMIT_EVERY = 50


@dataclass
class ImportReport:
    added: list[Path] = field(default_factory=list)
    duplicates: list[Path] = field(default_factory=list)
    failed: list[tuple[Path, str]] = field(default_factory=list)


def is_video(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTENSIONS


def find_videos(paths: Iterable[Path], recursive: bool = True) -> tuple[list[Path], list[Path]]:
    """Expand files and folders into video files. Returns (videos, paths that do not exist)."""
    videos: list[Path] = []
    missing: list[Path] = []
    for path in paths:
        if path.is_file():
            if is_video(path):
                videos.append(path)
        elif path.is_dir():
            pattern = "**/*" if recursive else "*"
            videos.extend(sorted(p for p in path.glob(pattern) if p.is_file() and is_video(p)))
        else:
            missing.append(path)
    # Keep order, drop repeats (e.g. a file passed directly and inside a passed folder).
    unique = list(dict.fromkeys(p.resolve() for p in videos))
    return unique, missing


def content_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(_HASH_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def get_manual_source(session: Session) -> Source:
    source = session.exec(select(Source).where(Source.type == SourceType.MANUAL)).first()
    if source is None:
        source = Source(type=SourceType.MANUAL, display_name="Video files")
        session.add(source)
        session.flush()
    return source


def import_paths(engine: Engine, paths: Iterable[Path], recursive: bool = True) -> ImportReport:
    """Register every new video found in ``paths``. Safe to run again: duplicates are skipped."""
    report = ImportReport()
    videos, missing = find_videos(paths, recursive)
    report.failed.extend((p, "File or folder not found.") for p in missing)

    with session_scope(engine) as session:
        source = get_manual_source(session)
        assert source.id is not None
        for i, path in enumerate(videos, start=1):
            try:
                digest = content_hash(path)
            except OSError as e:
                report.failed.append((path, f"Could not read the file ({e.strerror or e})."))
                continue

            exists = session.exec(select(Video.id).where(Video.content_hash == digest)).first()
            if exists is not None:
                report.duplicates.append(path)
                continue

            session.add(Video(source_id=source.id, file_path=str(path), content_hash=digest))
            session.flush()  # makes the hash visible to the duplicate check in this batch
            report.added.append(path)
            if i % _COMMIT_EVERY == 0:
                session.commit()  # a crash in a huge folder keeps what was already imported
    return report
