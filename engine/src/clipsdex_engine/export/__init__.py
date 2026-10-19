"""Exports (step 5.3): the library as JSON or CSV, and song / movie lists as Markdown.

The user must never be locked in: JSON holds everything the app knows about each video
(metadata, profile, current results with how they were found, transcript and on-screen
text), CSV holds one row per video for spreadsheets, and the Markdown lists are written
for notes apps like Obsidian. Briefs have their own command (``clipsdex brief``).

Song and movie results use these value shapes (the song_id and movie_id analyzers write
them): ``{"found": bool, "title", "artist"}`` and ``{"found": bool, "title", "year"}``,
each optionally with ``"candidates": [{"title", ..., "confidence"}]`` when not found.
"""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Sequence
from datetime import date, datetime
from pathlib import Path
from typing import Any

from sqlmodel import Session, col, select

from clipsdex_engine import __version__, categories, library
from clipsdex_engine.db.models import Artifact, Profile, Result, Source, Video

FORMAT_VERSION = 1
CSV_COLUMNS: tuple[str, ...] = (
    "id", "name", "original_url", "source", "author", "caption", "collection", "saved_at",
    "imported_at", "category", "category_by", "category_confidence", "duration_s", "cuts",
    "speech", "music", "silence", "has_vocals", "has_text", "language", "summary", "tags",
    "main_message", "song", "movie", "file_deleted",
)  # fmt: skip


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _latest(session: Session, video_id: int, type_: str) -> dict[str, Any] | None:
    found = session.exec(
        select(Artifact)
        .where(Artifact.video_id == video_id, Artifact.type == type_)
        .order_by(col(Artifact.created_at).desc(), col(Artifact.id).desc())
    ).first()
    return found.data if found is not None and isinstance(found.data, dict) else None


def _results(session: Session, video_id: int) -> dict[str, Result]:
    rows = session.exec(
        select(Result)
        .where(Result.video_id == video_id, col(Result.superseded).is_(False))
        .order_by(col(Result.id))
    )
    return {r.analyzer: r for r in rows}  # the newest per analyzer wins


def _value(results: dict[str, Result], analyzer: str) -> dict[str, Any]:
    r = results.get(analyzer)
    return r.value if r is not None and isinstance(r.value, dict) else {}


def song_label(value: dict[str, Any]) -> str | None:
    if not value.get("found") or not value.get("title"):
        return None
    return f"{value['title']} - {value['artist']}" if value.get("artist") else value["title"]


def movie_label(value: dict[str, Any]) -> str | None:
    if not value.get("found") or not value.get("title"):
        return None
    return f"{value['title']} ({value['year']})" if value.get("year") else value["title"]


def video_record(session: Session, video: Video) -> dict[str, Any]:
    """Everything known about one video, as plain JSON-ready data."""
    assert video.id is not None
    source = session.get_one(Source, video.source_id)
    found = library.current_assignment(session, video.id)
    assignment, category = found if found else (None, None)
    profile = session.get(Profile, video.id)
    results = _results(session, video.id)
    transcript = _latest(session, video.id, "transcript")
    ocr = _latest(session, video.id, "ocr")
    described = _latest(session, video.id, "frame_descriptions")
    return {
        "id": video.id,
        "name": library.display_name(video),
        "original_url": video.original_url,
        "file_path": video.file_path,
        "file_deleted": video.file_deleted,
        "source": str(source.type),
        "author": video.author,
        "caption": video.caption,
        "collection": video.collection_name,
        "saved_at": _iso(video.saved_at),
        "imported_at": _iso(video.imported_at),
        "category": category.name if category else None,
        "category_by": str(assignment.assigned_by) if assignment else None,
        "category_confidence": assignment.confidence if assignment else None,
        "profile": None
        if profile is None
        else {
            "duration_s": profile.duration_s,
            "width": profile.width,
            "height": profile.height,
            "has_audio": profile.has_audio,
            "cuts": profile.cut_count,
            "speech": profile.speech_ratio,
            "music": profile.music_ratio,
            "silence": profile.silence_ratio,
            "has_vocals": profile.has_vocals,
            "has_text": profile.has_text,
        },
        "results": {
            name: {
                "value": r.value,
                "confidence": r.confidence,
                "method": r.method,
                "evidence": r.evidence,
                "created_at": _iso(r.created_at),
            }
            for name, r in results.items()
        },
        "transcript": None
        if transcript is None
        else {"language": transcript.get("language"), "text": transcript.get("text")},
        "on_screen_text": (ocr or {}).get("text") or (profile.ocr_text if profile else None),
        "scenes": [f["description"] for f in (described or {}).get("frames", [])
                   if "description" in f],
    }  # fmt: skip


def library_json(session: Session, videos: Sequence[Video]) -> str:
    data = {
        "format": "clipsdex-library",
        "format_version": FORMAT_VERSION,
        "app_version": __version__,
        "exported_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "categories": [
            {"name": c.name, "description": c.description, "color": c.color,
             "created_by": str(c.created_by)}
            for c in categories.all_categories(session)
        ],
        "videos": [video_record(session, v) for v in videos],
    }  # fmt: skip
    return json.dumps(data, ensure_ascii=False, indent=2) + "\n"


def csv_row(record: dict[str, Any]) -> dict[str, Any]:
    profile = record["profile"] or {}
    results = record["results"]
    categorized = (results.get("categorize") or {}).get("value") or {}
    notes = (results.get("brief") or {}).get("value") or {}
    return {
        **{k: record[k] for k in ("id", "name", "original_url", "source", "author", "caption",
                                  "collection", "saved_at", "imported_at", "category",
                                  "category_by", "category_confidence", "file_deleted")},
        "duration_s": profile.get("duration_s"),
        "cuts": profile.get("cuts"),
        "speech": profile.get("speech"),
        "music": profile.get("music"),
        "silence": profile.get("silence"),
        "has_vocals": profile.get("has_vocals"),
        "has_text": profile.get("has_text"),
        "language": (record["transcript"] or {}).get("language"),
        "summary": categorized.get("summary"),
        "tags": "; ".join(categorized.get("tags", [])),
        "main_message": notes.get("main_message") if notes.get("has_advice") else None,
        "song": song_label((results.get("song_id") or {}).get("value") or {}),
        "movie": movie_label((results.get("movie_id") or {}).get("value") or {}),
    }  # fmt: skip


def library_csv(session: Session, videos: Sequence[Video]) -> str:
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=CSV_COLUMNS, lineterminator="\r\n")
    writer.writeheader()
    for v in videos:
        writer.writerow(csv_row(video_record(session, v)))
    return out.getvalue()


def _found_list(
    session: Session,
    videos: Sequence[Video],
    analyzer: str,
    label: Any,
    title: str,
    noun: str,
    today: date,
) -> str:
    found: list[str] = []
    missing: list[str] = []
    for v in videos:
        assert v.id is not None
        result = library.current_result(session, v.id, analyzer)
        if result is None or not isinstance(result.value, dict):
            continue
        where = v.original_url or library.display_name(v)
        conf = f"{result.confidence:.0%}" if result.confidence is not None else "?"
        how = f"found by {result.method}" + (f": {result.evidence}" if result.evidence else "")
        name = label(result.value)
        if name is not None:
            found.append(f"- **{name}** ({conf} sure)  \n  {where}  \n  *{how}*")
            continue
        guesses = [
            f"{label({'found': True, **c}) or c.get('title')} ({c.get('confidence', 0):.0%})"
            for c in result.value.get("candidates", [])
            if isinstance(c, dict) and c.get("title")
        ]
        best = f" Best guesses: {', '.join(guesses)}." if guesses else ""
        missing.append(f"- {where}: not identified.{best}")
    out = [f"# {title}", "", f"*Exported by ClipsDex on {today.isoformat()}.*", ""]
    if not found and not missing:
        out.append(f"No {noun} results yet.")
    if found:
        out += [f"## Found ({len(found)})", "", *found, ""]
    if missing:
        out += [f"## Not identified ({len(missing)})", "", *missing, ""]
    return "\n".join(out).rstrip() + "\n"


def songs_markdown(session: Session, videos: Sequence[Video], today: date) -> str:
    return _found_list(session, videos, "song_id", song_label, "Songs", "song", today)


def movies_markdown(session: Session, videos: Sequence[Video], today: date) -> str:
    return _found_list(session, videos, "movie_id", movie_label, "Movies", "movie", today)


def write_text(path: Path, content: str, *, bom: bool = False) -> None:
    """Write atomically, so a crash never leaves half an export behind. ``bom`` lets
    Excel on Windows read UTF-8 CSV correctly."""
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    part.write_text(content, encoding="utf-8-sig" if bom else "utf-8", newline="")
    part.replace(path)
