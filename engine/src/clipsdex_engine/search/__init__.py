"""Semantic search (step 5.1): "that reel about negotiating with clients".

Each video gets one search text, built from everything known about it (summary, tags,
brief notes, caption, on-screen text, transcript, frame descriptions), turned into a
vector by the embedding model and stored in a sqlite-vec table. A search embeds the query
the same way and returns the closest videos, optionally only among videos that match
plain filters (category, author, source, date).

Vectors are only recomputed when a video's search text or the embedding model changed,
so indexing again after new analyses is cheap.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import sqlite_vec
from sqlalchemy import Connection, Engine, func
from sqlmodel import Session, col, select

from clipsdex_engine import db, library
from clipsdex_engine.db.models import (
    Artifact,
    EmbeddingMeta,
    Profile,
    Source,
    SourceType,
    Video,
    utcnow,
)
from clipsdex_engine.llm import LLMProvider

TABLE = db.VEC_TABLE
MAX_DOC_CHARS = 6000  # ~1,500 tokens; the most telling parts come first
BATCH = 16  # texts per embedding call
# Some embedding models expect a task prefix; without it results get clearly worse.
PREFIXES: dict[str, tuple[str, str]] = {
    "nomic-embed-text": ("search_document: ", "search_query: "),
}


@dataclass(frozen=True)
class SearchFilter:
    category: str | None = None
    author: str | None = None
    source_type: SourceType | None = None
    since: datetime | None = None  # saved (or imported) on or after
    until: datetime | None = None  # saved (or imported) before

    @property
    def active(self) -> bool:
        return any(v is not None for v in vars(self).values())


@dataclass(frozen=True)
class Hit:
    video_id: int
    score: float  # cosine similarity, 1 = same meaning
    name: str
    category: str | None
    snippet: str


@dataclass
class IndexReport:
    embedded: int = 0
    unchanged: int = 0
    empty: int = 0  # nothing known about the video yet


def prefixes(model: str) -> tuple[str, str]:
    return PREFIXES.get(model.split(":")[0], ("", ""))


def _clip(value: str | None, limit: int) -> str:
    value = " ".join((value or "").split())
    return value if len(value) <= limit else value[: limit - 1] + "…"


def _latest(session: Session, video_id: int, type_: str) -> Any:
    """Data of the newest artifact of this type (any model or settings)."""
    found = session.exec(
        select(Artifact)
        .where(Artifact.video_id == video_id, Artifact.type == type_)
        .order_by(col(Artifact.created_at).desc(), col(Artifact.id).desc())
    ).first()
    return found.data if found is not None and isinstance(found.data, dict) else None


def document(session: Session, video: Video) -> str:
    """The search text of a video, most telling parts first (long texts are cut at the end)."""
    assert video.id is not None
    parts: list[str] = []
    category, _by = library.current_category(session, video.id)
    if category is not None:
        parts.append(f"Category: {category.name}")
    categorized = library.current_result(session, video.id, "categorize")
    if categorized is not None and isinstance(categorized.value, dict):
        if summary := categorized.value.get("summary"):
            parts.append(f"Summary: {summary}")
        if tags := categorized.value.get("tags"):
            parts.append(f"Tags: {', '.join(tags)}")
    notes = library.current_result(session, video.id, "brief")
    if notes is not None and isinstance(notes.value, dict) and notes.value.get("has_advice"):
        lines = [notes.value.get("main_message") or "", *notes.value.get("advice", [])]
        parts.append("Notes: " + " ".join(line for line in lines if line))
    if video.caption:
        parts.append(f"Caption: {_clip(video.caption, 600)}")
    if video.author:
        parts.append(f"Author: {video.author}")
    if video.collection_name:
        parts.append(f"Collection: {video.collection_name}")
    ocr = _latest(session, video.id, "ocr")
    profile = session.get(Profile, video.id)
    on_screen = ocr.get("text") if ocr else (profile.ocr_text if profile else None)
    if on_screen:
        parts.append(f"On-screen text: {_clip(on_screen, 600)}")
    described = _latest(session, video.id, "frame_descriptions")
    if described:
        shots = [f["description"] for f in described.get("frames", []) if "description" in f]
        if shots:
            parts.append("Scenes: " + " ".join(shots))
    transcript = _latest(session, video.id, "transcript")
    if transcript and transcript.get("text"):
        parts.append(f"Transcript: {transcript['text']}")
    return "\n".join(parts)[:MAX_DOC_CHARS]


def text_hash(model: str, doc: str) -> str:
    return hashlib.sha256(f"{model}\n{doc}".encode()).hexdigest()[:16]


def table_dims(conn: Connection) -> int | None:
    """Vector size of the search table, or None if it does not exist yet."""
    sql = conn.exec_driver_sql(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (TABLE,)
    ).scalar()
    found = re.search(r"float\[(\d+)\]", sql or "")
    return int(found.group(1)) if found else None


def _ensure_table(session: Session, dims: int) -> None:
    conn = session.connection()
    current = table_dims(conn)
    if current == dims:
        return
    if current is not None:  # another embedding model: every vector must be redone
        conn.exec_driver_sql(f"DROP TABLE {TABLE}")
        conn.exec_driver_sql("DELETE FROM embedding_meta")
    conn.exec_driver_sql(
        f"CREATE VIRTUAL TABLE {TABLE} USING vec0("
        f"video_id INTEGER PRIMARY KEY, embedding float[{dims}] distance_metric=cosine)"
    )


def _store(session: Session, video_id: int, model: str, digest: str, vector: list[float]) -> None:
    conn = session.connection()
    conn.exec_driver_sql(f"DELETE FROM {TABLE} WHERE video_id = ?", (video_id,))
    conn.exec_driver_sql(
        f"INSERT INTO {TABLE}(video_id, embedding) VALUES (?, ?)",
        (video_id, sqlite_vec.serialize_float32(vector)),
    )
    meta = session.get(EmbeddingMeta, video_id) or EmbeddingMeta(
        video_id=video_id, model=model, dims=len(vector), text_hash=digest
    )
    meta.model, meta.dims, meta.text_hash = model, len(vector), digest
    meta.created_at = utcnow()
    session.add(meta)


def update_index(
    engine: Engine,
    provider: LLMProvider,
    model: str,
    video_ids: Iterable[int] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> IndexReport:
    """Embed every video whose search text (or the model) changed since it was indexed.

    Raises ``llm.LLMError`` subclasses if the embedding model can't be used.
    """
    report = IndexReport()
    todo: list[tuple[int, str, str]] = []  # (video id, text, hash)
    with Session(engine) as session:
        query = select(Video).order_by(col(Video.id))
        if video_ids is not None:
            query = query.where(col(Video.id).in_(list(video_ids)))
        for v in session.exec(query):
            assert v.id is not None
            doc = document(session, v)
            if not doc:
                report.empty += 1
                continue
            digest = text_hash(model, doc)
            meta = session.get(EmbeddingMeta, v.id)
            if meta is not None and meta.text_hash == digest:
                report.unchanged += 1
                continue
            todo.append((v.id, doc, digest))

    with db.session_scope(engine) as session:  # vectors of deleted videos
        conn = session.connection()
        if table_dims(conn) is not None:
            conn.exec_driver_sql(
                f"DELETE FROM {TABLE} WHERE video_id NOT IN (SELECT id FROM videos)"
            )

    doc_prefix, _query_prefix = prefixes(model)
    for start in range(0, len(todo), BATCH):
        batch = todo[start : start + BATCH]
        vectors = provider.embed(model, [doc_prefix + doc for _id, doc, _h in batch])
        with db.session_scope(engine) as session:
            _ensure_table(session, len(vectors[0]))
            for (vid, _doc, digest), vector in zip(batch, vectors, strict=True):
                _store(session, vid, model, digest, vector)
        report.embedded += len(batch)
        if progress is not None:
            progress(report.embedded, len(todo))
    return report


def _filtered_ids(session: Session, filters: SearchFilter) -> list[int]:
    query = select(Video)
    if filters.source_type is not None:
        query = query.join(Source).where(Source.type == filters.source_type)
    if filters.author is not None:
        query = query.where(col(Video.author).ilike(filters.author))
    when = func.coalesce(Video.saved_at, Video.imported_at)
    if filters.since is not None:
        query = query.where(when >= filters.since)
    if filters.until is not None:
        query = query.where(when < filters.until)
    ids: list[int] = []
    wanted = filters.category.casefold() if filters.category else None
    for v in session.exec(query):
        assert v.id is not None
        if wanted is not None:
            category, _by = library.current_category(session, v.id)
            if category is None or category.name.casefold() != wanted:
                continue
        ids.append(v.id)
    return ids


def _snippet(session: Session, video: Video) -> str:
    assert video.id is not None
    for analyzer, key in (("categorize", "summary"), ("brief", "main_message")):
        result = library.current_result(session, video.id, analyzer)
        if result is not None and isinstance(result.value, dict) and result.value.get(key):
            return _clip(result.value[key], 160)
    if video.caption:
        return _clip(video.caption, 160)
    transcript = _latest(session, video.id, "transcript")
    return _clip(transcript.get("text") if transcript else "", 160)


def search(
    engine: Engine,
    provider: LLMProvider,
    model: str,
    query: str,
    filters: SearchFilter | None = None,
    limit: int = 10,
) -> list[Hit]:
    """The videos closest in meaning to ``query``, best first."""
    filters = filters or SearchFilter()
    _doc_prefix, query_prefix = prefixes(model)
    [vector] = provider.embed(model, [query_prefix + query])
    blob = sqlite_vec.serialize_float32(vector)
    with Session(engine) as session:
        conn = session.connection()
        if table_dims(conn) != len(vector):
            return []  # nothing indexed with this model yet
        rows: Sequence[Any]
        if filters.active:
            ids = _filtered_ids(session, filters)
            if not ids:
                return []
            marks = ",".join("?" * len(ids))
            rows = conn.exec_driver_sql(
                f"SELECT video_id, vec_distance_cosine(embedding, ?) AS d FROM {TABLE} "
                f"WHERE video_id IN ({marks}) ORDER BY d LIMIT ?",
                (blob, *ids, limit),
            ).all()
        else:
            rows = conn.exec_driver_sql(
                f"SELECT video_id, distance FROM {TABLE} WHERE embedding MATCH ? AND k = ? "
                "ORDER BY distance",
                (blob, limit),
            ).all()
        hits: list[Hit] = []
        for video_id, distance in rows:
            video = session.get(Video, video_id)
            if video is None:
                continue  # deleted since it was indexed
            category, _by = library.current_category(session, video_id)
            hits.append(
                Hit(
                    video_id=video_id,
                    score=1.0 - float(distance),
                    name=library.display_name(video),
                    category=category.name if category else None,
                    snippet=_snippet(session, video),
                )
            )
    return hits
