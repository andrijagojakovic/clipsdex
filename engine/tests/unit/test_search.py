import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import Engine
from sqlmodel import Session, select
from typer.testing import CliRunner

from clipsdex_engine import artifacts, categories, config, db, llm, search
from clipsdex_engine.cli import app
from clipsdex_engine.config import DataPaths
from clipsdex_engine.db.models import (
    Artifact,
    AssignedBy,
    CategoryAssignment,
    EmbeddingMeta,
    Result,
    Source,
    SourceType,
    Video,
)
from mocks.fake_llm import FakeLLM

MODEL = "nomic-embed-text"
TOPICS = ("price", "client", "negotiat", "bread", "recipe", "scene", "car", "dog")


class TopicLLM(FakeLLM):
    """Embeds by counting topic words, so similar texts get similar vectors."""

    def __init__(self, dims: int = len(TOPICS)) -> None:
        super().__init__(models=[MODEL, "other-embed"])
        self.dims = dims
        self.embedded: list[str] = []

    def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        self._check(model)
        self.embedded.extend(texts)
        size = self.dims if model == MODEL else self.dims + 2
        return [
            [float(t.lower().count(w)) + 0.01 for w in TOPICS] + [0.01] * (size - len(TOPICS))
            for t in texts
        ]


def add_artifact(session: Session, video_id: int, type_: str, data: object) -> None:
    session.add(
        Artifact(
            video_id=video_id,
            type=type_,
            params_hash=artifacts.params_hash({}),
            params={},
            data=data,
        )
    )


@pytest.fixture
def engine(tmp_path: Path) -> Engine:
    """1 business (negotiating prices), 2 food (bread), 3 movie (car scene), 4 nothing known."""
    engine = db.open_database(DataPaths(tmp_path / "data"))
    with db.session_scope(engine) as session:
        categories.ensure_default_categories(session)
        by_name = {c.name: c.id or 0 for c in categories.all_categories(session)}
        source = Source(type=SourceType.MANUAL, display_name="Video files")
        session.add(source)
        session.flush()
        assert source.id is not None
        sid = source.id
        session.add_all(
            [
                Video(id=1, source_id=sid, author="@coach",
                      saved_at=datetime(2026, 3, 1, tzinfo=UTC)),
                Video(id=2, source_id=sid, caption="My sourdough bread recipe",
                      saved_at=datetime(2026, 9, 1, tzinfo=UTC)),
                Video(id=3, source_id=sid, author="@films"),
                Video(id=4, source_id=sid),
            ]
        )  # fmt: skip
        session.flush()
        for vid, cat in ((1, "business"), (2, "food"), (3, "movie")):
            session.add(
                CategoryAssignment(
                    video_id=vid, category_id=by_name[cat], assigned_by=AssignedBy.AUTO
                )
            )
        session.add(
            Result(
                video_id=1,
                analyzer="categorize",
                value={"category": "business", "summary": "How to negotiate a price with a client.",
                       "tags": ["pricing", "clients"]},
                method="test",
            )
        )  # fmt: skip
        add_artifact(session, 1, "transcript", {"text": "Never accept the first price."})
        add_artifact(session, 3, "transcript", {"text": "Get in the car, now."})
        add_artifact(
            session, 3, "frame_descriptions",
            {"frames": [{"t": 1.0, "description": "A car chase scene at night.",
                         "shot_type": "film or TV scene"}, {"t": 5.0, "error": "x"}]},
        )  # fmt: skip
    return engine


def test_search_text_has_the_telling_parts_first(engine: Engine) -> None:
    with Session(engine) as session:
        doc = search.document(session, session.get_one(Video, 1))
        assert search.document(session, session.get_one(Video, 4)) == ""
        scenes = search.document(session, session.get_one(Video, 3))
    assert doc.splitlines() == [
        "Category: business",
        "Summary: How to negotiate a price with a client.",
        "Tags: pricing, clients",
        "Author: @coach",
        "Transcript: Never accept the first price.",
    ]
    assert "Scenes: A car chase scene at night." in scenes


def test_only_new_or_changed_videos_are_embedded(engine: Engine) -> None:
    fake = TopicLLM()
    first = search.update_index(engine, fake, MODEL)
    assert (first.embedded, first.unchanged, first.empty) == (3, 0, 1)
    assert all(t.startswith("search_document: ") for t in fake.embedded)

    again = search.update_index(engine, fake, MODEL)
    assert (again.embedded, again.unchanged) == (0, 3)

    with db.session_scope(engine) as session:
        session.add(
            Result(video_id=2, analyzer="brief", method="test",
                   value={"has_advice": True, "main_message": "Bake bread slowly.", "advice": []})
        )  # fmt: skip
    assert search.update_index(engine, fake, MODEL).embedded == 1


def test_finds_by_meaning_best_first(engine: Engine) -> None:
    fake = TopicLLM()
    search.update_index(engine, fake, MODEL)
    hits = search.search(engine, fake, MODEL, "negotiating prices with clients")
    assert fake.embedded[-1] == "search_query: negotiating prices with clients"
    assert hits[0].video_id == 1
    assert hits[0].category == "business"
    assert hits[0].snippet == "How to negotiate a price with a client."
    assert hits[0].score > hits[-1].score
    assert search.search(engine, fake, MODEL, "car scene", limit=1)[0].video_id == 3


def test_filters_narrow_the_results(engine: Engine) -> None:
    fake = TopicLLM()
    search.update_index(engine, fake, MODEL)

    def ids(**f: object) -> list[int]:
        found = search.search(engine, fake, MODEL, "price", search.SearchFilter(**f))  # type: ignore[arg-type]
        return sorted(h.video_id for h in found)

    assert ids(category="Food") == [2]
    assert ids(author="@films") == [3]
    assert ids(source_type=SourceType.INSTAGRAM_EXPORT) == []
    assert ids(since=datetime(2026, 6, 1, tzinfo=UTC), until=datetime(2026, 10, 1, tzinfo=UTC)) == [
        2
    ]


def test_new_model_or_deleted_video_rebuilds_cleanly(engine: Engine) -> None:
    fake = TopicLLM()
    search.update_index(engine, fake, MODEL)
    report = search.update_index(engine, fake, "other-embed")  # bigger vectors
    assert report.embedded == 3
    with Session(engine) as session:
        assert search.table_dims(session.connection()) == len(TOPICS) + 2
    assert search.search(engine, fake, MODEL, "price") == []  # index is for the other model

    with db.session_scope(engine) as session:
        session.delete(session.get_one(Video, 3))
    search.update_index(engine, fake, "other-embed")
    with Session(engine) as session:
        rows = session.connection().exec_driver_sql(f"SELECT video_id FROM {search.TABLE}").all()
        metas = session.exec(select(EmbeddingMeta.video_id)).all()
    assert sorted(r[0] for r in rows) == [1, 2] and sorted(metas) == [1, 2]


def test_cli_search(engine: Engine, monkeypatch: pytest.MonkeyPatch) -> None:
    root = Path(str(engine.url.database)).parent
    monkeypatch.setenv(config.DATA_DIR_ENV, str(root))
    fake = TopicLLM()
    monkeypatch.setattr(llm, "default_provider", lambda *_a, **_k: fake)
    runner = CliRunner()

    out = runner.invoke(app, ["search", "negotiating prices with clients", "--limit", "2"])
    assert out.exit_code == 0, out.output
    assert "Index: 3 updated" in out.output
    assert "#1    " in out.output and "How to negotiate a price with a client." in out.output

    as_json = runner.invoke(app, ["search", "bread", "--json", "--category", "food"])
    [hit] = json.loads(as_json.output)
    assert hit["video_id"] == 2 and "Index:" not in as_json.output

    assert runner.invoke(app, ["search", "x", "--since", "yesterday"]).exit_code == 1
    assert "Index: 0 updated, 3 unchanged" in runner.invoke(app, ["index"]).output

    monkeypatch.setattr(llm, "default_provider", lambda *_a, **_k: FakeLLM(running=False))
    down = runner.invoke(app, ["search", "x"])
    assert down.exit_code == 1 and "Search needs the embedding model" in down.output
