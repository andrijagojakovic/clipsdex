import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from sqlalchemy import Engine
from sqlmodel import Session, select

from clipsdex_engine import categories, db, library
from clipsdex_engine.config import DataPaths
from clipsdex_engine.db.models import (
    AssignedBy,
    Category,
    CategoryAssignment,
    CreatedBy,
    Profile,
    Result,
    Source,
    SourceType,
    Video,
)
from clipsdex_engine.llm import Message
from clipsdex_engine.profile import category_guess as cg
from mocks.fake_llm import FakeLLM

MODEL = "gemma3:4b"


def guess(category: str, confidence: float = 0.8, reason: str = "fast cuts and music") -> str:
    return json.dumps({"category": category, "confidence": confidence, "reason": reason})


@pytest.fixture
def engine(tmp_path: Path) -> Engine:
    engine = db.open_database(DataPaths(tmp_path / "data"))
    with db.session_scope(engine) as session:
        source = Source(type=SourceType.MANUAL, display_name="Video files")
        session.add(source)
        session.flush()
        assert source.id is not None
        for i in (1, 2):
            session.add(Video(id=i, source_id=source.id, file_path=f"C:/v/{i}.mp4"))
        session.flush()
        session.add(Profile(video_id=1, duration_s=20, cut_count=14, music_ratio=0.9))
        session.add(Profile(video_id=2, duration_s=60, cut_count=1, speech_ratio=1.0))
    return engine


def test_describe_video_full() -> None:
    video = Video(source_id=1, author="coach.mike", caption="Stop  undercharging\n#money")
    profile = Profile(
        video_id=1,
        duration_s=42.4,
        cut_count=3,
        has_audio=True,
        speech_ratio=0.9,
        music_ratio=0.1,
        silence_ratio=0.0,
        has_vocals=False,
        has_text=True,
        ocr_text="PRICING\nMISTAKE #1",
    )
    text = cg.describe_video(video, profile)
    assert "Author: coach.mike" in text
    assert "Caption: <data>Stop undercharging #money</data>" in text
    assert "Duration: 42 s" in text
    assert "Hard cuts: 3 (0.7 per 10 s)" in text
    assert "Audio: speech 90%, music 10%, silence 0%" in text
    assert "Music has singing: no" in text
    assert "On-screen text: <data>PRICING MISTAKE #1</data>" in text


def test_describe_video_sparse() -> None:
    assert cg.describe_video(Video(source_id=1), None) == "No facts available."
    no_audio = Profile(video_id=1, has_audio=False, has_text=False)
    text = cg.describe_video(Video(source_id=1), no_audio)
    assert "Audio: none" in text and "On-screen text: none" in text


def test_long_text_is_clipped() -> None:
    text = cg.describe_video(Video(source_id=1, caption="x" * 2000), None)
    assert len(text) < cg.MAX_TEXT_CHARS + 50 and text.endswith("…</data>")


def test_messages_include_category_descriptions() -> None:
    cats = [Category(name="edit", description="music edit"), Category(name="business")]
    messages = cg.build_messages(Video(source_id=1), None, cats)
    assert messages[0].role == "system" and "<data>" in messages[0].content
    assert "- edit: music edit" in messages[1].content
    assert "- business: business" in messages[1].content  # falls back to the name


def test_schema_only_allows_known_categories() -> None:
    schema = cg.guess_schema(["edit", "business"])
    assert schema.model_json_schema()["properties"]["category"]["enum"] == ["edit", "business"]
    schema.model_validate_json(guess("edit"))
    with pytest.raises(ValidationError):
        schema.model_validate_json(guess("cooking"))
    with pytest.raises(ValidationError):
        schema.model_validate_json(guess("edit", confidence=1.5))


def test_default_categories_created_once_and_never_overwritten(engine: Engine) -> None:
    with db.session_scope(engine) as session:
        categories.ensure_default_categories(session)
        first = session.exec(select(Category).where(Category.name == "edit")).one()
        first.name = "my edits"  # a user rename
    with db.session_scope(engine) as session:
        categories.ensure_default_categories(session)
        names = [c.name for c in categories.all_categories(session)]
    assert "my edits" in names and "edit" not in names
    assert len(names) == len(categories.DEFAULT_CATEGORIES)
    with Session(engine) as session:
        assert all(c.description for c in categories.all_categories(session))
        assert all(c.created_by is CreatedBy.SYSTEM for c in categories.all_categories(session))


def test_guesses_are_saved_with_how_they_were_found(engine: Engine) -> None:
    fake = FakeLLM(replies=[guess("edit", 0.85), guess("business", 0.7, "talking head")])

    report = cg.run_category_guess(engine, fake, MODEL)

    assert report.guessed == [1, 2] and report.failed == []
    assert fake.events == [f"load {MODEL}", f"unload {MODEL}"]  # one load for all
    assert "Hard cuts: 14" in fake.calls[0].messages[1].content
    with Session(engine) as session:
        details = library.video_details(session, 1)
        assert details is not None and details.category is not None
        assert details.category.name == "edit"
        assert details.category_assigned_by is AssignedBy.AUTO
        assert details.category_confidence == 0.85
        result = details.results[0]
        assert result.analyzer == cg.ANALYZER
        assert result.method == f"quick profile + {MODEL}"
        assert result.evidence == "fast cuts and music"


def test_second_run_does_nothing_and_redo_supersedes(engine: Engine) -> None:
    cg.run_category_guess(engine, FakeLLM(default_reply=guess("edit")), MODEL)
    again = FakeLLM()
    assert cg.run_category_guess(engine, again, MODEL).guessed == []
    assert again.events == []  # nothing to do: the model is not even loaded

    cg.run_category_guess(engine, FakeLLM(default_reply=guess("movie")), MODEL, [1], redo=True)
    with Session(engine) as session:
        results = session.exec(select(Result).where(Result.video_id == 1)).all()
        assert sorted(r.superseded for r in results) == [False, True]
        category, _ = library.current_category(session, 1)
        assert category is not None and category.name == "movie"


def test_user_choice_still_wins_over_a_new_guess(engine: Engine) -> None:
    with db.session_scope(engine) as session:
        categories.ensure_default_categories(session)
        food = session.exec(select(Category).where(Category.name == "food")).one()
        assert food.id is not None
        session.add(
            CategoryAssignment(video_id=1, category_id=food.id, assigned_by=AssignedBy.USER)
        )
    cg.run_category_guess(engine, FakeLLM(default_reply=guess("edit")), MODEL)
    with Session(engine) as session:
        category, by = library.current_category(session, 1)
        assert category is not None and category.name == "food" and by is AssignedBy.USER


def test_ollama_not_running_skips_with_reason(engine: Engine) -> None:
    report = cg.run_category_guess(engine, FakeLLM(running=False), MODEL)
    assert report.guessed == [] and report.skipped_reason is not None


def test_missing_model_skips_with_pull_hint(engine: Engine) -> None:
    report = cg.run_category_guess(engine, FakeLLM(models=["llama3.2:3b"]), MODEL)
    assert report.skipped_reason is not None and f"ollama pull {MODEL}" in report.skipped_reason


def test_unusable_answers_fail_one_video_only(engine: Engine) -> None:
    fake = FakeLLM(replies=["nonsense", guess("cooking"), guess("business")])
    report = cg.run_category_guess(engine, fake, MODEL)
    assert [vid for vid, _ in report.failed] == [1]
    assert report.guessed == [2]
    assert fake.events[-1] == f"unload {MODEL}"


def test_retry_tells_the_model_what_was_wrong(engine: Engine) -> None:
    fake = FakeLLM(replies=[guess("cooking"), guess("food")])
    cg.run_category_guess(engine, fake, MODEL, [1])
    retry: list[Message] = fake.calls[1].messages
    assert "not valid" in retry[-1].content
