import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from sqlmodel import Session, select
from typer.testing import CliRunner

from clipsdex_engine import artifacts, config, db, llm
from clipsdex_engine.analyzers import brief
from clipsdex_engine.artifacts import current
from clipsdex_engine.cli import app
from clipsdex_engine.config import DataPaths
from clipsdex_engine.db.models import Artifact, Result, Source, SourceType, Video
from clipsdex_engine.jobs import builtin
from clipsdex_engine.jobs.handlers import WorkerEnv
from clipsdex_engine.jobs.worker import Worker
from mocks.fake_llm import FakeLLM

TRANSCRIPT = (
    "Three pricing mistakes. First, charging by the hour. I went from 50 dollars an hour "
    "to a 2,000 dollar package. Second, no follow-up. Send a follow-up after two days."
)


def notes_reply(**extra: object) -> str:
    notes = {
        "has_advice": True,
        "main_message": "Price packages, not hours, and always follow up.",
        "advice": ["Charge per package instead of per hour.", "Follow up after two days."],
        "facts": [
            "Went from $50 an hour to a $2,000 package.",
            "Clients who get a follow-up pay 47% more.",  # not in the video: invented
        ],
        "steps": ["Write a package offer.", "Send a follow-up after 2 days."],
        "confidence": 0.9,
    }
    return json.dumps(notes | extra)


def merge_reply(*themes: tuple[str, list[tuple[str, list[int]]]], summary: str = "S.") -> str:
    return json.dumps(
        {
            "summary": summary,
            "themes": [
                {"title": t, "points": [{"text": x, "sources": s} for x, s in pts]}
                for t, pts in themes
            ],
        }
    )


@pytest.fixture
def env(tmp_path: Path) -> tuple[WorkerEnv, FakeLLM]:
    fake = FakeLLM()
    paths = DataPaths(tmp_path / "data")
    engine = db.open_database(paths)
    with db.session_scope(engine) as session:
        source = Source(type=SourceType.MANUAL, display_name="Video files")
        session.add(source)
        session.flush()
        assert source.id is not None
        session.add(Video(id=1, source_id=source.id, file_path="C:/v/pricing.mp4"))
        session.add(
            Video(id=2, source_id=source.id, original_url="https://example.com/r/2", author="@a")
        )
        session.add(Video(id=3, source_id=source.id, file_path="C:/v/joke.mp4"))
        session.flush()
        models = db.load_settings(session).models()
        for name, data in (
            ("transcript", {"language": "en", "text": TRANSCRIPT}),
            ("ocr", {"text": "3 PRICING MISTAKES"}),
        ):
            params = current.params(name, models)
            session.add(
                Artifact(
                    video_id=1,
                    type=name,
                    params_hash=artifacts.params_hash(params),
                    params=params,
                    data=data,
                )
            )
    return WorkerEnv(engine, paths, provider_factory=lambda: fake), fake


def add_notes(env: WorkerEnv, video_id: int, **notes: Any) -> None:
    value = {
        "has_advice": True,
        "main_message": f"Message {video_id}.",
        "advice": [f"Advice {video_id}."],
        "facts": [],
        "steps": [],
    } | notes
    with db.session_scope(env.engine) as session:
        session.add(Result(video_id=video_id, analyzer="brief", value=value, method="test"))


def test_invented_numbers_are_dropped() -> None:
    assert brief.clean_list(
        [
            "Went from $50 an hour to a $2,000 package.",
            "Clients pay 47% more.",
            "Make three offers, not 3 hundred.",
            " Went from $50 an hour  to a $2,000 package. ",
        ],
        TRANSCRIPT,
    ) == ["Went from $50 an hour to a $2,000 package.", "Make three offers, not 3 hundred."]


def test_notes_job_keeps_only_what_the_video_says(env: tuple[WorkerEnv, FakeLLM]) -> None:
    worker_env, fake = env
    fake.replies = [notes_reply()]
    builtin.enqueue_analyzer(worker_env, "brief", [1])
    assert len(Worker(worker_env, builtin.HANDLERS).run().done) == 1

    [call] = fake.calls
    assert f"Transcript: <data>{TRANSCRIPT}</data>" in call.messages[1].content
    assert "On-screen text: <data>3 PRICING MISTAKES</data>" in call.messages[1].content
    assert "in English" in call.messages[0].content
    with Session(worker_env.engine) as session:
        result = session.exec(select(Result).where(Result.analyzer == "brief")).one()
    assert result.value["facts"] == ["Went from $50 an hour to a $2,000 package."]
    assert result.value["steps"] == ["Write a package offer.", "Send a follow-up after 2 days."]
    assert result.value["language"] == "en"
    assert result.method == "transcript and on-screen text read by gemma3:4b"
    assert result.evidence == "Price packages, not hours, and always follow up."
    assert result.confidence == 0.9


def test_no_advice_means_low_confidence(env: tuple[WorkerEnv, FakeLLM]) -> None:
    worker_env, fake = env
    fake.replies = [notes_reply(has_advice=False, advice=[], facts=[], steps=[])]
    builtin.enqueue_analyzer(worker_env, "brief", [1])
    Worker(worker_env, builtin.HANDLERS).run()
    with Session(worker_env.engine) as session:
        result = session.exec(select(Result).where(Result.analyzer == "brief")).one()
    assert result.confidence is not None and result.confidence <= 0.3


def sources(n: int, has_advice: bool = True) -> list[brief.Source]:
    return [
        brief.Source(i, i, f"video {i}.mp4", None, None, {
            "has_advice": has_advice, "main_message": f"Message {i}.",
            "advice": [f"Advice {i} " + "x" * 40], "facts": [], "steps": [],
        })
        for i in range(1, n + 1)
    ]  # fmt: skip


def test_merge_keeps_sources_and_drops_unknown_numbers() -> None:
    fake = FakeLLM(
        replies=[merge_reply(("Pricing", [("Sell packages.", [1, 2, 9]), ("Raise prices.", [])]))]
    )
    report = brief.merge_briefs(fake, "gemma3:4b", sources(2))
    assert report.calls == 1 and report.rounds == 1 and report.dropped_sources == [9]
    assert [p.sources for p in report.brief.themes[0].points] == [[1, 2], []]
    prompt = fake.calls[0].messages[1].content
    assert "[1] <data>Message 1.</data>" in prompt and "[2] <data>Message 2.</data>" in prompt


def test_many_videos_are_merged_in_rounds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(brief, "MERGE_INPUT_CHARS", 200)  # about two sources per call
    fake = FakeLLM(
        default_reply=merge_reply(("Final", [("All of it.", [1, 5])]), summary="Overall."),
        replies=[
            merge_reply(("A", [("From one and two.", [1, 2])])),
            merge_reply(("B", [("From three and four.", [3, 4])])),
            merge_reply(("C", [("From five.", [5])])),
        ],
    )
    report = brief.merge_briefs(fake, "gemma3:4b", sources(5))
    assert report.rounds == 2 and report.calls == 4
    last = fake.calls[-1].messages[1].content
    assert "Theme: A" in last and "From three and four.</data> [3][4]" in last
    assert report.brief.summary == "Overall."


def test_markdown_is_traceable() -> None:
    srcs = sources(2)
    srcs[1].link, srcs[1].author = "https://example.com/r/2", "@coach"
    srcs.append(sources(3, has_advice=False)[2])
    merged = brief.MergedBrief(
        summary="Two videos about pricing.",
        themes=[brief.Theme(title="Pricing", points=[brief.Point(text="Sell packages.",
                                                                 sources=[1, 2])])],
    )  # fmt: skip
    md = brief.render_markdown("Brief: pricing", srcs, merged, "gemma3:4b", date(2026, 10, 15))
    assert md.startswith("# Brief: pricing\n")
    assert "from 2 video(s) with gemma3:4b" in md
    assert "## Pricing\n\n- Sell packages. [1][2]" in md
    assert "### [1] video 1.mp4\n\n**Main message:** Message 1." in md
    assert "### [3]" not in md
    assert "2. https://example.com/r/2 by @coach" in md
    assert "3. video 3.mp4 (no advice found, left out)" in md


def test_cli_brief(
    env: tuple[WorkerEnv, FakeLLM], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    worker_env, fake = env
    monkeypatch.setenv(config.DATA_DIR_ENV, str(worker_env.paths.root))
    monkeypatch.setattr(llm, "default_provider", lambda *_a, **_k: fake)
    add_notes(worker_env, 1)
    add_notes(worker_env, 2)
    fake.replies = [merge_reply(("Pricing", [("Sell packages.", [1, 2])]))]
    runner = CliRunner()

    out_file = tmp_path / "мој brief.md"
    result = runner.invoke(
        app, ["brief", "--video", "1", "--video", "2", "--video", "3", "--output", str(out_file)]
    )
    assert result.exit_code == 0, result.output
    assert "1 video(s) have no notes yet" in result.output
    assert "clipsdex plan --analyzer brief --video 3 --queue" in result.output
    assert "Brief from 2 video(s), 1 theme(s)" in result.output
    text = out_file.read_text(encoding="utf-8")
    assert "- Sell packages. [1][2]" in text
    assert fake.events == ["load gemma3:4b", "unload gemma3:4b"]

    default = runner.invoke(app, ["brief", "--video", "1", "--no-merge", "--title", "Ideas"])
    assert default.exit_code == 0 and len(fake.calls) == 1  # no merge call for one video
    [written] = (worker_env.paths.root / "exports").glob("ideas-*.md")
    assert "## Notes per video" in written.read_text(encoding="utf-8")

    assert runner.invoke(app, ["brief", "--video", "3"]).exit_code == 1
