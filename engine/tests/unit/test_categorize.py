import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from sqlmodel import Session, col, select

from clipsdex_engine import artifacts, categories, db, library
from clipsdex_engine.analyzers import categorize
from clipsdex_engine.artifacts import current
from clipsdex_engine.config import DataPaths
from clipsdex_engine.db.models import (
    Artifact,
    AssignedBy,
    Category,
    CategoryAssignment,
    Correction,
    Job,
    JobDependency,
    Profile,
    Result,
    Source,
    SourceType,
    Video,
)
from clipsdex_engine.jobs import builtin
from clipsdex_engine.jobs import plan as planning
from clipsdex_engine.jobs.handlers import WorkerEnv
from clipsdex_engine.jobs.worker import Worker
from mocks.fake_llm import FakeLLM

T0 = datetime(2026, 10, 1, tzinfo=UTC)
INPUTS: dict[str, Any] = {
    "transcript": {"language": "en", "text": "Raise your prices. Clients pay for results."},
    "ocr": {"text": "3 PRICING MISTAKES"},
    "frame_descriptions": {
        "model": "gemma3:4b",
        "frames": [
            {"t": 2.0, "description": "A man talks to the camera in an office.",
             "shot_type": "person talking to camera"},
            {"t": 9.0, "error": "stuck"},
        ],
        "skipped": [],
    },
}  # fmt: skip


def reply(category: str = "business", **extra: object) -> str:
    answer = {
        "category": category,
        "confidence": 0.85,
        "tags": ["Pricing", "freelancing", "pricing", "Business"],
        "summary": " Three pricing mistakes freelancers make. ",
        "reason": "Advice about pricing work.",
    }
    return json.dumps(answer | extra)


@pytest.fixture
def env(tmp_path: Path) -> tuple[WorkerEnv, FakeLLM]:
    """Video 1: to categorize, all inputs computed. Video 2: the user moved it from edit to
    business. Video 3: the user put it in comedy, no correction row."""
    fake = FakeLLM()
    paths = DataPaths(tmp_path / "data")
    engine = db.open_database(paths)
    with db.session_scope(engine) as session:
        categories.ensure_default_categories(session)
        by_name = {c.name: c.id or 0 for c in categories.all_categories(session)}
        source = Source(type=SourceType.MANUAL, display_name="Video files")
        session.add(source)
        session.flush()
        assert source.id is not None
        session.add(Video(id=1, source_id=source.id, file_path="C:/v/1.mp4", author="@coach"))
        session.add(Video(id=2, source_id=source.id, caption="How I priced my first gig"))
        session.add(Video(id=3, source_id=source.id, caption="when the wifi drops"))
        session.flush()
        session.add(Profile(video_id=1, duration_s=30, speech_ratio=0.9, ocr_text="quick text"))
        models = db.load_settings(session).models()
        for name, data in INPUTS.items():
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
        session.add_all(
            [
                CategoryAssignment(
                    video_id=2, category_id=by_name["edit"], assigned_by=AssignedBy.AUTO,
                    created_at=T0,
                ),
                CategoryAssignment(
                    video_id=2, category_id=by_name["business"], assigned_by=AssignedBy.USER,
                    created_at=T0 + timedelta(minutes=1),
                ),
                Correction(
                    video_id=2, from_category_id=by_name["edit"],
                    to_category_id=by_name["business"],
                ),
                CategoryAssignment(
                    video_id=3, category_id=by_name["comedy"], assigned_by=AssignedBy.USER,
                    created_at=T0 + timedelta(minutes=2),
                ),
            ]
        )  # fmt: skip
    return WorkerEnv(engine, paths, provider_factory=lambda: fake), fake


def run_categorize(env: WorkerEnv, video_ids: list[int]) -> Any:
    builtin.enqueue_analyzer(env, "categorize", video_ids)
    return Worker(env, builtin.HANDLERS).run()


def test_reads_everything_and_learns_from_the_users_choices(
    env: tuple[WorkerEnv, FakeLLM],
) -> None:
    worker_env, fake = env
    fake.replies = [reply()]
    report = run_categorize(worker_env, [1])
    assert len(report.done) == 1

    [call] = fake.calls
    assert call.model == "gemma3:4b"
    prompt = call.messages[1].content
    assert "<data>Raise your prices. Clients pay for results.</data>" in prompt
    assert "On-screen text: <data>3 PRICING MISTAKES</data>" in prompt
    assert "quick text" not in prompt  # the full OCR replaces the quick check
    assert "- 2 s [person talking to camera] A man talks to the camera" in prompt
    assert "stuck" not in prompt
    assert "Author: @coach" in prompt
    # The user's choices, newest first, with what the app had guessed before.
    assert prompt.index("when the wifi drops") < prompt.index("How I priced my first gig")
    assert "User's category: business (the app had guessed edit; the user changed it)" in prompt
    assert "User's category: comedy\n" in prompt
    assert "- business: Advice about work" in prompt
    schema = call.json_schema
    assert schema is not None and "travel" in json.dumps(schema)

    with Session(worker_env.engine) as session:
        result = session.exec(select(Result).where(Result.analyzer == "categorize")).one()
        category, by = library.current_category(session, 1)
    assert result.value == {
        "category": "business",
        "tags": ["pricing", "freelancing"],
        "summary": "Three pricing mistakes freelancers make.",
        "prompt_version": categorize.PROMPT_VERSION,
    }
    assert result.confidence == 0.85 and result.evidence == "Advice about pricing work."
    assert result.method == (
        "transcript, on-screen text, frame descriptions read by gemma3:4b, "
        "with 2 of your own category choices as examples"
    )
    assert category is not None and category.name == "business" and by is AssignedBy.AUTO


def test_the_users_choice_still_wins(env: tuple[WorkerEnv, FakeLLM]) -> None:
    worker_env, fake = env
    with db.session_scope(worker_env.engine) as session:
        comedy = session.exec(select(Category).where(Category.name == "comedy")).one()
        assert comedy.id is not None
        session.add(
            CategoryAssignment(video_id=1, category_id=comedy.id, assigned_by=AssignedBy.USER)
        )
    fake.replies = [reply("business")]
    run_categorize(worker_env, [1])
    with Session(worker_env.engine) as session:
        category, by = library.current_category(session, 1)
    assert category is not None and category.name == "comedy" and by is AssignedBy.USER
    # Its own example is never shown to the model.
    assert "User's category: comedy" in fake.calls[0].messages[1].content
    assert fake.calls[0].messages[1].content.count("User's category") == 2


def test_rerun_supersedes_the_old_result(env: tuple[WorkerEnv, FakeLLM]) -> None:
    worker_env, fake = env
    fake.replies = [reply("business"), reply("education")]
    run_categorize(worker_env, [1])
    run_categorize(worker_env, [1])
    with Session(worker_env.engine) as session:
        rows = session.exec(
            select(Result).where(Result.analyzer == "categorize").order_by(col(Result.id))
        ).all()
        category, _by = library.current_category(session, 1)
    assert [(r.value["category"], r.superseded) for r in rows] == [
        ("business", True),
        ("education", False),
    ]
    assert category is not None and category.name == "education"


def test_unknown_category_is_retried_then_fails(env: tuple[WorkerEnv, FakeLLM]) -> None:
    worker_env, fake = env
    fake.replies = [reply("cats"), reply("dogs")]
    report = run_categorize(worker_env, [1])
    assert len(fake.calls) == 2 and len(report.failed) == 1
    with Session(worker_env.engine) as session:
        assert session.exec(select(Result).where(Result.analyzer == "categorize")).all() == []


def test_without_speech_or_frames_it_says_what_it_used(env: tuple[WorkerEnv, FakeLLM]) -> None:
    inputs = {
        "transcript": {"language": None, "text": ""},
        "ocr": {"text": ""},
        "frame_descriptions": {"frames": [], "skipped": []},
    }
    facts = categorize.video_facts(Video(id=9, source_id=1), None, inputs)
    assert "What is said: nothing (no speech)" in facts
    assert "On-screen text: none" in facts and "Frames:" not in facts
    assert categorize._method(inputs, "m", 0) == "metadata and profile read by m"


def test_plan_queues_inputs_then_categorize(env: tuple[WorkerEnv, FakeLLM]) -> None:
    worker_env, _fake = env
    with db.session_scope(worker_env.engine) as session:
        session.get_one(Video, 2).file_path = "C:/v/2.mp4"
    plan = planning.build_plan(worker_env.engine, ["categorize"], video_ids=[1, 2])
    assert plan.not_available == []
    assert {v.video_id: v.missing for v in plan.videos} == {
        1: [],
        2: ["frames", "transcript", "ocr", "frame_descriptions"],
    }
    assert [j.name for j in plan.jobs][-1] == "categorize"
    ids = planning.queue_plan(worker_env, plan)
    with Session(worker_env.engine) as session:
        jobs = {j.id: j for j in session.exec(select(Job).where(col(Job.id).in_(ids)))}
        cat_jobs = [j for j in jobs.values() if j.name == "categorize"]
        deps = {
            j.video_id: {
                jobs[d.depends_on_id].name
                for d in session.exec(select(JobDependency).where(JobDependency.job_id == j.id))
            }
            for j in cat_jobs
        }
    assert deps == {1: set(), 2: {"frames", "transcript", "ocr", "frame_descriptions"}}
