from pathlib import Path
from typing import Any, ClassVar

import pytest
from sqlmodel import Session, col, select
from typer.testing import CliRunner

from clipsdex_engine import analyzers, artifacts, config, db
from clipsdex_engine.analyzers import Analyzer, AnalyzerContext, AnalyzerResult, Applicability
from clipsdex_engine.analyzers.brief import Brief
from clipsdex_engine.analyzers.song_id import SongId
from clipsdex_engine.artifacts import transcript
from clipsdex_engine.cli import app
from clipsdex_engine.config import DataPaths
from clipsdex_engine.db.models import (
    Artifact,
    AssignedBy,
    Category,
    CategoryAssignment,
    Job,
    JobDependency,
    JobStatus,
    Profile,
    Result,
    Source,
    SourceType,
    Video,
)
from clipsdex_engine.jobs import builtin, queue
from clipsdex_engine.jobs import plan as planning
from clipsdex_engine.jobs.handlers import WorkerEnv, spec_for
from clipsdex_engine.jobs.worker import Worker

VIDEO = Video(id=1, source_id=1)


def test_registry_holds_the_four_analyzers() -> None:
    assert set(analyzers.REGISTRY) == {"categorize", "brief", "movie_id", "song_id"}
    assert analyzers.get("song_id").default_categories == ("edit", "music")
    with pytest.raises(analyzers.UnknownAnalyzerError, match="Known: categorize"):
        analyzers.get("teleport")
    with pytest.raises(ValueError, match="registered twice"):
        analyzers.register(SongId())


@pytest.mark.parametrize(
    ("profile", "applies", "reason"),
    [
        (None, True, "not profiled yet"),
        (Profile(video_id=1, has_audio=False), False, "no sound"),
        (Profile(video_id=1, has_audio=True, music_ratio=0.04), False, "almost no music (4%"),
        (Profile(video_id=1, has_audio=True, music_ratio=0.8), True, "music in 80%"),
    ],
)
def test_song_id_needs_music(profile: Profile | None, applies: bool, reason: str) -> None:
    check = SongId().is_applicable(VIDEO, profile)
    assert check.applies is applies and reason in check.reason


@pytest.mark.parametrize(
    ("profile", "applies", "reason"),
    [
        (Profile(video_id=1, speech_ratio=0.6), True, "speech in 60%"),
        (Profile(video_id=1, speech_ratio=0.0, has_text=True), True, "on-screen text"),
        (Profile(video_id=1, speech_ratio=0.05, has_text=False), False, "no speech"),
    ],
)
def test_brief_needs_words(profile: Profile, applies: bool, reason: str) -> None:
    check = Brief().is_applicable(VIDEO, profile)
    assert check.applies is applies and reason in check.reason


def test_unavailable_analyzer_says_so() -> None:
    ctx: Any = None
    with pytest.raises(analyzers.AnalyzerNotReadyError, match="Find songs is not available"):
        SongId().run(ctx)


@pytest.fixture
def env(tmp_path: Path) -> WorkerEnv:
    """1 edit with music, 2 edit without music, 3 business talk, 4 business, file gone."""
    paths = DataPaths(tmp_path / "data")
    engine = db.open_database(paths)
    with db.session_scope(engine) as session:
        source = Source(type=SourceType.MANUAL, display_name="Video files")
        session.add(source)
        session.flush()
        assert source.id is not None
        for vid in (1, 2, 3):
            session.add(Video(id=vid, source_id=source.id, file_path=f"C:/v/{vid}.mp4"))
        session.add(Video(id=4, source_id=source.id, file_path=None, file_deleted=True))
        session.add_all([Category(id=1, name="edit"), Category(id=2, name="business")])
        session.flush()
        rows = ((1, 1, 0.9, 0.0), (2, 1, 0.0, 0.0), (3, 2, 0.0, 0.8), (4, 2, 0.0, 0.8))
        for vid, cat, music, speech in rows:
            session.add(
                Profile(
                    video_id=vid,
                    duration_s=20.0,
                    has_audio=True,
                    music_ratio=music,
                    speech_ratio=speech,
                )
            )
            session.add(
                CategoryAssignment(video_id=vid, category_id=cat, assigned_by=AssignedBy.AUTO)
            )
    return WorkerEnv(engine, paths)


def test_plan_uses_usual_categories_and_skips_with_a_reason(env: WorkerEnv) -> None:
    plan = planning.build_plan(env.engine, ["song_id"])
    by_id = {v.video_id: v for v in plan.videos}
    assert set(by_id) == {1, 2}  # business videos are not looked at
    assert by_id[1].analyzers == ["song_id"] and by_id[1].missing == []
    assert by_id[2].analyzers == []
    assert "almost no music" in by_id[2].skipped["song_id"]
    assert plan.not_available == ["song_id"]
    assert plan.jobs == []  # nothing it reads is missing, and it can't run yet


def test_a_named_category_or_video_overrides_the_usual_categories(env: WorkerEnv) -> None:
    plan = planning.build_plan(env.engine, ["song_id"], category="business")
    assert {v.video_id: v.skipped.get("song_id", "") for v in plan.videos} == {
        3: "almost no music (0% of the audio)",
        4: "almost no music (0% of the audio)",
    }
    # Picked by ID: the user wants it anyway, so the rule is not applied.
    forced = planning.build_plan(env.engine, ["song_id"], video_ids=[3])
    assert forced.videos[0].analyzers == ["song_id"] and forced.videos[0].skipped == {}


def test_brief_plans_its_inputs_per_applicable_video(env: WorkerEnv) -> None:
    plan = planning.build_plan(env.engine, ["brief", "song_id"], category="business")
    by_id = {v.video_id: v for v in plan.videos}
    assert by_id[3].analyzers == ["brief"]
    assert by_id[3].missing == ["frames", "transcript", "ocr"]
    assert by_id[4].blocked is not None  # file gone, frames can't be taken
    assert [j.name for j in plan.jobs] == ["frames", "transcript", "ocr", "brief"]


class EchoTranscript(Analyzer):
    """A test analyzer: returns the transcript text it was given."""

    name: ClassVar[str] = "echo"
    title: ClassVar[str] = "Echo"
    requires: ClassVar[tuple[str, ...]] = ("transcript",)

    def __init__(self) -> None:
        self.seen: list[AnalyzerContext] = []

    def is_applicable(self, video: Video, profile: Profile | None) -> Applicability:
        return Applicability(True, "always")

    def run(self, ctx: AnalyzerContext) -> AnalyzerResult:
        self.seen.append(ctx)
        ctx.progress(0.5)
        text = ctx.artifacts["transcript"]["text"]
        return AnalyzerResult({"text": text}, 0.9, "transcript", evidence=text[:10])


def add_transcript(env: WorkerEnv, video_id: int, text: str) -> None:
    with db.session_scope(env.engine) as session:
        params = transcript.params_for(db.load_settings(session).models().whisper)
        session.add(
            Artifact(
                video_id=video_id,
                type=transcript.TYPE,
                params_hash=artifacts.params_hash(params),
                params=params,
                data={"text": text},
            )
        )


def test_analyzer_job_reads_artifacts_and_stores_a_result(env: WorkerEnv) -> None:
    echo = EchoTranscript()
    add_transcript(env, 3, "raise your prices")
    with db.session_scope(env.engine) as session:
        session.add(Result(video_id=3, analyzer="echo", value={}, method="old"))
    [job_id] = builtin.enqueue_analyzer(env, echo, [3])

    report = Worker(env, {"echo": builtin.AnalyzerHandler(echo)}).run()

    assert report.done == [job_id]
    assert echo.seen[0].video.id == 3 and echo.seen[0].profile is not None
    with Session(env.engine) as session:
        rows = session.exec(select(Result).where(Result.video_id == 3).order_by(col(Result.id)))
        old, new = rows.all()
    assert old.superseded and not new.superseded
    assert new.value == {"text": "raise your prices"}
    assert (new.confidence, new.method, new.evidence) == (0.9, "transcript", "raise your")


def test_analyzer_job_waits_for_queued_inputs(env: WorkerEnv) -> None:
    [t_job] = builtin.enqueue_transcripts(env, video_ids=[3])
    [a_job] = builtin.enqueue_analyzer(env, EchoTranscript(), [3])
    with Session(env.engine) as session:
        deps = session.exec(select(JobDependency).where(JobDependency.job_id == a_job)).all()
    assert [d.depends_on_id for d in deps] == [t_job]


def test_missing_input_or_unavailable_analyzer_fails_clearly(env: WorkerEnv) -> None:
    echo = EchoTranscript()
    [no_input] = builtin.enqueue_analyzer(env, echo, [1])
    report = Worker(env, {"echo": builtin.AnalyzerHandler(echo)}).run()
    assert (
        report.failed[0][0] == no_input
        and "transcript for this video is missing" in (report.failed[0][1])
    )

    song = analyzers.get("song_id")
    queue.enqueue_many(
        env.engine,
        [
            spec_for(
                builtin.HANDLERS["song_id"],
                1,
                {"models": config.PRESETS[config.Preset.BALANCED].model_dump()},
            )
        ],
    )
    report = Worker(env, builtin.HANDLERS).run()
    assert [reason for _id, reason in report.failed] == [
        f"{song.title} is not available in this version yet."
    ]
    with Session(env.engine) as session:
        assert session.exec(select(Job).where(Job.status == JobStatus.FAILED)).all()


def test_cli_plan_explains_skips(env: WorkerEnv, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(config.DATA_DIR_ENV, str(env.paths.root))
    out = CliRunner().invoke(app, ["plan", "--analyzer", "song_id"]).output
    assert "Plan: song_id for the usual categories (edit, music): 2 video(s)" in out
    assert "skips song_id: almost no music (0% of the audio)" in out
    assert "Find songs (song_id) is not available in this version yet" in out
