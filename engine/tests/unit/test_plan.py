from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlmodel import Session
from typer.testing import CliRunner

from clipsdex_engine import artifacts, config, db
from clipsdex_engine.artifacts import frames, ocr, transcript
from clipsdex_engine.cli import app
from clipsdex_engine.config import DataPaths
from clipsdex_engine.db.models import (
    Artifact,
    AssignedBy,
    Category,
    CategoryAssignment,
    Job,
    JobKind,
    JobStatus,
    Profile,
    Source,
    SourceType,
    Video,
)
from clipsdex_engine.jobs import estimates, queue
from clipsdex_engine.jobs import plan as planning
from clipsdex_engine.jobs.handlers import WorkerEnv

T0 = datetime(2026, 10, 1, tzinfo=UTC)


def add_artifact(session: Session, video_id: int, type_: str, params: dict, data: object) -> None:
    session.add(
        Artifact(
            video_id=video_id,
            type=type_,
            params_hash=artifacts.params_hash(params),
            params=params,
            data=data,
        )
    )


@pytest.fixture
def env(tmp_path: Path) -> WorkerEnv:
    """Videos: 1 business (60 s, transcript done), 2 business (30 s, file gone),
    3 edit (20 s, frames done, with 3 frames)."""
    paths = DataPaths(tmp_path / "data")
    engine = db.open_database(paths)
    with db.session_scope(engine) as session:
        source = Source(type=SourceType.MANUAL, display_name="Video files")
        session.add(source)
        session.flush()
        assert source.id is not None
        session.add(Video(id=1, source_id=source.id, file_path="C:/v/1.mp4"))
        session.add(Video(id=2, source_id=source.id, file_path=None, file_deleted=True))
        session.add(Video(id=3, source_id=source.id, file_path="C:/v/3.mp4"))
        session.add_all([Category(id=1, name="business"), Category(id=2, name="edit")])
        session.flush()
        for vid, cat, duration in ((1, 1, 60.0), (2, 1, 30.0), (3, 2, 20.0)):
            session.add(Profile(video_id=vid, duration_s=duration))
            session.add(
                CategoryAssignment(video_id=vid, category_id=cat, assigned_by=AssignedBy.AUTO)
            )
        add_artifact(session, 1, "transcript", transcript.params_for("small"), {"text": "hi"})
        add_artifact(
            session, 3, "frames", frames.params_for(4), {"frames": [{"t": 1}, {"t": 2}, {"t": 3}]}
        )
    return WorkerEnv(engine, paths)


def finished_job(
    session: Session, name: str, video_id: int, seconds: float, **params: object
) -> None:
    session.add(
        Job(
            kind=JobKind.ARTIFACT,
            name=name,
            video_id=video_id,
            params=dict(params),
            status=JobStatus.DONE,
            resource="whisper:small" if name == "transcript" else "cpu",
            started_at=T0,
            finished_at=T0 + timedelta(seconds=seconds),
        )
    )


def test_targets_expand_and_unknown_names_fail() -> None:
    assert planning.expand_targets(["brief"], ["frames"]) == ("transcript", "ocr", "frames")
    with pytest.raises(ValueError, match="Unknown analyzer"):
        planning.expand_targets(["teleport"], [])


def test_brief_for_business_lists_missing_and_blocked(env: WorkerEnv) -> None:
    plan = planning.build_plan(env.engine, ("transcript", "ocr"), category="business")
    by_id = {v.video_id: v for v in plan.videos}
    assert set(by_id) == {1, 2}
    assert by_id[1].missing == ["frames", "ocr"] and by_id[1].blocked is None
    assert by_id[2].blocked is not None and "file is gone" in by_id[2].blocked
    names = [j.name for j in plan.jobs]
    assert names == ["frames", "ocr"]  # nothing for the blocked video, dependencies first
    assert plan.total_seconds > 0


def test_default_estimates_use_duration_and_frame_counts(env: WorkerEnv) -> None:
    plan = planning.build_plan(env.engine, ("ocr", "transcript"), video_ids=[3])
    jobs = {j.name: j for j in plan.jobs}
    assert jobs["transcript"].seconds == pytest.approx(20 * estimates.DEFAULT_RATES["transcript"])
    assert jobs["ocr"].seconds == pytest.approx(3 * estimates.DEFAULT_RATES["ocr"])  # 3 frames
    assert all(j.measured_jobs == 0 for j in plan.jobs)
    assert (
        plan.load_seconds == estimates.LOAD_SECONDS["whisper:"] + estimates.LOAD_SECONDS["rapidocr"]
    )


def test_this_machines_history_wins_and_ignores_one_outlier(env: WorkerEnv) -> None:
    with db.session_scope(env.engine) as session:
        # 0.2, 0.2, 0.2 s per second of video, and one stuck job at 10 s per second.
        for seconds in (12.0, 12.0, 12.0, 600.0):
            finished_job(session, "transcript", 1, seconds)
    with Session(env.engine) as session:
        r = estimates.rate(session, "transcript", "whisper:small", 4)
    assert r.measured_jobs == 4
    assert r.seconds_per_unit == pytest.approx(0.2)  # the median ignores the stuck job
    plan = planning.build_plan(env.engine, ("transcript",), video_ids=[3])
    assert plan.jobs[0].measured_jobs == 4


def test_too_little_history_falls_back_to_defaults(env: WorkerEnv) -> None:
    with db.session_scope(env.engine) as session:
        finished_job(session, "transcript", 1, 1.0)
    with Session(env.engine) as session:
        assert estimates.rate(session, "transcript", "whisper:small", 4).measured_jobs == 0


def test_queue_plan_queues_missing_work_once(env: WorkerEnv) -> None:
    plan = planning.build_plan(env.engine, ("transcript", "ocr"), category="business")
    ids = planning.queue_plan(env, plan)
    with Session(env.engine) as session:
        names = sorted(session.get_one(Job, i).name for i in ids)
    assert names == ["frames", "ocr"]
    assert planning.queue_plan(env, plan) == ids  # same jobs, not new ones
    assert queue.counts(env.engine)[JobStatus.QUEUED] == 2


def test_humanize() -> None:
    assert estimates.humanize(0.2) == "~1 s"
    assert estimates.humanize(90) == "~2 min"
    assert estimates.humanize(3 * 3600 + 20 * 60) == "~3 h 20 min"
    assert estimates.humanize(7200) == "~2 h"


def test_cli_plan(env: WorkerEnv, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(config.DATA_DIR_ENV, str(env.paths.root))
    runner = CliRunner()
    out = runner.invoke(app, ["plan", "--analyzer", "brief", "--category", "business"]).output
    assert 'Plan: brief for category "business": 2 video(s)' in out
    assert "needs frames, ocr" in out
    assert "can't run: the video file is gone" in out
    assert "default estimate" in out and "Total: ~" in out

    queued = runner.invoke(
        app, ["plan", "--analyzer", "brief", "--category", "business", "--queue"]
    )
    assert "Queued 2 job(s)" in queued.output
    assert runner.invoke(app, ["plan"]).exit_code == 1
    assert runner.invoke(app, ["plan", "--analyzer", "nope"]).exit_code == 1


def test_ready_when_everything_exists(env: WorkerEnv) -> None:
    with db.session_scope(env.engine) as session:
        add_artifact(session, 1, "frames", frames.params_for(4), {"frames": [{"t": 1}]})
        add_artifact(session, 1, "ocr", ocr.params_for(4), {"text": ""})
    plan = planning.build_plan(env.engine, ("transcript", "ocr"), video_ids=[1])
    assert plan.videos[0].missing == [] and plan.jobs == []
