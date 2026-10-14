import json
from pathlib import Path

import pytest
from sqlmodel import Session
from typer.testing import CliRunner

from clipsdex_engine import artifacts, config, db, llm
from clipsdex_engine.artifacts import frame_desc
from clipsdex_engine.cli import app
from clipsdex_engine.config import DataPaths
from clipsdex_engine.db.models import Job, JobStatus
from clipsdex_engine.jobs import builtin, queue
from clipsdex_engine.jobs.handlers import WorkerEnv
from clipsdex_engine.jobs.worker import Worker
from clipsdex_engine.sources import manual
from mocks.fake_llm import FakeLLM
from synthetic import SyntheticMedia, needs_ffmpeg

pytestmark = needs_ffmpeg
MODEL = "gemma3:4b"


def answer(description: str, shot: str = "film or TV scene") -> str:
    return json.dumps({"description": description, "shot_type": shot})


def make_env(tmp_path: Path, fake: FakeLLM) -> WorkerEnv:
    paths = DataPaths(tmp_path / "Подаци")
    return WorkerEnv(db.open_database(paths), paths, provider_factory=lambda: fake)


def descriptions(env: WorkerEnv, video_id: int = 1) -> dict[str, object]:
    with Session(env.engine) as session:
        artifact = artifacts.find(
            session, video_id, frame_desc.TYPE, frame_desc.params_for(MODEL, 4)
        )
        assert artifact is not None and isinstance(artifact.data, dict)
        return artifact.data


def test_one_vision_call_per_frame(tmp_path: Path, synthetic_media: SyntheticMedia) -> None:
    fake = FakeLLM(
        replies=[
            answer("A plain red screen.", "other"),
            answer("A colorful test pattern with numbers.", "screen recording"),
            answer("A dark blue gradient.", "other"),
        ]
    )
    env = make_env(tmp_path, fake)
    manual.import_paths(env.engine, [synthetic_media.three_shots])

    ids = builtin.enqueue_frame_descriptions(env)
    report = Worker(env, builtin.HANDLERS).run()

    assert len(ids) == 2 and len(report.done) == 2  # frames, then descriptions
    assert len(fake.calls) == 3
    for call in fake.calls:
        [message] = call.messages
        assert len(message.images) == 1 and message.images[0][:2] == b"\xff\xd8"  # one JPEG
        assert "Do not name, identify or guess who any person is" in message.content
        assert call.max_tokens == frame_desc.MAX_TOKENS
    data = descriptions(env)
    frames = data["frames"]
    assert isinstance(frames, list)
    assert [f["shot_type"] for f in frames] == ["other", "screen recording", "other"]
    assert data["model"] == MODEL
    assert fake.events == [f"load {MODEL}", f"unload {MODEL}"]


def test_one_bad_frame_does_not_lose_the_others(
    tmp_path: Path, synthetic_media: SyntheticMedia
) -> None:
    stuck = llm.LLMOutputError("The model got stuck repeating itself.")
    fake = FakeLLM(replies=[answer("Red."), stuck, stuck, answer("Blue.")])
    env = make_env(tmp_path, fake)
    manual.import_paths(env.engine, [synthetic_media.three_shots])
    builtin.enqueue_frame_descriptions(env)

    assert len(Worker(env, builtin.HANDLERS).run().done) == 2
    frames = descriptions(env)["frames"]
    assert isinstance(frames, list)
    assert [("error" in f) for f in frames] == [False, True, False]
    assert fake.calls[2].temperature > 0  # the retry varies the answer a little


def test_job_fails_if_no_frame_can_be_described(
    tmp_path: Path, synthetic_media: SyntheticMedia
) -> None:
    env = make_env(tmp_path, FakeLLM(default_reply="not json"))
    manual.import_paths(env.engine, [synthetic_media.silent])
    builtin.enqueue_frame_descriptions(env)
    report = Worker(env, builtin.HANDLERS).run()
    assert "did not return a usable answer" in report.failed[0][1]


def test_ollama_down_is_retried_later(tmp_path: Path, synthetic_media: SyntheticMedia) -> None:
    env = make_env(tmp_path, FakeLLM(running=False))
    manual.import_paths(env.engine, [synthetic_media.silent])
    ids = builtin.enqueue_frame_descriptions(env)
    report = Worker(env, builtin.HANDLERS).run()
    assert len(report.retried) == queue.MAX_ATTEMPTS - 1
    with Session(env.engine) as session:
        assert session.get_one(Job, ids[-1]).status is JobStatus.FAILED


def test_shares_the_model_load_with_category_guesses(tmp_path: Path) -> None:
    key = builtin.FRAME_DESCRIPTIONS.resource_key(1, frame_desc.params_for(MODEL, 4))
    assert key == builtin.CATEGORY_GUESS.resource_key(1, {"model": MODEL})


def test_cli_describe_and_show(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, synthetic_media: SyntheticMedia
) -> None:
    monkeypatch.setenv(config.DATA_DIR_ENV, str(tmp_path / "data"))
    fake = FakeLLM(default_reply=answer("A plain blue screen.", "other"))
    monkeypatch.setattr(llm, "default_provider", lambda *_a, **_k: fake)
    runner = CliRunner()
    runner.invoke(app, ["import", str(synthetic_media.silent)])
    assert "Queued 2 job(s)" in runner.invoke(app, ["describe"]).output
    assert runner.invoke(app, ["worker"]).exit_code == 0
    shown = runner.invoke(app, ["show", "1"]).output
    assert "Frame descriptions (gemma3:4b):" in shown
    assert "[other] A plain blue screen." in shown
