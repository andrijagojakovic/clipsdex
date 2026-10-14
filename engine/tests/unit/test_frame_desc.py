import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from sqlmodel import Session
from typer.testing import CliRunner

from clipsdex_engine import artifacts, config, db, llm
from clipsdex_engine.artifacts import frame_desc, frames
from clipsdex_engine.cli import app
from clipsdex_engine.config import DataPaths
from clipsdex_engine.db.models import Artifact, Job, JobStatus, Source, SourceType, Video
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


def frames_t(env: WorkerEnv, video_id: int = 1) -> list[float]:
    with Session(env.engine) as session:
        found = artifacts.find(session, video_id, "frames", frames.params_for(4))
        assert found is not None and isinstance(found.data, dict)
        return [f["t"] for f in found.data["frames"]]


def jpeg(image: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", image)
    assert ok
    return buf.tobytes()


def test_plain_frames_are_recognized() -> None:
    rng = np.random.default_rng(0)
    slide = np.full((1280, 720, 3), 30, np.uint8)
    cv2.putText(slide, "NOW PLAYING", (40, 600), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (255,) * 3, 5)
    cv2.putText(slide, "Midnight City", (40, 720), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (255,) * 3, 5)
    assert frame_desc.is_plain(jpeg(slide))
    assert frame_desc.is_plain(jpeg(np.zeros((360, 640, 3), np.uint8)))
    scene = rng.integers(0, 255, (360, 640, 3), dtype=np.uint8)
    assert not frame_desc.is_plain(jpeg(scene))
    dark = np.tile(np.linspace(0, 90, 640, dtype=np.uint8)[None, :, None], (360, 1, 3))
    assert not frame_desc.is_plain(jpeg(dark))  # a dark scene still has something to see


def test_at_most_a_few_frames_are_described() -> None:
    assert [frame_desc.max_described(b) for b in (1, 2, 4, 8)] == [1, 2, 2, 4]
    assert frame_desc.spread(4, 2) == [1, 3]
    assert frame_desc.spread(8, 4) == [1, 3, 5, 7]
    assert frame_desc.spread(1, 2) == [0]


def test_budget_limits_vision_calls(tmp_path: Path) -> None:
    fake = FakeLLM(default_reply=answer("A busy street.", "nature or travel"))
    env = make_env(tmp_path, fake)
    folder = env.paths.artifacts / "1" / "frames-test"
    folder.mkdir(parents=True)
    rng = np.random.default_rng(1)
    entries = []
    for n in range(4):
        path = folder / f"frame_{n}.jpg"
        path.write_bytes(jpeg(rng.integers(0, 255, (90, 160, 3), dtype=np.uint8)))
        entries.append({"t": float(n), "path": artifacts.to_stored_path(env.paths, path)})
    with db.session_scope(env.engine) as session:
        source = Source(type=SourceType.MANUAL, display_name="Video files")
        session.add(source)
        session.flush()
        assert source.id is not None
        session.add(Video(id=1, source_id=source.id, file_path="C:/v/1.mp4"))
        session.flush()
        params = frames.params_for(4)
        session.add(
            Artifact(
                video_id=1,
                type="frames",
                params_hash=artifacts.params_hash(params),
                params=params,
                data={"frames": entries},
            )
        )
    data = frame_desc.build_descriptions(env.engine, env.paths, 1, 4, fake, MODEL)
    assert [f["t"] for f in data["frames"]] == [1.0, 3.0]
    assert len(fake.calls) == 2 and data["skipped"] == []


def test_one_vision_call_per_frame(tmp_path: Path, synthetic_media: SyntheticMedia) -> None:
    fake = FakeLLM(
        replies=[
            answer("A colorful test pattern with numbers.", "screen recording"),
            answer("A dark blue gradient.", "other"),
        ]
    )
    env = make_env(tmp_path, fake)
    manual.import_paths(env.engine, [synthetic_media.three_shots])

    ids = builtin.enqueue_frame_descriptions(env)
    report = Worker(env, builtin.HANDLERS).run()

    assert len(ids) == 2 and len(report.done) == 2  # frames, then descriptions
    assert len(fake.calls) == 2  # the plain red shot is skipped
    for call in fake.calls:
        [message] = call.messages
        assert len(message.images) == 1 and message.images[0][:2] == b"\xff\xd8"  # one JPEG
        assert "Do not name, identify or guess who any person is" in message.content
        assert call.max_tokens == frame_desc.MAX_TOKENS
    data = descriptions(env)
    frames = data["frames"]
    assert isinstance(frames, list)
    assert [f["shot_type"] for f in frames] == ["screen recording", "other"]
    assert data["skipped"] == [{"t": frames_t(env)[0], "reason": frame_desc.PLAIN_REASON}]
    assert data["model"] == MODEL
    assert fake.events == [f"load {MODEL}", f"unload {MODEL}"]


def test_one_bad_frame_does_not_lose_the_others(
    tmp_path: Path, synthetic_media: SyntheticMedia
) -> None:
    stuck = llm.LLMOutputError("The model got stuck repeating itself.")
    fake = FakeLLM(replies=[answer("A test pattern."), stuck, stuck])
    env = make_env(tmp_path, fake)
    manual.import_paths(env.engine, [synthetic_media.three_shots])
    builtin.enqueue_frame_descriptions(env)

    assert len(Worker(env, builtin.HANDLERS).run().done) == 2
    frames = descriptions(env)["frames"]
    assert isinstance(frames, list)
    assert [("error" in f) for f in frames] == [False, True]
    assert fake.calls[2].temperature > 0  # the retry varies the answer a little


def test_job_fails_if_no_frame_can_be_described(
    tmp_path: Path, synthetic_media: SyntheticMedia
) -> None:
    env = make_env(tmp_path, FakeLLM(default_reply="not json"))
    manual.import_paths(env.engine, [synthetic_media.with_audio])
    builtin.enqueue_frame_descriptions(env)
    report = Worker(env, builtin.HANDLERS).run()
    assert "did not return a usable answer" in report.failed[0][1]


def test_ollama_down_is_retried_later(tmp_path: Path, synthetic_media: SyntheticMedia) -> None:
    env = make_env(tmp_path, FakeLLM(running=False))
    manual.import_paths(env.engine, [synthetic_media.with_audio])
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
    fake = FakeLLM(default_reply=answer("A colorful test pattern.", "screen recording"))
    monkeypatch.setattr(llm, "default_provider", lambda *_a, **_k: fake)
    runner = CliRunner()
    runner.invoke(app, ["import", str(synthetic_media.with_audio)])
    runner.invoke(app, ["import", str(synthetic_media.silent)])
    assert "Queued 4 job(s)" in runner.invoke(app, ["describe"]).output
    assert runner.invoke(app, ["worker"]).exit_code == 0
    shown = runner.invoke(app, ["show", "1"]).output
    assert "Frame descriptions (gemma3:4b):" in shown
    assert "[screen recording] A colorful test pattern." in shown
    assert "skipped: plain frame" in runner.invoke(app, ["show", "2"]).output
