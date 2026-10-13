from pathlib import Path

import cv2
import numpy as np
import pytest
from sqlmodel import Session
from typer.testing import CliRunner

from clipsdex_engine import artifacts, config, db
from clipsdex_engine.artifacts import frames
from clipsdex_engine.cli import app
from clipsdex_engine.config import DataPaths
from clipsdex_engine.jobs import builtin
from clipsdex_engine.jobs.handlers import WorkerEnv
from clipsdex_engine.jobs.worker import Worker
from clipsdex_engine.sources import manual
from synthetic import SyntheticMedia, make_text_video, needs_ffmpeg


def solid(value: int, n: int) -> list[np.ndarray]:
    return [np.full((60, 40, 3), value, np.uint8) for _ in range(n)]


def pattern(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 255, (60, 40, 3), dtype=np.uint8)


def test_phash_same_similar_and_different() -> None:
    a = cv2.resize(pattern(1), (200, 300), interpolation=cv2.INTER_NEAREST)
    noisy = np.clip(a.astype(int) + np.random.default_rng(2).integers(-8, 8, a.shape), 0, 255)
    b = cv2.resize(pattern(3), (200, 300), interpolation=cv2.INTER_NEAREST)
    assert frames.hamming(frames.phash(a), frames.phash(a)) == 0
    assert frames.hamming(frames.phash(a), frames.phash(noisy.astype(np.uint8))) <= 10
    assert frames.hamming(frames.phash(a), frames.phash(b)) > frames.DUPLICATE_DISTANCE


def test_scan_finds_scene_middles() -> None:
    fps = 10.0
    # 3 shots of 2 s each, differing in brightness (a real cut for ContentDetector).
    shots = solid(20, 20) + solid(240, 20) + solid(110, 20)
    candidates = frames.scan(shots, fps=fps)
    assert [c.scene for c in candidates] == [0, 1, 2]
    assert [c.t for c in candidates] == pytest.approx([1.0, 3.0, 5.0], abs=0.15)


def test_long_scene_gets_extra_candidates() -> None:
    candidates = frames.scan(solid(90, 350), fps=10.0)  # one 35 s shot
    assert len(candidates) >= 4  # middle + one every 10 s
    assert {c.scene for c in candidates} == {0}


def test_select_dedupes_then_spreads_over_budget() -> None:
    def cand(t: float, seed: int) -> frames.Candidate:
        return frames.Candidate(t, int(t), frames.thumbnail(pattern(seed)))

    same = [cand(t, 7) for t in (1.0, 2.0, 3.0)]  # one picture three times
    assert [c.t for c in frames.select(same, budget=8)] == [1.0]

    many = [cand(float(t), t) for t in range(10)]
    picked = frames.select(many, budget=4)
    assert len(picked) == 4
    assert picked[0].t == 0.0 and picked[-1].t == 9.0  # covers start and end


@pytest.fixture
def env(tmp_path: Path) -> WorkerEnv:
    paths = DataPaths(tmp_path / "Подаци")
    return WorkerEnv(db.open_database(paths), paths)


def frames_data(env: WorkerEnv, video_id: int, budget: int = 4) -> dict[str, object]:
    with Session(env.engine) as session:
        artifact = artifacts.find(session, video_id, frames.TYPE, frames.params_for(budget))
        assert artifact is not None and isinstance(artifact.data, dict)
        return artifact.data


@needs_ffmpeg
def test_three_shots_give_three_frames(env: WorkerEnv, synthetic_media: SyntheticMedia) -> None:
    manual.import_paths(env.engine, [synthetic_media.three_shots])
    [job_id] = builtin.enqueue_frames(env)
    report = Worker(env, builtin.HANDLERS).run()
    assert report.done == [job_id]

    data = frames_data(env, 1)
    saved = data["frames"]
    assert isinstance(saved, list) and len(saved) == 3 and data["scenes"] == 3
    for frame in saved:
        jpg = env.paths.root / frame["path"]
        image = cv2.imdecode(np.frombuffer(jpg.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        assert image is not None and image.shape[1] == 320  # never upscaled past the source
    assert builtin.enqueue_frames(env) == []


@needs_ffmpeg
def test_static_talking_head_needs_one_frame(env: WorkerEnv, tmp_path: Path) -> None:
    video = make_text_video(tmp_path / "head", ["HOW TO PRICE", "YOUR WORK"], seconds=35)
    manual.import_paths(env.engine, [video])
    builtin.enqueue_frames(env)
    Worker(env, builtin.HANDLERS).run()
    data = frames_data(env, 1)
    assert data["candidates"] >= 4  # type: ignore[operator]
    assert len(data["frames"]) == 1  # type: ignore[arg-type]


@needs_ffmpeg
def test_redo_replaces_files_and_budget_is_respected(
    env: WorkerEnv, synthetic_media: SyntheticMedia
) -> None:
    manual.import_paths(env.engine, [synthetic_media.three_shots])
    frames.build_frames(env.engine, env.paths, 1, budget=4)
    data = frames.build_frames(env.engine, env.paths, 1, budget=4)
    folder = env.paths.root / Path(data["frames"][0]["path"]).parent
    assert len(list(folder.glob("*.jpg"))) == 3

    two = frames.build_frames(env.engine, env.paths, 1, budget=2)
    assert [round(f["t"]) for f in two["frames"]] == [
        round(data["frames"][0]["t"]),
        round(data["frames"][-1]["t"]),
    ]


@needs_ffmpeg
def test_missing_file_fails_clearly(env: WorkerEnv, tmp_path: Path) -> None:
    gone = tmp_path / "gone.mp4"
    gone.write_bytes(b"x")
    manual.import_paths(env.engine, [gone])
    gone.unlink()
    builtin.enqueue_frames(env)
    report = Worker(env, builtin.HANDLERS).run()
    assert "deleted or moved" in report.failed[0][1]


@needs_ffmpeg
def test_cli_frames_and_show(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, synthetic_media: SyntheticMedia
) -> None:
    monkeypatch.setenv(config.DATA_DIR_ENV, str(tmp_path / "data"))
    runner = CliRunner()
    runner.invoke(app, ["import", str(synthetic_media.three_shots)])
    assert "Queued 1 frame job(s)" in runner.invoke(app, ["frames"]).output
    assert runner.invoke(app, ["worker"]).exit_code == 0
    shown = runner.invoke(app, ["show", "1"]).output
    assert "Key frames: 3 of 3 candidates, 3 scene(s), budget 4" in shown
