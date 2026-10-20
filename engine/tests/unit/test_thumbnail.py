from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

from clipsdex_engine import artifacts, db
from clipsdex_engine.api.app import create_app
from clipsdex_engine.artifacts import thumbnail
from clipsdex_engine.config import DataPaths
from clipsdex_engine.db.models import Artifact, Source, SourceType, Video
from clipsdex_engine.sources import manual
from mocks.fake_llm import FakeLLM
from synthetic import SyntheticMedia, needs_ffmpeg

TOKEN = "z" * 40
AUTH = {"Authorization": f"Bearer {TOKEN}"}


def size_of(path: Path) -> tuple[int, int]:
    image = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    assert image is not None
    return image.shape[1], image.shape[0]


@needs_ffmpeg
def test_made_from_the_video_once(tmp_path: Path, synthetic_media: SyntheticMedia) -> None:
    paths = DataPaths(tmp_path / "Подаци")
    engine = db.open_database(paths)
    manual.import_paths(engine, [synthetic_media.with_audio])  # 320x240, no profile yet
    first = thumbnail.thumbnail(engine, paths, 1)
    assert first is not None and first.name == thumbnail.FILE_NAME
    assert size_of(first) == (320, 240)  # never upscaled
    stamp = first.stat().st_mtime_ns
    assert thumbnail.thumbnail(engine, paths, 1) == first
    assert first.stat().st_mtime_ns == stamp  # kept, not made again


def test_made_from_the_first_key_frame(tmp_path: Path) -> None:
    paths = DataPaths(tmp_path / "data")
    engine = db.open_database(paths)
    folder = artifacts.artifact_dir(paths, 1) / "frames-test"
    folder.mkdir(parents=True)
    ok, buf = cv2.imencode(".jpg", np.full((1280, 720, 3), 200, np.uint8))
    assert ok
    (folder / "frame_01.jpg").write_bytes(buf.tobytes())
    with db.session_scope(engine) as session:
        source = Source(type=SourceType.MANUAL, display_name="Video files")
        session.add(source)
        session.flush()
        assert source.id is not None
        # The file is gone: only the saved key frame can be used.
        session.add(Video(id=1, source_id=source.id, file_path=None, file_deleted=True))
        session.add(Video(id=2, source_id=source.id, original_url="https://e.com/2"))
        session.flush()
        stored = artifacts.to_stored_path(paths, folder / "frame_01.jpg")
        session.add(
            Artifact(video_id=1, type="frames", params_hash="x", params={},
                     data={"frames": [{"t": 1.0, "path": stored}]})
        )  # fmt: skip
    made = thumbnail.thumbnail(engine, paths, 1)
    assert made is not None and size_of(made) == (360, 640)
    assert thumbnail.thumbnail(engine, paths, 2) is None  # a link only: nothing to show
    assert thumbnail.thumbnail(engine, paths, 99) is None

    app = create_app(paths, TOKEN, provider_factory=FakeLLM, run_worker=False)
    with TestClient(app, base_url="http://127.0.0.1") as c:
        ok_response = c.get("/videos/1/thumbnail", headers=AUTH)
        assert ok_response.headers["content-type"] == "image/jpeg"
        assert ok_response.content[:2] == b"\xff\xd8"
        missing = c.get("/videos/2/thumbnail", headers=AUTH)
        assert missing.status_code == 404 and "no picture" in missing.json()["error"]
        assert c.get("/videos/1/thumbnail").status_code == 401

        items = {v["id"]: v for v in c.get("/videos", headers=AUTH).json()["videos"]}
    assert items[1]["has_file"] is False and items[2]["source"] == "manual"
    assert items[1]["results"] == [] and items[1]["profiled"] is False
