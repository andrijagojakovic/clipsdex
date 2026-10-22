import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from clipsdex_engine import db
from clipsdex_engine.api.app import create_app
from clipsdex_engine.api.security import MediaLinks
from clipsdex_engine.config import DataPaths
from clipsdex_engine.db.models import Source, SourceType, Video
from mocks.fake_llm import FakeLLM

TOKEN = "p" * 40
AUTH = {"Authorization": f"Bearer {TOKEN}"}
CONTENT = os.urandom(5000)


def test_links_name_one_video_and_expire() -> None:
    now = [1_000_000.0]
    links = MediaLinks(key=b"k" * 32, clock=lambda: now[0])
    link = links.link(7, seconds=60)
    path, query = link.split("?")
    assert path == "/media/7"
    params = dict(part.split("=") for part in query.split("&"))
    exp, sig = int(params["exp"]), params["sig"]
    assert links.valid(7, exp, sig)
    assert not links.valid(8, exp, sig)  # another video
    assert not links.valid(7, exp + 1, sig)  # a longer life can't be forged
    assert not links.valid(7, exp, sig[:-1] + ("0" if sig[-1] != "0" else "1"))
    assert not links.valid(7, exp, "ž" * 64)  # odd input is refused, not a crash
    assert not MediaLinks(key=b"x" * 32, clock=lambda: now[0]).valid(7, exp, sig)  # other start
    now[0] += 61
    assert not links.valid(7, exp, sig)  # expired


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    """Video 1 is a local file (Cyrillic path), 2 is a link only, 3's file was moved away."""
    folder = tmp_path / "Моји видео"
    folder.mkdir()
    (folder / "клип.mp4").write_bytes(CONTENT)
    paths = DataPaths(tmp_path / "data")
    engine = db.open_database(paths)
    with db.session_scope(engine) as session:
        source = Source(type=SourceType.MANUAL, display_name="Video files")
        session.add(source)
        session.flush()
        assert source.id is not None
        session.add(Video(id=1, source_id=source.id, file_path=str(folder / "клип.mp4")))
        session.add(Video(id=2, source_id=source.id, original_url="https://e.com/2"))
        session.add(Video(id=3, source_id=source.id, file_path=str(folder / "gone.mp4")))
    app = create_app(paths, TOKEN, provider_factory=FakeLLM, run_worker=False)
    with TestClient(app, base_url="http://127.0.0.1") as c:
        yield c


def test_player_streams_the_file_with_seeking(client: TestClient) -> None:
    link = client.get("/videos/1/play", headers=AUTH).json()
    assert link["url"].startswith("/media/1?exp=") and link["expires_in"] == 7200

    full = client.get(link["url"])  # no token: the signed link is enough
    assert full.status_code == 200 and full.content == CONTENT
    assert full.headers["content-type"] == "video/mp4"
    assert full.headers["accept-ranges"] == "bytes"

    part = client.get(link["url"], headers={"Range": "bytes=100-199"})
    assert part.status_code == 206 and part.content == CONTENT[100:200]
    assert part.headers["content-range"] == f"bytes 100-199/{len(CONTENT)}"


def test_media_needs_a_valid_link(client: TestClient) -> None:
    url = client.get("/videos/1/play", headers=AUTH).json()["url"]
    for bad in (
        "/media/1",
        "/media/1?exp=99999999999&sig=abc",
        url.replace("/media/1", "/media/3"),
    ):
        r = client.get(bad)
        assert r.status_code == 403 and r.json() == {
            "error": "This video link is not valid or has expired."
        }
    assert client.get(url, headers={"Host": "evil.example.com"}).status_code == 400
    # Only reading is exempt from the token, and only at /media/<id>.
    assert client.post(url).status_code == 401
    assert client.get(url.replace("/media/1", "/media/1/x")).status_code == 401
    assert client.get("/videos/1/play").status_code == 401


def test_videos_without_a_file_say_why(client: TestClient) -> None:
    link_only = client.get("/videos/2/play", headers=AUTH)
    assert link_only.status_code == 404 and "only the link is kept" in link_only.json()["error"]
    moved = client.get("/videos/3/play", headers=AUTH)
    assert moved.status_code == 404 and "no longer at" in moved.json()["error"]
    assert client.get("/videos/99/play", headers=AUTH).status_code == 404
