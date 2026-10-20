"""Whole flows through the API, as the desktop app will use it (fake models only)."""

import json
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from clipsdex_engine import db, export
from clipsdex_engine import profile as quick_profile
from clipsdex_engine.api.app import create_app
from clipsdex_engine.config import DataPaths
from clipsdex_engine.db.models import Result, Source, SourceType, Video
from mocks.fake_llm import FakeLLM
from synthetic import FakeReader, FakeTagger, SyntheticMedia, needs_ffmpeg

TOKEN = "f" * 40
AUTH = {"Authorization": f"Bearer {TOKEN}"}


def guess(category: str) -> str:
    return json.dumps({"category": category, "confidence": 0.7, "reason": "Talks about prices."})


def wait_until_idle(client: TestClient, seconds: float = 30) -> dict[str, int]:
    deadline = time.monotonic() + seconds
    counts: dict[str, int] = {}
    while time.monotonic() < deadline:
        counts = client.get("/jobs", headers=AUTH).json()["counts"]
        if counts["queued"] == 0 and counts["running"] == 0:
            return counts
        time.sleep(0.2)
    raise AssertionError(f"jobs still running: {counts}")


@pytest.fixture
def client(tmp_path: Path) -> Iterator[tuple[TestClient, FakeLLM, DataPaths]]:
    paths = DataPaths(tmp_path / "Подаци")
    fake = FakeLLM()
    app = create_app(paths, TOKEN, provider_factory=lambda: fake, run_worker=True)
    with TestClient(app, base_url="http://127.0.0.1") as c:
        yield c, fake, paths


@needs_ffmpeg
def test_import_profile_and_first_guess(
    client: tuple[TestClient, FakeLLM, DataPaths],
    synthetic_media: SyntheticMedia,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    c, fake, _paths = client
    monkeypatch.setattr(
        quick_profile,
        "default_models",
        lambda *_a, **_k: quick_profile.ProfileModels(
            tagger=quick_profile.Lazy(lambda: FakeTagger(speech=0.9)),
            ocr=quick_profile.Lazy(lambda: FakeReader([("3 PRICING MISTAKES", 0.95)])),
        ),
    )
    fake.default_reply = guess("business")

    imported = c.post(
        "/import",
        headers=AUTH,
        json={"paths": [str(synthetic_media.with_audio), str(synthetic_media.not_video)]},
    ).json()
    assert len(imported["added"]) == 2 and imported["failed"] == []
    again = c.post("/import", headers=AUTH, json={"paths": [str(synthetic_media.with_audio)]})
    assert len(again.json()["duplicates"]) == 1

    queued = c.post("/profile", headers=AUTH, json={}).json()["queued"]
    assert len(queued) == 4  # a profile and a guess per video
    counts = wait_until_idle(c)
    # The broken file can't be read; the guess that needed its profile is skipped.
    assert counts["done"] == 2 and counts["failed"] == 2

    jobs = c.get("/jobs", headers=AUTH).json()["jobs"]
    failed = {j["name"]: j["error"] for j in jobs if j["status"] == "failed"}
    assert failed["profile"]
    assert failed["category_guess"].startswith("Skipped: the job it needs")
    assert len(fake.calls) == 1  # only the good video was guessed

    videos = {v["name"]: v for v in c.get("/videos", headers=AUTH).json()["videos"]}
    assert videos["tone.mp4"]["category"] == "business"
    assert videos["tone.mp4"]["category_by"] == "auto"
    one = c.get(f"/videos/{videos['tone.mp4']['id']}", headers=AUTH).json()
    assert one["profile"]["duration_s"] == pytest.approx(2.0, abs=0.1)
    assert one["on_screen_text"] == "3 PRICING MISTAKES"
    assert one["results"]["category_guess"]["evidence"] == "Talks about prices."


def _add_notes(paths: DataPaths) -> None:
    engine = db.open_database(paths)
    with db.session_scope(engine) as session:
        source = Source(type=SourceType.MANUAL, display_name="Video files")
        session.add(source)
        session.flush()
        assert source.id is not None
        for vid in (1, 2):
            session.add(Video(id=vid, source_id=source.id, original_url=f"https://e.com/{vid}"))
        session.flush()
        for vid, message in ((1, "Sell packages."), (2, "Always follow up.")):
            session.add(
                Result(video_id=vid, analyzer="brief", method="test",
                       value={"has_advice": True, "main_message": message,
                              "advice": [message], "facts": [], "steps": []})
            )  # fmt: skip


def test_brief_merge_through_the_api(client: tuple[TestClient, FakeLLM, DataPaths]) -> None:
    c, fake, paths = client
    _add_notes(paths)
    fake.replies = [
        json.dumps({"summary": "Sell and follow up.", "themes": [
            {"title": "Sales", "points": [{"text": "Sell packages and follow up.",
                                           "sources": [1, 2]}]}]})
    ]  # fmt: skip
    doc = c.post("/brief", headers=AUTH, json={"video_ids": [1, 2], "title": "Sales"}).json()
    assert (doc["title"], doc["used"], doc["themes"], doc["missing"]) == ("Sales", 2, 1, [])
    assert "## Sales\n\n- Sell packages and follow up. [1][2]" in doc["markdown"]
    assert fake.events == ["load gemma3:4b", "unload gemma3:4b"]


def test_export_json_shape(client: tuple[TestClient, FakeLLM, DataPaths]) -> None:
    c, _fake, paths = client
    _add_notes(paths)
    r = c.get("/export/json", headers=AUTH)
    assert r.headers["content-type"] == "application/json; charset=utf-8"
    data = r.json()
    assert set(data) == {"format", "format_version", "app_version", "exported_at", "categories",
                         "videos"}  # fmt: skip
    assert data["format_version"] == export.FORMAT_VERSION
    record = data["videos"][0]
    assert set(record) >= {"id", "name", "category", "profile", "results", "transcript"}
    assert record["results"]["brief"]["value"]["main_message"] == "Sell packages."
    one = c.get("/export/json?video_ids=2", headers=AUTH).json()
    assert [v["id"] for v in one["videos"]] == [2]
    songs = c.get("/export/songs", headers=AUTH)
    assert songs.headers["content-type"].startswith("text/markdown")


BAD_REQUESTS: list[tuple[str, str, Any, int]] = [
    ("GET", "/videos/abc", None, 422),
    ("GET", "/videos?limit=0", None, 422),
    ("GET", "/videos/99", None, 404),
    ("POST", "/videos/move", {}, 422),
    ("POST", "/videos/move", {"video_ids": [99], "to": "edit"}, 400),
    ("POST", "/categories", {"name": ""}, 400),
    ("PATCH", "/categories/nope", {"description": "x"}, 400),
    ("POST", "/categories/nope/merge", {"into": "edit"}, 400),
    ("DELETE", "/categories/nope", None, 400),
    ("POST", "/plan", {"artifacts": ["vibes"]}, 400),
    ("POST", "/run", {"analyzers": ["movie_id"]}, 400),
    ("GET", "/jobs?status=weird", None, 422),
    ("POST", "/jobs/cancel", {"job_ids": "all"}, 422),
    ("GET", "/search", None, 422),
    ("POST", "/brief", {"video_ids": [1]}, 400),
    ("GET", "/export/pdf", None, 422),
    ("PUT", "/settings", {"max_frames": 0}, 422),
    ("POST", "/import", {"paths": "C:/x"}, 422),
    ("DELETE", "/health", None, 405),
    ("GET", "/no-such-route", None, 404),
]


@pytest.mark.parametrize(("method", "path", "body", "status"), BAD_REQUESTS)
def test_every_error_has_the_same_shape(
    client: tuple[TestClient, FakeLLM, DataPaths],
    method: str,
    path: str,
    body: Any,
    status: int,
) -> None:
    c, _fake, _paths = client
    r = c.request(method, path, headers=AUTH, json=body)
    assert r.status_code == status
    data = r.json()
    assert list(data) == ["error"] and isinstance(data["error"], str) and data["error"]


def test_broken_json_body_is_a_clear_error(
    client: tuple[TestClient, FakeLLM, DataPaths],
) -> None:
    c, _fake, _paths = client
    r = c.post("/plan", headers={**AUTH, "Content-Type": "application/json"}, content=b"{")
    assert r.status_code == 422 and r.json()["error"].startswith("Invalid request")
