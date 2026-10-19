import json
import os
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from clipsdex_engine import artifacts, categories, config, db
from clipsdex_engine.api import security, server
from clipsdex_engine.api.app import create_app
from clipsdex_engine.artifacts import current
from clipsdex_engine.config import DataPaths
from clipsdex_engine.db.models import (
    Artifact,
    AssignedBy,
    CategoryAssignment,
    Source,
    SourceType,
    Video,
)
from mocks.fake_llm import FakeLLM

TOKEN = "t" * 40
AUTH = {"Authorization": f"Bearer {TOKEN}"}
ANSWER = json.dumps(
    {"category": "business", "confidence": 0.8, "tags": ["pricing"],
     "summary": "Pricing advice.", "reason": "Talks about prices."}
)  # fmt: skip


@pytest.fixture
def paths(tmp_path: Path) -> DataPaths:
    """Video 1 (business) has every input categorize needs; video 2 is an edit."""
    paths = DataPaths(tmp_path / "Подаци")
    engine = db.open_database(paths)
    with db.session_scope(engine) as session:
        categories.ensure_default_categories(session)
        source = Source(type=SourceType.MANUAL, display_name="Video files")
        session.add(source)
        session.flush()
        assert source.id is not None
        session.add(Video(id=1, source_id=source.id, file_path="C:/v/pricing.mp4", author="@a"))
        session.add(Video(id=2, source_id=source.id, original_url="https://example.com/r/2"))
        session.flush()
        for vid, name in ((1, "business"), (2, "edit")):
            session.add(
                CategoryAssignment(
                    video_id=vid,
                    category_id=categories.get(session, name).id or 0,
                    assigned_by=AssignedBy.AUTO,
                )
            )
        models = db.load_settings(session).models()
        for name, data in (
            ("transcript", {"language": "en", "text": "Raise your prices."}),
            ("ocr", {"text": ""}),
            ("frame_descriptions", {"frames": [], "skipped": []}),
        ):
            params = current.params(name, models)
            session.add(
                Artifact(video_id=1, type=name, params_hash=artifacts.params_hash(params),
                         params=params, data=data)
            )  # fmt: skip
    return paths


@pytest.fixture
def fake() -> FakeLLM:
    return FakeLLM(default_reply=ANSWER)


@pytest.fixture
def client(paths: DataPaths, fake: FakeLLM) -> Iterator[TestClient]:
    app = create_app(paths, TOKEN, provider_factory=lambda: fake, run_worker=False)
    with TestClient(app, base_url="http://127.0.0.1:5000") as c:
        yield c


# --- Security ----------------------------------------------------------------------------


def test_every_request_needs_the_token(client: TestClient) -> None:
    assert client.get("/health").status_code == 401
    assert client.get("/health", headers={"Authorization": "Bearer nope"}).status_code == 401
    assert client.get("/health", headers={"Authorization": TOKEN}).status_code == 401
    assert client.get("/videos/1", headers={"Authorization": f"bearer {TOKEN}"}).status_code == 401
    ok = client.get("/health", headers=AUTH)
    assert ok.status_code == 200 and ok.json()["status"] == "ok"
    # Unknown routes are hidden too: without the token nothing is revealed.
    assert client.get("/nope").status_code == 401


@pytest.mark.parametrize(
    ("host", "allowed"),
    [
        ("127.0.0.1:5000", True),
        ("localhost:5000", True),
        ("[::1]:5000", True),
        ("evil.example.com", False),
        ("127.0.0.1.evil.example.com:5000", False),
        ("", False),
    ],
)
def test_only_loopback_host_names(client: TestClient, host: str, allowed: bool) -> None:
    r = client.get("/health", headers={**AUTH, "Host": host})
    assert (r.status_code == 200) is allowed
    if not allowed:
        assert r.status_code == 400 and "local machine" in r.json()["error"]


def test_no_public_docs(client: TestClient) -> None:
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert client.get(path).status_code == 401
        assert client.get(path, headers=AUTH).status_code == 404


def test_cors_only_for_the_desktop_app(client: TestClient) -> None:
    preflight = {"Access-Control-Request-Method": "POST",
                 "Access-Control-Request-Headers": "authorization"}  # fmt: skip
    ok = client.options("/plan", headers={**preflight, "Origin": "http://tauri.localhost"})
    assert ok.status_code == 200
    assert ok.headers["access-control-allow-origin"] == "http://tauri.localhost"
    bad = client.options("/plan", headers={**preflight, "Origin": "https://evil.example.com"})
    assert "access-control-allow-origin" not in bad.headers
    # A real cross-site request still needs the token.
    assert client.get("/health", headers={"Origin": "https://evil.example.com"}).status_code == 401


def test_token_from_the_parent_must_be_strong(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(security.TOKEN_ENV, raising=False)
    assert security.token_from_env() is None
    monkeypatch.setenv(security.TOKEN_ENV, "short")
    with pytest.raises(security.TokenError, match="at least 32"):
        security.token_from_env()
    monkeypatch.setenv(security.TOKEN_ENV, "x" * 40)
    assert security.token_from_env() == "x" * 40
    assert len(security.new_token()) >= 40 and security.new_token() != security.new_token()


def test_binds_to_loopback_only() -> None:
    sock = server.bind(0)
    try:
        host, port = sock.getsockname()
        assert host == "127.0.0.1" and port > 0
    finally:
        sock.close()


def _start_engine(paths: DataPaths, env_token: str | None) -> tuple[subprocess.Popen[str], dict]:
    env = {**os.environ, config.DATA_DIR_ENV: str(paths.root)}
    env.pop(security.TOKEN_ENV, None)
    if env_token:
        env[security.TOKEN_ENV] = env_token
    proc = subprocess.Popen(
        [sys.executable, "-c", "from clipsdex_engine.cli import app; app()", "serve"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    assert proc.stdout is not None
    line = proc.stdout.readline()  # blocks until the server accepts connections
    assert line, proc.stderr.read() if proc.stderr else "no output"
    return proc, json.loads(line)


@pytest.mark.parametrize("env_token", [None, "p" * 40])
def test_real_server_announces_itself(paths: DataPaths, env_token: str | None) -> None:
    proc, ready = _start_engine(paths, env_token)
    try:
        assert ready["event"] == "ready" and ready["port"] > 0
        if env_token:
            assert "token" not in ready  # the parent knows it; never printed
        token = env_token or ready["token"]
        url = f"http://127.0.0.1:{ready['port']}/health"
        assert httpx.get(url, timeout=10).status_code == 401
        r = httpx.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=10)
        assert r.status_code == 200 and r.json()["worker"] in (
            "running",
            "waiting: another worker is running",
        )
    finally:
        proc.terminate()
        proc.wait(timeout=20)
        if proc.stdout:
            proc.stdout.close()
        if proc.stderr:
            proc.stderr.close()


# --- Functions -------------------------------------------------------------------------


def test_library_and_videos(client: TestClient, tmp_path: Path) -> None:
    videos = client.get("/videos", headers=AUTH).json()["videos"]
    assert [(v["id"], v["category"]) for v in videos] == [(1, "business"), (2, "edit")]
    assert [
        v["id"] for v in client.get("/videos?category=EDIT", headers=AUTH).json()["videos"]
    ] == [2]
    one = client.get("/videos/1", headers=AUTH).json()
    assert one["name"] == "pricing.mp4" and one["transcript"]["text"] == "Raise your prices."
    missing = client.get("/videos/99", headers=AUTH)
    assert missing.status_code == 404 and missing.json() == {"error": "There is no video #99."}

    folder = tmp_path / "нови"
    folder.mkdir()
    (folder / "notes.txt").write_text("not a video")
    r = client.post("/import", headers=AUTH, json={"paths": [str(folder / "gone.mp4")]})
    assert r.json()["added"] == [] and r.json()["failed"][0]["path"].endswith("gone.mp4")
    bad = client.post("/import", headers=AUTH, json={"paths": []})
    assert bad.status_code == 422 and bad.json()["error"].startswith("Invalid request: paths")


def test_categories_and_moves(client: TestClient) -> None:
    assert client.post("/categories", headers=AUTH, json={"name": "cars"}).status_code == 201
    dup = client.post("/categories", headers=AUTH, json={"name": "Cars"})
    assert dup.status_code == 400 and "already exists" in dup.json()["error"]
    renamed = client.patch("/categories/cars", headers=AUTH,
                           json={"new_name": "autos", "description": "Car reviews."})  # fmt: skip
    assert renamed.json() == {"name": "autos", "description": "Car reviews.", "color": None}
    moved = client.post("/videos/move", headers=AUTH, json={"video_ids": [2], "to": "autos"})
    assert moved.json() == {"changed": [2]}
    listed = {c["name"]: c for c in client.get("/categories", headers=AUTH).json()["categories"]}
    assert listed["autos"]["videos"] == 1 and listed["autos"]["by_user"] == 1
    merged = client.post("/categories/autos/merge", headers=AUTH, json={"into": "edit"})
    assert merged.json() == {"moved": 1}
    assert client.delete("/categories/travel", headers=AUTH).json() == {"deleted": "travel"}
    assert client.delete("/categories/edit", headers=AUTH).status_code == 400


def test_plan_run_and_jobs(client: TestClient) -> None:
    body = {"analyzers": ["categorize"], "video_ids": [1]}
    plan = client.post("/plan", headers=AUTH, json=body).json()
    assert plan["videos"][0]["analyzers"] == ["categorize"] and plan["total_seconds"] > 0
    assert client.post("/plan", headers=AUTH, json={}).status_code == 400
    unknown = client.post("/plan", headers=AUTH, json={"analyzers": ["teleport"]})
    assert unknown.status_code == 400 and "Unknown analyzer" in unknown.json()["error"]
    assert client.post("/run", headers=AUTH, json={"analyzers": ["song_id"]}).status_code == 400

    run = client.post("/run", headers=AUTH, json=body).json()
    [job_id] = run["queued"]
    jobs = client.get("/jobs", headers=AUTH).json()
    assert jobs["counts"]["queued"] == 1 and jobs["jobs"][0]["id"] == job_id
    assert client.post("/jobs/cancel", headers=AUTH, json={}).json() == {"cancelled": 1}
    assert client.get("/jobs?status=cancelled", headers=AUTH).json()["jobs"][0]["id"] == job_id


def test_background_worker_does_the_jobs(paths: DataPaths, fake: FakeLLM) -> None:
    app = create_app(paths, TOKEN, provider_factory=lambda: fake, run_worker=True)
    with TestClient(app, base_url="http://127.0.0.1") as c:
        assert c.get("/health", headers=AUTH).json()["worker"] == "running"
        c.post("/run", headers=AUTH, json={"analyzers": ["categorize"], "video_ids": [1]})
        deadline = time.monotonic() + 20
        jobs: list[dict[str, object]] = []
        while time.monotonic() < deadline:
            jobs = c.get("/jobs", headers=AUTH).json()["jobs"]
            if jobs and jobs[0]["status"] == "done":
                break
            time.sleep(0.2)
        assert jobs[0]["status"] == "done"
        result = c.get("/videos/1", headers=AUTH).json()["results"]["categorize"]
        assert result["value"]["summary"] == "Pricing advice."


def test_settings_doctor_search_brief_export(client: TestClient) -> None:
    s = client.get("/settings", headers=AUTH).json()
    assert s["settings"]["preset"] == "balanced" and s["models"]["text"] == "gemma3:4b"
    changed = client.put("/settings", headers=AUTH, json={**s["settings"], "preset": "fast"})
    assert changed.json()["models"]["text"] == "llama3.2:3b"
    assert client.put("/settings", headers=AUTH, json={"preset": "turbo"}).status_code == 422

    doctor = client.get("/doctor", headers=AUTH).json()
    assert doctor["runtime"]["ok"] is True and {t["name"] for t in doctor["tools"]} >= {"ffmpeg"}

    hits = client.get("/search?q=prices", headers=AUTH).json()["hits"]
    assert {h["video_id"] for h in hits} == {1, 2}
    assert client.get("/search?q=", headers=AUTH).status_code == 422

    nothing = client.post("/brief", headers=AUTH, json={})
    assert nothing.status_code == 400 and "No notes" in nothing.json()["error"]

    csv_text = client.get("/export/csv?category=business", headers=AUTH)
    assert csv_text.headers["content-type"].startswith("text/csv")
    assert csv_text.text.count("\n") == 2  # header + video 1
    assert client.get("/export/pdf", headers=AUTH).status_code == 422


def test_runtime_down_is_a_clear_error(paths: DataPaths) -> None:
    app = create_app(paths, TOKEN, provider_factory=lambda: FakeLLM(running=False),
                     run_worker=False)  # fmt: skip
    with TestClient(app, base_url="http://127.0.0.1") as c:
        r = c.get("/search?q=x", headers=AUTH)
    assert r.status_code == 503 and "not running" in r.json()["error"]
