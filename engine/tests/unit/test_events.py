import json
import threading
import time
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine
from test_api import ANSWER, AUTH, TOKEN
from test_api import paths as paths

from clipsdex_engine import db
from clipsdex_engine.api import events
from clipsdex_engine.api.app import create_app
from clipsdex_engine.config import DataPaths
from clipsdex_engine.db.models import JobKind
from clipsdex_engine.jobs import queue
from mocks.fake_llm import FakeLLM


def parse(text: str) -> list[tuple[str, dict[str, Any]]]:
    """(event type, data) for every event in an SSE body; comments are skipped."""
    found = []
    for block in text.split("\n\n"):
        fields = dict(
            line.split(": ", 1) for line in block.splitlines() if ": " in line and line[0] != ":"
        )
        if "event" in fields:
            found.append((fields["event"], json.loads(fields["data"])))
    return found


def read_stream(client: TestClient, seconds: float) -> list[tuple[str, dict[str, Any]]]:
    with client.stream("GET", f"/events?for_seconds={seconds}", headers=AUTH) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        return parse(r.read().decode())


@pytest.fixture
def client(paths: DataPaths) -> Iterator[TestClient]:
    app = create_app(paths, TOKEN, provider_factory=lambda: FakeLLM(default_reply=ANSWER),
                     run_worker=False)  # fmt: skip
    with TestClient(app, base_url="http://127.0.0.1") as c:
        yield c


def engine_of(paths: DataPaths) -> Engine:
    return db.open_database(paths)


def test_events_need_the_token(client: TestClient) -> None:
    assert client.get("/events?for_seconds=0.1").status_code == 401


def test_snapshot_then_job_changes(client: TestClient, paths: DataPaths) -> None:
    [job_id] = client.post(
        "/run", headers=AUTH, json={"analyzers": ["categorize"], "video_ids": [1]}
    ).json()["queued"]
    engine = engine_of(paths)

    def work() -> None:  # what a worker does, while the client listens
        time.sleep(0.6)
        claimed = queue.claim_next(engine)
        assert claimed is not None and claimed.id == job_id
        time.sleep(0.6)
        queue.set_progress(engine, job_id, 0.5)
        time.sleep(0.6)
        queue.mark_done(engine, job_id)

    t = threading.Thread(target=work)
    t.start()
    got = read_stream(client, 3.0)
    t.join()

    kind, snapshot = got[0]
    assert kind == "snapshot"
    assert snapshot["counts"]["queued"] == 1 and [j["id"] for j in snapshot["jobs"]] == [job_id]
    states = [(d["status"], d["progress"]) for k, d in got if k == "job"]
    assert states == [("running", 0.0), ("running", 0.5), ("done", 1.0)]
    last_counts = [d for k, d in got if k == "counts"][-1]
    assert last_counts["done"] == 1 and last_counts["queued"] == 0


def test_a_finished_job_is_reported_once(paths: DataPaths) -> None:
    engine = engine_of(paths)
    watcher = events.JobWatcher(engine)
    assert watcher.snapshot()["jobs"] == []
    with db.session_scope(engine) as session:
        job = queue.enqueue(session, queue.JobSpec(JobKind.ARTIFACT, "frames", 1, {}))
        assert job.id is not None
        job_id = job.id
    assert [e["status"] for e in watcher.changes()] == ["queued"]
    queue.claim_next(engine)
    queue.mark_done(engine, job_id)
    assert [e["status"] for e in watcher.changes()] == ["done"]
    assert watcher.changes() == [] and watcher.changes() == []


def test_worker_messages_arrive_live(paths: DataPaths) -> None:
    app = create_app(paths, TOKEN, provider_factory=lambda: FakeLLM(default_reply=ANSWER),
                     run_worker=True)  # fmt: skip
    with TestClient(app, base_url="http://127.0.0.1") as c:
        got: list[tuple[str, dict[str, Any]]] = []
        listener = threading.Thread(target=lambda: got.extend(read_stream(c, 5.0)))
        listener.start()
        time.sleep(0.5)  # connected before the work starts
        c.post("/run", headers=AUTH, json={"analyzers": ["categorize"], "video_ids": [1]})
        listener.join()
    messages = [d["message"] for k, d in got if k == "worker"]
    assert "load ollama:gemma3:4b" in messages
    assert any(m.startswith("run #") and m.endswith("categorize") for m in messages)
    assert ("done", 1.0) in [(d["status"], d["progress"]) for k, d in got if k == "job"]


def test_slow_clients_lose_old_worker_messages_not_the_server() -> None:
    hub = events.EventHub()
    inbox = hub.subscribe()
    for n in range(events.SUBSCRIBER_BUFFER + 50):
        hub.worker_message(f"m{n}")
    first = inbox.get_nowait()
    assert first["message"] == "m50" and inbox.qsize() == events.SUBSCRIBER_BUFFER - 1
    hub.unsubscribe(inbox)
    hub.worker_message("after")  # no subscribers: nothing to do, no error


def test_sse_format() -> None:
    assert events.sse("job", {"id": 1, "name": "ćao"}, 7) == (
        'id: 7\nevent: job\ndata: {"id": 1, "name": "ćao"}\n\n'
    )
