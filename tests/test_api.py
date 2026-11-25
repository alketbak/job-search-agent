"""FastAPI integration tests - spin up the app with TestClient."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.router import router
from database import init_db
from tests.auth_helpers import override_auth


@pytest.fixture(scope="module")
def client() -> TestClient:
    init_db()
    # Minimal app - we don't boot the scheduler or MCP server in tests.
    app = FastAPI()
    app.include_router(router)
    override_auth(app)
    with TestClient(app) as c:
        yield c


def test_health(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "version" in body


def test_list_jobs_returns_list(client: TestClient) -> None:
    r = client.get("/jobs")
    assert r.status_code == 200
    body = r.json()
    assert "total" in body and "jobs" in body
    assert isinstance(body["jobs"], list)


def test_graph_returns_nodes_and_links(client: TestClient) -> None:
    r = client.get("/graph")
    assert r.status_code == 200
    body = r.json()
    assert "nodes" in body and "links" in body
    assert isinstance(body["nodes"], list)
    assert isinstance(body["links"], list)


def test_runs_returns_list(client: TestClient) -> None:
    r = client.get("/runs")
    assert r.status_code == 200
    body = r.json()
    assert "runs" in body
    assert isinstance(body["runs"], list)


def test_skill_gaps_returns_list(client: TestClient) -> None:
    r = client.get("/skill-gaps")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_invalid_status_rejected(client: TestClient) -> None:
    r = client.get("/jobs", params={"status": "totally-invalid"})
    assert r.status_code == 400
