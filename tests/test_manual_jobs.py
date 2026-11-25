"""API tests for hand-added jobs (`POST /jobs`, `POST /jobs/{id}/score`).

Covers: job + application created together and shown on the Kanban board;
the job is private to the account that added it (list, detail, status
patch, applications, scoring, dedup); reuse of a shared scraped job;
scoring queued only when the account has a processed CV and quota left;
per-account scores on the Kanban/detail views; GDPR export/delete of
hand-added jobs; `POST /run` refusing accounts without a CV.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import func, select

import api.router as router_module
from agents.dedup import canonical_url, content_hash, find_duplicate
from api.router import router
from auth.users import current_active_user
from database import init_db
from database.db import get_session
from database.models import CV, CandidateProfile, Job, MatchResult, PipelineRun, User
from observability.quota import LLMQuotaExceeded


@pytest.fixture()
def client() -> TestClient:
    init_db()
    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def fake_queue(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    """Replace the arq pool and quota counter: no Redis, nothing enqueued."""
    pool = SimpleNamespace(enqueue_job=AsyncMock())
    monkeypatch.setattr(router_module, "get_arq_pool", AsyncMock(return_value=pool))
    monkeypatch.setattr(router_module, "check_and_increment", AsyncMock(return_value=1))
    return pool.enqueue_job


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


async def _create_user() -> str:
    async with get_session() as session:
        user = User(email=f"manual-{uuid.uuid4().hex[:8]}@manual-jobs-test.local", hashed_password="!")
        session.add(user)
        await session.flush()
        return user.id


async def _give_cv(user_id: str) -> str:
    """A processed uploaded CV (CV row + extracted CandidateProfile)."""
    async with get_session() as session:
        cv = CV(
            user_id=user_id,
            raw_text="Python developer, 5 years.",
            content_hash=uuid.uuid4().hex,
            consent_given_at=datetime.utcnow(),
            last_extraction_status="success",
        )
        session.add(cv)
        await session.flush()
        session.add(CandidateProfile(user_id=user_id, cv_id=cv.id, prompt_version="test"))
        return cv.id


def _as(client: TestClient, user_id: str) -> None:
    client.app.dependency_overrides[current_active_user] = lambda: SimpleNamespace(id=user_id)


def _as_new_user(client: TestClient, with_cv: bool = False) -> str:
    user_id = _run(_create_user())
    if with_cv:
        _run(_give_cv(user_id))
    _as(client, user_id)
    return user_id


def _unique(prefix: str) -> str:
    return f"{prefix} {uuid.uuid4().hex[:8]}"


async def _job_count(**filters: str) -> int:
    async with get_session() as session:
        stmt = select(func.count()).select_from(Job)
        for column, value in filters.items():
            stmt = stmt.where(getattr(Job, column) == value)
        return int((await session.execute(stmt)).scalar_one())


async def _insert_shared_job(**fields: Any) -> str:
    async with get_session() as session:
        job = Job(source="mock", **fields)
        session.add(job)
        await session.flush()
        return job.id


# ---------- creating ----------

def test_add_job_creates_private_job_and_kanban_card(client: TestClient) -> None:
    user_id = _as_new_user(client)
    title = _unique("Manual ML Engineer")
    r = client.post(
        "/jobs",
        json={
            "title": f"  {title}  ",
            "company": "Acme",
            "location": "Berlin",
            "url": "https://example.com/jobs/" + uuid.uuid4().hex,
            "description": "Build things.",
        },
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["status"] == "interested"
    assert body["applied_at"] is None
    assert body["scoring_status"] == "no_cv"
    assert body["scoring_run_id"] is None

    job = client.get(f"/jobs/{body['job_id']}").json()
    assert job["title"] == title  # whitespace trimmed
    assert job["source"] == "manual"
    assert job["owner_user_id"] == user_id

    cards = client.get("/applications/kanban").json()["columns"]["interested"]
    assert [c["application_id"] for c in cards] == [body["id"]]
    assert cards[0]["score_final"] is None


def test_add_job_with_applied_status_sets_applied_at(client: TestClient) -> None:
    _as_new_user(client)
    r = client.post("/jobs", json={"title": _unique("Applied Job"), "company": "Acme", "status": "applied"})
    assert r.status_code == 201
    assert r.json()["status"] == "applied"
    assert r.json()["applied_at"] is not None


def test_add_same_job_twice_returns_409(client: TestClient) -> None:
    _as_new_user(client)
    payload = {"title": _unique("Twice"), "company": "Acme"}
    first = client.post("/jobs", json=payload)
    second = client.post("/jobs", json=payload)
    assert first.status_code == 201
    assert second.status_code == 409
    assert first.json()["id"] in second.json()["detail"]


def test_add_job_reuses_shared_scraped_job_with_same_url(client: TestClient) -> None:
    url = "https://example.com/jobs/scraped-" + uuid.uuid4().hex
    scraped_id = _run(
        _insert_shared_job(
            title="Scraped Title",
            company="Acme",
            location="Munich",
            url=url,
            canonical_url=canonical_url(url),
            content_hash=content_hash("Scraped Title", "Acme", "Munich"),
        )
    )
    _as_new_user(client)
    r = client.post("/jobs", json={"title": "Slightly different title", "company": "Acme", "url": url})
    assert r.status_code == 201
    assert r.json()["job_id"] == scraped_id
    assert client.get(f"/jobs/{scraped_id}").json()["owner_user_id"] is None  # still shared
    assert _run(_job_count(url=url)) == 1


@pytest.mark.parametrize(
    "payload",
    [
        {"title": "   ", "company": "Acme"},
        {"title": "Engineer", "company": ""},
        {"company": "Acme"},
        {"title": "Engineer", "company": "Acme", "status": "not-a-status"},
        {"title": "x" * 513, "company": "Acme"},
    ],
)
def test_add_job_rejects_invalid_input(client: TestClient, payload: dict[str, str]) -> None:
    _as_new_user(client)
    before = _run(_job_count(source="manual"))
    r = client.post("/jobs", json=payload)
    assert r.status_code == 422
    assert _run(_job_count(source="manual")) == before


# ---------- privacy ----------

def test_hand_added_job_is_invisible_to_other_accounts(client: TestClient) -> None:
    _as_new_user(client)
    title = _unique("Private")
    job_id = client.post("/jobs", json={"title": title, "company": "Acme"}).json()["job_id"]

    _as_new_user(client, with_cv=True)
    assert client.get(f"/jobs/{job_id}").status_code == 404
    assert job_id not in [j["id"] for j in client.get("/jobs?limit=200").json()["jobs"]]
    assert client.post("/applications", json={"job_id": job_id}).status_code == 404
    assert client.patch(f"/jobs/{job_id}/status", json={"status": "applied"}).status_code == 404
    assert client.post(f"/jobs/{job_id}/score").status_code == 404


def test_same_job_added_by_two_accounts_gives_two_private_copies(client: TestClient) -> None:
    payload = {"title": _unique("Both"), "company": "Acme", "url": "https://example.com/" + uuid.uuid4().hex}
    _as_new_user(client)
    first = client.post("/jobs", json=payload).json()
    _as_new_user(client)
    second = client.post("/jobs", json=payload).json()
    assert first["job_id"] != second["job_id"]


def test_pipeline_dedup_never_matches_a_private_job(client: TestClient) -> None:
    payload = {"title": _unique("Dedup"), "company": "Acme", "location": "Berlin",
               "url": "https://example.com/" + uuid.uuid4().hex}
    _as_new_user(client)
    assert client.post("/jobs", json=payload).status_code == 201

    async def _pipeline_lookup() -> Job | None:
        async with get_session() as session:
            return await find_duplicate(session, payload)  # no owner: what the pipeline does

    assert _run(_pipeline_lookup()) is None


def test_kanban_and_detail_only_show_own_scores(client: TestClient) -> None:
    other_id = _run(_create_user())
    job_id = _run(_insert_shared_job(title=_unique("Shared"), company="Acme"))

    async def _other_users_score() -> None:
        async with get_session() as session:
            session.add(MatchResult(user_id=other_id, job_id=job_id, score_final=0.93))

    _run(_other_users_score())
    _as_new_user(client)
    app_id = client.post("/applications", json={"job_id": job_id}).json()["id"]
    card = next(
        c for c in client.get("/applications/kanban").json()["columns"]["interested"] if c["application_id"] == app_id
    )
    assert card["score_final"] is None
    assert client.get(f"/applications/{app_id}").json()["match_result"] is None


# ---------- scoring ----------

def test_add_job_queues_scoring_when_account_has_cv(client: TestClient, fake_queue: AsyncMock) -> None:
    user_id = _as_new_user(client, with_cv=True)
    body = client.post("/jobs", json={"title": _unique("Scored"), "company": "Acme"}).json()
    assert body["scoring_status"] == "queued"
    run_id = body["scoring_run_id"]
    fake_queue.assert_awaited_once_with("score_job_job", run_id, body["job_id"], user_id, _job_id=run_id)

    async def _run_row() -> PipelineRun | None:
        async with get_session() as session:
            return await session.get(PipelineRun, run_id)

    run = _run(_run_row())
    assert run is not None and run.status == "pending" and run.user_id == user_id


def test_add_job_over_quota_still_saves_job(client: TestClient, fake_queue: AsyncMock,
                                            monkeypatch: pytest.MonkeyPatch) -> None:
    user_id = _as_new_user(client, with_cv=True)
    monkeypatch.setattr(
        router_module, "check_and_increment", AsyncMock(side_effect=LLMQuotaExceeded(user_id, 51, 50))
    )
    r = client.post("/jobs", json={"title": _unique("Over quota"), "company": "Acme"})
    assert r.status_code == 201
    assert r.json()["scoring_status"] == "quota_exceeded"
    assert client.get(f"/jobs/{r.json()['job_id']}").status_code == 200
    fake_queue.assert_not_awaited()


def test_rescore_endpoint(client: TestClient, fake_queue: AsyncMock) -> None:
    user_id = _as_new_user(client)
    job_id = client.post("/jobs", json={"title": _unique("Later"), "company": "Acme"}).json()["job_id"]
    assert client.post(f"/jobs/{job_id}/score").json() == {"scoring_status": "no_cv", "scoring_run_id": None}

    _run(_give_cv(user_id))
    r = client.post(f"/jobs/{job_id}/score")
    assert r.status_code == 202
    assert r.json()["scoring_status"] == "queued"
    assert fake_queue.await_args.args[2] == job_id


def test_run_requires_an_uploaded_cv(client: TestClient, fake_queue: AsyncMock) -> None:
    _as_new_user(client)
    r = client.post("/run", json={})
    assert r.status_code == 409
    fake_queue.assert_not_awaited()

    _as_new_user(client, with_cv=True)
    assert client.post("/run", json={}).status_code == 202


def test_my_cv_endpoint(client: TestClient) -> None:
    user_id = _as_new_user(client)
    assert client.get("/me/cv").json() is None
    cv_id = _run(_give_cv(user_id))
    body = client.get("/me/cv").json()
    assert body["id"] == cv_id
    assert body["profile"] is not None


# ---------- GDPR ----------

def test_export_and_delete_cover_hand_added_jobs(client: TestClient) -> None:
    _as_new_user(client)
    job_id = client.post("/jobs", json={"title": _unique("Mine"), "company": "Acme"}).json()["job_id"]

    export = client.get("/me/export").json()
    assert [j["id"] for j in export["jobs_added"]] == [job_id]

    deleted = client.delete("/me").json()["deleted_counts"]
    assert deleted["jobs_added"] == 1

    async def _job_exists() -> bool:
        async with get_session() as session:
            return await session.get(Job, job_id) is not None

    assert _run(_job_exists()) is False
