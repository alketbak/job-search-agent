"""Multi-user isolation tests.

Every endpoint is user-scoped: user A must never read user B's CV or
matches. Two real users are created in the test DB
(FK constraints on user_id need real rows, not arbitrary strings) and the
`current_active_user` dependency is swapped between them mid-test via
`app.dependency_overrides` - see tests/auth_helpers.py's docstring for why
that's the right way to test auth-protected routes without a real
register/login/JWT round trip per test.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.router import router
from auth.users import current_active_user
from database import init_db
from database.db import get_session
from database.models import CV, Job, MatchResult, User


@pytest.fixture()
def client() -> TestClient:
    init_db()
    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as c:
        yield c


async def _create_user(email: str) -> str:
    async with get_session() as session:
        user = User(email=email, hashed_password="!")
        session.add(user)
        await session.flush()
        return user.id


def _as(client: TestClient, user_id: str) -> None:
    client.app.dependency_overrides[current_active_user] = lambda: SimpleNamespace(id=user_id)


@pytest.fixture()
def two_users() -> tuple[str, str]:
    # conftest.py shares one sqlite file across the whole test session, so
    # a fixed email would collide across these tests' separate runs -
    # unique per invocation, same as pytest's own test isolation would
    # otherwise provide if each test got a fresh database.
    suffix = uuid.uuid4().hex[:8]
    alice_id = asyncio.run(_create_user(f"alice-{suffix}@isolation-test.local"))
    bob_id = asyncio.run(_create_user(f"bob-{suffix}@isolation-test.local"))
    return alice_id, bob_id


async def _create_shared_job(title: str = "Isolation Test Job") -> str:
    async with get_session() as session:
        job = Job(title=title, company="Acme", location="Remote", description="desc", source="mock")
        session.add(job)
        await session.flush()
        return job.id


def test_user_a_cannot_see_user_bs_application(client: TestClient, two_users: tuple[str, str]) -> None:
    alice_id, bob_id = two_users
    job_id = asyncio.run(_create_shared_job())

    _as(client, bob_id)
    created = client.post("/applications", json={"job_id": job_id, "status": "interested"})
    assert created.status_code == 201
    app_id = created.json()["id"]

    # Bob can see his own application.
    assert client.get(f"/applications/{app_id}").status_code == 200
    assert any(a["id"] == app_id for a in client.get("/applications").json())

    # Alice cannot - same 404 as a nonexistent id, not a 403 (no enumeration signal).
    _as(client, alice_id)
    assert client.get(f"/applications/{app_id}").status_code == 404
    assert not any(a["id"] == app_id for a in client.get("/applications").json())
    assert client.patch(f"/applications/{app_id}", json={"notes": "hijacked"}).status_code == 404
    assert (
        client.post(f"/applications/{app_id}/interviews", json={"stage": "phone screen"}).status_code
        == 404
    )
    assert client.get(f"/applications/{app_id}/interviews").status_code == 404
    assert client.post(f"/applications/{app_id}/feedback", json={"outcome": "rejected"}).status_code == 404
    assert client.get(f"/applications/{app_id}/feedback").status_code == 404

    # Alice's own application list/kanban stay empty - Bob's row isn't "shared".
    assert client.get("/applications").json() == []
    kanban = client.get("/applications/kanban").json()["columns"]
    assert all(col == [] for col in kanban.values())


def test_user_a_cannot_see_user_bs_cv(client: TestClient, two_users: tuple[str, str]) -> None:
    alice_id, bob_id = two_users

    async def _create_cv() -> str:
        async with get_session() as session:
            cv = CV(
                user_id=bob_id,
                raw_text="Bob's CV content",
                content_hash="deadbeef",
                file_type="text",
                consent_given_at=datetime.utcnow(),
            )
            session.add(cv)
            await session.flush()
            return cv.id

    cv_id = asyncio.run(_create_cv())

    _as(client, bob_id)
    assert client.get(f"/cv/{cv_id}").status_code == 200

    _as(client, alice_id)
    assert client.get(f"/cv/{cv_id}").status_code == 404


def test_user_a_cannot_see_user_bs_match_results(client: TestClient, two_users: tuple[str, str]) -> None:
    alice_id, bob_id = two_users
    job_id = asyncio.run(_create_shared_job("Match Isolation Job"))

    async def _create_match_result() -> None:
        async with get_session() as session:
            session.add(MatchResult(user_id=bob_id, job_id=job_id, score_final=0.9))

    asyncio.run(_create_match_result())

    _as(client, bob_id)
    bob_results = client.get("/match-results").json()
    assert any(r["job_id"] == job_id for r in bob_results)

    _as(client, alice_id)
    alice_results = client.get("/match-results").json()
    assert not any(r["job_id"] == job_id for r in alice_results)
