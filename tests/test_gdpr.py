"""GDPR basics tests.

Covers consent-on-upload (the actual upload-rejection test lives in
tests/test_api_cv.py, next to the rest of the upload tests), data export,
and real account deletion.
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
from database.models import CV, Application, CandidateProfile, Job, MatchResult, User


@pytest.fixture()
def client() -> TestClient:
    init_db()
    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def seeded_user(client: TestClient) -> str:
    """A real user with one CV, one candidate profile, one application,
    and one match result - enough to exercise both export and deletion
    across every personal-data table (CVs, embeddings/profiles, matches)."""

    async def _seed() -> str:
        async with get_session() as session:
            user = User(email=f"gdpr-{uuid.uuid4().hex[:8]}@example.com", hashed_password="!")
            session.add(user)
            await session.flush()

            job = Job(title="GDPR Test Job", company="Acme", location="Remote", source="mock")
            session.add(job)

            cv = CV(
                user_id=user.id,
                raw_text="Some CV content",
                content_hash=uuid.uuid4().hex,
                file_type="text",
                consent_given_at=datetime.utcnow(),
            )
            session.add(cv)
            await session.flush()

            profile = CandidateProfile(user_id=user.id, cv_id=cv.id)
            session.add(profile)
            await session.flush()

            session.add(Application(user_id=user.id, job_id=job.id))
            session.add(MatchResult(user_id=user.id, job_id=job.id, score_final=0.75))

            return user.id

    user_id = asyncio.run(_seed())
    client.app.dependency_overrides[current_active_user] = lambda: SimpleNamespace(id=user_id)
    return user_id


def test_export_includes_everything_the_user_owns(client: TestClient, seeded_user: str) -> None:
    resp = client.get("/me/export")
    assert resp.status_code == 200
    body = resp.json()

    assert len(body["cvs"]) == 1
    assert len(body["candidate_profiles"]) == 1
    assert len(body["applications"]) == 1
    assert len(body["match_results"]) == 1
    assert body["exported_at"] is not None


def test_delete_me_removes_every_owned_row(client: TestClient, seeded_user: str) -> None:
    resp = client.delete("/me")
    assert resp.status_code == 200
    counts = resp.json()["deleted_counts"]
    assert counts["cvs"] == 1
    assert counts["candidate_profiles"] == 1
    assert counts["applications"] == 1
    assert counts["match_results"] == 1
    assert counts["user"] == 1
    assert "Langfuse" in resp.json()["langfuse_traces_note"]

    async def _row_counts() -> tuple[int, int, int, int, int]:
        async with get_session() as session:
            from sqlalchemy import select

            cvs = (await session.execute(select(CV).where(CV.user_id == seeded_user))).scalars().all()
            profiles = (
                (await session.execute(select(CandidateProfile).where(CandidateProfile.user_id == seeded_user)))
                .scalars()
                .all()
            )
            apps = (
                (await session.execute(select(Application).where(Application.user_id == seeded_user)))
                .scalars()
                .all()
            )
            matches = (
                (await session.execute(select(MatchResult).where(MatchResult.user_id == seeded_user)))
                .scalars()
                .all()
            )
            user_row = await session.get(User, seeded_user)
            return len(cvs), len(profiles), len(apps), len(matches), (1 if user_row else 0)

    cv_n, profile_n, app_n, match_n, user_n = asyncio.run(_row_counts())
    assert (cv_n, profile_n, app_n, match_n, user_n) == (0, 0, 0, 0, 0)
