"""Tests for agents/dedup.py - canonical URL -> normalized triple ->
embedding similarity."""

from __future__ import annotations

import numpy as np
import pytest

from agents.dedup import canonical_url, content_hash, find_duplicate
from database import Job, get_session, init_db
from database.seed import ensure_default_user
from tests.pg_helpers import pg_available, pg_test_session, truncate_all


@pytest.fixture(autouse=True)
def _bootstrap_db() -> None:
    init_db()


def test_canonical_url_blank_is_none() -> None:
    assert canonical_url(None) is None
    assert canonical_url("   ") is None
    assert canonical_url("https://x.example/job/1") == "https://x.example/job/1"


def test_content_hash_normalizes_whitespace_and_case() -> None:
    a = content_hash("Senior  AI  Engineer", "Acme  GmbH", "Munich")
    b = content_hash("senior ai engineer", "acme gmbh", "munich")
    assert a == b


def test_content_hash_differs_for_different_jobs() -> None:
    a = content_hash("AI Engineer", "Acme", "Munich")
    b = content_hash("AI Engineer", "Acme", "Berlin")
    assert a != b


@pytest.mark.asyncio
async def test_find_duplicate_by_canonical_url() -> None:
    await ensure_default_user()
    async with get_session() as session:
        existing = Job(
            title="AI Engineer",
            company="Acme",
            location="Munich",
            url="https://acme.example/jobs/1",
            canonical_url="https://acme.example/jobs/1",
            content_hash=content_hash("AI Engineer", "Acme", "Munich"),
            source="mock",
        )
        session.add(existing)
        await session.flush()

        candidate = {
            "title": "AI Engineer (edited title)",
            "company": "Acme",
            "location": "Munich",
            "url": "https://acme.example/jobs/1",
        }
        dup = await find_duplicate(session, candidate)
        assert dup is not None
        assert dup.id == existing.id


@pytest.mark.asyncio
async def test_find_duplicate_by_normalized_triple_when_no_url() -> None:
    # Distinct (title, company, location) from the canonical-url test above -
    # tests/conftest.py's temp SQLite file persists across the whole test
    # session, so colliding content_hash values here would make stage-2
    # matching ambiguous between tests.
    await ensure_default_user()
    async with get_session() as session:
        existing = Job(
            title="Platform Engineer",
            company="Beta Robotics",
            location="Berlin",
            url="",
            canonical_url=None,
            content_hash=content_hash("Platform Engineer", "Beta Robotics", "Berlin"),
            source="arbeitnow",
        )
        session.add(existing)
        await session.flush()

        candidate = {"title": "platform engineer", "company": "beta robotics", "location": "berlin", "url": ""}
        dup = await find_duplicate(session, candidate)
        assert dup is not None
        assert dup.id == existing.id


@pytest.mark.asyncio
async def test_find_duplicate_returns_none_when_no_match() -> None:
    await ensure_default_user()
    async with get_session() as session:
        candidate = {"title": "Totally New Role", "company": "Nobody Inc", "location": "Nowhere", "url": ""}
        assert await find_duplicate(session, candidate) is None


@pytest.mark.asyncio
async def test_find_duplicate_embedding_stage_degrades_on_sqlite() -> None:
    """The SQLite JSON fallback has no `<=>` operator - the embedding stage
    must degrade to "no match" rather than raising, so a backend failure
    never crashes the pipeline."""
    await ensure_default_user()
    async with get_session() as session:
        candidate = {"title": "Unrelated Role", "company": "X", "location": "Y", "url": ""}
        fake_embedding = np.zeros(768, dtype=np.float32)
        dup = await find_duplicate(session, candidate, embedding=fake_embedding)
        assert dup is None


@pytest.mark.asyncio
async def test_find_duplicate_embedding_stage_matches_on_postgres() -> None:
    if not await pg_available():
        pytest.skip("jobradar_test Postgres not reachable - see tests/pg_helpers.py")

    await truncate_all()
    from database.models import JobEmbedding, User

    async with pg_test_session() as session:
        user = User(email="default@jobradar.local", is_default=True, hashed_password="!")
        session.add(user)
        await session.flush()

        vec = np.random.default_rng(42).normal(size=768).astype(np.float32)
        vec = vec / np.linalg.norm(vec)

        existing = Job(
            title="Senior MLOps Engineer",
            company="Acme",
            location="Berlin",
            url="",
            content_hash=content_hash("Senior MLOps Engineer", "Acme", "Berlin"),
            source="adzuna",
        )
        session.add(existing)
        await session.flush()
        session.add(JobEmbedding(job_id=existing.id, embedding_model="test-model", embedding=vec.tolist()))
        await session.flush()

        # A near-identical vector (tiny perturbation) should still match -
        # simulates the same posting re-scraped with slightly different text.
        near_duplicate_vec = vec + np.random.default_rng(1).normal(scale=0.01, size=768).astype(np.float32)
        near_duplicate_vec = near_duplicate_vec / np.linalg.norm(near_duplicate_vec)

        candidate = {
            "title": "Senior MLOps Engineer (re-posted)",
            "company": "Different Listing Co",
            "location": "Berlin, DE",
            "url": "",
        }
        dup = await find_duplicate(session, candidate, embedding=near_duplicate_vec)
        assert dup is not None
        assert dup.id == existing.id

        # An unrelated random vector should NOT match.
        unrelated_vec = np.random.default_rng(99).normal(size=768).astype(np.float32)
        unrelated_vec = unrelated_vec / np.linalg.norm(unrelated_vec)
        candidate2 = {"title": "Totally Different Role", "company": "Other Co", "location": "Hamburg", "url": ""}
        assert await find_duplicate(session, candidate2, embedding=unrelated_vec) is None
