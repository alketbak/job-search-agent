"""Per-account CV scoring in the pipeline.

A signed-in account is scored against the newest CV it uploaded - never
the operator's `CV_PATH` file - and a single hand-added job can be scored
on its own (`agents/orchestrator.py::score_job_for_user`). LLM calls are
mocked; embeddings run for real (same as tests/test_orchestrator.py).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy import select

from agents.orchestrator import extract_node, run_pipeline, score_job_for_user
from agents.retrieval import ensure_chunks_for_cv, get_cv_chunks
from database import init_db
from database.db import get_session
from database.models import (
    CV,
    CandidateProfile,
    CandidateSkill,
    CVChunk,
    Job,
    JobRequirement,
    MatchResult,
    PipelineRun,
    User,
)
from extraction.cv_extractor import load_uploaded_candidate_profile
from extraction.schemas import JobProfileExtraction, JobRequirementClaim


@pytest.fixture(autouse=True)
def _db() -> None:
    init_db()


async def _user() -> str:
    async with get_session() as session:
        user = User(email=f"scoring-{uuid.uuid4().hex[:8]}@per-user-test.local", hashed_password="!")
        session.add(user)
        await session.flush()
        return user.id


async def _uploaded_cv(
    user_id: str, text: str, skill: str | None = "Python", uploaded_at: datetime | None = None
) -> tuple[str, str | None]:
    """CV row, plus an extracted profile with one skill unless skill=None
    (simulating an upload whose extraction failed / is pending)."""
    async with get_session() as session:
        cv = CV(
            user_id=user_id,
            raw_text=text,
            content_hash=uuid.uuid4().hex,
            consent_given_at=datetime.utcnow(),
            uploaded_at=uploaded_at or datetime.utcnow(),
        )
        session.add(cv)
        await session.flush()
        if skill is None:
            return cv.id, None
        profile = CandidateProfile(user_id=user_id, cv_id=cv.id, seniority_target="senior")
        session.add(profile)
        await session.flush()
        session.add(
            CandidateSkill(
                candidate_profile_id=profile.id,
                skill_name=skill,
                category="language",
                evidence_level="professional",
                evidence_snippet=f"5 years {skill}",
            )
        )
        return cv.id, profile.id


async def _private_job(user_id: str) -> str:
    async with get_session() as session:
        job = Job(
            title="AI Engineer",
            company="Acme",
            location="Berlin",
            description="Python, Kubernetes",
            source="manual",
            owner_user_id=user_id,
        )
        session.add(job)
        await session.flush()
        return job.id


async def _fake_extract_job_profile(job_text: str, pipeline_run_id: str | None = None) -> JobProfileExtraction:
    return JobProfileExtraction(
        title="AI Engineer",
        seniority="senior",
        requirements=[
            JobRequirementClaim(text="Python", mandatory=True, category="skill"),
            JobRequirementClaim(text="Kubernetes", mandatory=False, category="skill"),
        ],
    )


@pytest.mark.asyncio
async def test_load_uploaded_profile_uses_newest_cv_with_a_profile() -> None:
    user_id = await _user()
    now = datetime.utcnow()
    _, old_profile = await _uploaded_cv(user_id, "old cv", skill="Java", uploaded_at=now - timedelta(days=2))
    new_cv, new_profile = await _uploaded_cv(user_id, "new cv", skill="Python", uploaded_at=now - timedelta(days=1))
    await _uploaded_cv(user_id, "newest, extraction failed", skill=None, uploaded_at=now)

    result = await load_uploaded_candidate_profile(user_id)
    assert result is not None
    profile, profile_id, cv_id = result
    assert (profile_id, cv_id) == (new_profile, new_cv)
    assert [s.name for s in profile.skills] == ["Python"]


@pytest.mark.asyncio
async def test_load_uploaded_profile_none_without_cv() -> None:
    assert await load_uploaded_candidate_profile(await _user()) is None


@pytest.mark.asyncio
async def test_signed_in_user_never_falls_back_to_cv_path_file() -> None:
    file_loader = AsyncMock()
    with patch("agents.orchestrator.ensure_candidate_profile_from_file", new=file_loader):
        result = await extract_node({"user_id": await _user(), "jobs": [], "errors": []})
    file_loader.assert_not_called()
    assert result["candidate_profile"] is None
    assert any("no_cv" in e for e in result["errors"])


@pytest.mark.asyncio
async def test_run_without_user_still_uses_cv_path_file() -> None:
    file_loader = AsyncMock(return_value=None)
    with patch("agents.orchestrator.ensure_candidate_profile_from_file", new=file_loader):
        await extract_node({"jobs": [], "errors": []})
    file_loader.assert_awaited_once()


@pytest.mark.asyncio
async def test_cv_chunks_are_scoped_to_one_cv() -> None:
    user_id = await _user()
    cv_a, _ = await _uploaded_cv(user_id, "First CV about Python and data pipelines.")
    cv_b, _ = await _uploaded_cv(user_id, "Second CV about Rust and embedded systems.")
    await ensure_chunks_for_cv(cv_a)
    await ensure_chunks_for_cv(cv_b)
    await ensure_chunks_for_cv(cv_a)  # idempotent

    a_chunks = await get_cv_chunks(user_id, cv_id=cv_a)
    assert a_chunks and all("Python" in text for text, _ in a_chunks)
    async with get_session() as session:
        rows = (await session.execute(select(CVChunk).where(CVChunk.cv_id == cv_a))).scalars().all()
    assert len(rows) == len(a_chunks)


@pytest.mark.asyncio
async def test_score_job_for_user_scores_against_their_uploaded_cv() -> None:
    user_id = await _user()
    cv_id, profile_id = await _uploaded_cv(user_id, "Senior engineer. 5 years Python in production.")
    job_id = await _private_job(user_id)
    run_id = str(uuid.uuid4())
    async with get_session() as session:
        session.add(PipelineRun(id=run_id, user_id=user_id, status="pending"))

    with (
        patch("agents.orchestrator.extract_job_profile", new=AsyncMock(side_effect=_fake_extract_job_profile)),
        patch("agents.orchestrator.generate_explanation", new=AsyncMock(return_value="Strong Python match.")),
        patch("agents.orchestrator.ensure_candidate_profile_from_file", new=AsyncMock()) as file_loader,
    ):
        final = await score_job_for_user(job_id, user_id, run_id)

    file_loader.assert_not_called()
    assert final["errors"] == []
    async with get_session() as session:
        match = (await session.execute(select(MatchResult).where(MatchResult.job_id == job_id))).scalars().one()
        reqs = (await session.execute(select(JobRequirement).where(JobRequirement.job_id == job_id))).scalars().all()
        run = await session.get(PipelineRun, run_id)
        chunks = (await session.execute(select(CVChunk).where(CVChunk.cv_id == cv_id))).scalars().all()
        job = await session.get(Job, job_id)

    assert match.user_id == user_id
    assert match.candidate_profile_id == profile_id
    assert match.score_final is not None and match.score_final > 0
    assert match.llm_explanation == "Strong Python match."
    assert {r.text for r in reqs} == {"Python", "Kubernetes"}
    assert run.status == "success"
    assert chunks  # the uploaded CV got embedded on first use
    assert job.owner_user_id == user_id  # persist kept it private


@pytest.mark.asyncio
async def test_run_for_user_without_cv_stores_no_scores() -> None:
    user_id = await _user()
    with (
        patch("agents.orchestrator.extract_job_profile", new=AsyncMock(side_effect=_fake_extract_job_profile)),
        patch("agents.orchestrator.generate_explanation", new=AsyncMock(return_value="x")) as explain,
        patch("agents.delivery_agent.generate_motivation", new=AsyncMock(return_value="x")),
    ):
        final = await run_pipeline(query="AI Engineer", location="Munich", user_id=user_id)

    explain.assert_not_called()
    assert final.get("jobs_scored") == 0
    async with get_session() as session:
        rows = (await session.execute(select(MatchResult).where(MatchResult.user_id == user_id))).scalars().all()
    assert rows == []
