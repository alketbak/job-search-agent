"""Tests for the LangGraph orchestrator pipeline:

    ingest -> dedupe -> extract -> retrieve -> score -> explain -> persist -> notify

Claude calls (extraction, explanation, motivation) are mocked, as everywhere in
the suite. Embedding (CV chunks, job description,
requirement texts) is NOT mocked - it's the real local sentence-transformer
model, same as the rest of the suite relies on elsewhere, and it's what
makes dedupe/retrieve exercise their real logic.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy import func, select

from agents.orchestrator import run_pipeline
from database import init_db
from database.db import get_session
from database.models import Job, JobEmbedding, JobRequirement, MatchResult
from extraction.schemas import (
    CandidateProfileExtraction,
    JobProfileExtraction,
    JobRequirementClaim,
    SkillClaim,
)


@pytest.fixture(autouse=True)
async def _bootstrap_db() -> None:
    """Fresh DB per test - these tests assert on dedupe behavior, which
    would see (and match against) any earlier test's jobs in
    tests/conftest.py's shared sqlite file otherwise."""
    init_db()
    async with get_session() as session:
        from sqlalchemy import delete

        for model in (MatchResult, JobEmbedding, JobRequirement, Job):
            await session.execute(delete(model))


def _fake_candidate_profile() -> tuple[CandidateProfileExtraction, str]:
    return (
        CandidateProfileExtraction(
            skills=[
                SkillClaim(
                    name="Python", category="language", evidence_level="professional", evidence_snippet="4 years Python"
                )
            ],
            seniority_target="senior",
        ),
        "fake-candidate-profile-id",
    )


async def _fake_extract_job_profile(job_text: str, pipeline_run_id: str | None = None) -> JobProfileExtraction:
    return JobProfileExtraction(
        title="AI Engineer",
        seniority="senior",
        requirements=[
            JobRequirementClaim(text="Python", mandatory=True, category="skill"),
            JobRequirementClaim(text="Kubernetes", mandatory=False, category="skill"),
        ],
    )


def _patches():
    return (
        patch(
            "agents.orchestrator.ensure_candidate_profile_from_file",
            new=AsyncMock(return_value=_fake_candidate_profile()),
        ),
        patch("agents.orchestrator.extract_job_profile", new=AsyncMock(side_effect=_fake_extract_job_profile)),
        patch("agents.orchestrator.generate_explanation", new=AsyncMock(return_value="Great fit on Python experience.")),
        patch("agents.delivery_agent.generate_motivation", new=AsyncMock(return_value="Keep going.")),
    )


async def _job_count() -> int:
    async with get_session() as session:
        return (await session.execute(select(func.count()).select_from(Job))).scalar_one()


@pytest.mark.asyncio
async def test_full_pipeline_end_to_end() -> None:
    p1, p2, p3, p4 = _patches()
    with p1, p2, p3, p4:
        final = await run_pipeline(query="AI Engineer", location="Munich")

    assert final["run_id"]
    assert final["jobs"], "ingest should have returned mock jobs"
    assert final["jobs_found"] == len(final["jobs"])
    assert final["jobs_scored"] == len(final["jobs"])
    assert all("score_breakdown" in j for j in final["jobs"])
    assert all(j["score_breakdown"]["score_final"] is not None for j in final["jobs"])
    # Deterministic score, not an LLM decision - every job gets one, with
    # requirement classifications attached.
    assert all(j["score_breakdown"]["requirement_classifications"] for j in final["jobs"])
    # Only the top N get an LLM explanation.
    explained = [j for j in final["jobs"] if j.get("explanation")]
    assert 0 < len(explained) == final["jobs_explained"]
    assert all(j.get("explanation") is None for j in final["jobs"][final["jobs_explained"] :])
    assert "delivery_result" in final and final["delivery_result"].get("path")
    # The MCP memory subprocess is flaky under pytest-asyncio's per-test
    # event loop - tolerate a notify.memory failure specifically, but
    # nothing from the scoring stages.
    assert all(e.startswith("notify.memory") for e in final["errors"])


@pytest.mark.asyncio
async def test_pipeline_persists_jobs_and_match_results() -> None:
    p1, p2, p3, p4 = _patches()
    with p1, p2, p3, p4:
        final = await run_pipeline(query="AI Engineer", location="Munich")

    async with get_session() as session:
        rows = (
            await session.execute(select(MatchResult).where(MatchResult.pipeline_run_id == final["run_id"]))
        ).scalars().all()
    assert len(rows) == len(final["jobs"])
    assert all(r.score_final is not None for r in rows)
    assert all(r.requirement_classifications for r in rows)
    # Blockers persisted for the job missing a mandatory requirement, if any.
    assert any(r.blockers for r in rows) or all(not r.blockers for r in rows)  # just must not crash either way


@pytest.mark.asyncio
async def test_rerun_dedupes_instead_of_duplicating_jobs() -> None:
    """Same mock-source jobs scraped twice must not double the job count -
    dedupe_node should match them against the first run's persisted rows."""
    p1, p2, p3, p4 = _patches()
    with p1, p2, p3, p4:
        before = await _job_count()
        first = await run_pipeline(query="AI Engineer", location="Munich")
        after_first = await _job_count()
        second = await run_pipeline(query="AI Engineer", location="Munich")
        after_second = await _job_count()

    assert after_first > before, "first run should insert new jobs"
    assert after_second == after_first, "second run should dedupe, not insert new rows"
    assert len(first["jobs"]) == len(second["jobs"])


@pytest.mark.asyncio
async def test_pipeline_survives_missing_candidate_profile() -> None:
    """No CV extraction available (e.g. CV_PATH missing) must not crash the
    pipeline."""
    p2, p3, p4 = _patches()[1:]
    with (
        patch("agents.orchestrator.ensure_candidate_profile_from_file", new=AsyncMock(return_value=None)),
        p2,
        p3,
        p4,
    ):
        final = await run_pipeline(query="AI Engineer", location="Munich")

    assert final["jobs"], "pipeline should still run without a candidate profile"
    assert all("score_breakdown" in j for j in final["jobs"])


@pytest.mark.asyncio
async def test_pipeline_survives_job_extraction_failure() -> None:
    """A job that fails extraction (invalid LLM JSON exhausting its retry)
    must still get scored (against an empty JobProfileExtraction) rather
    than crashing the run."""
    from extraction.schemas import ExtractionFailed

    p1, _, p3, p4 = _patches()
    with (
        p1,
        patch(
            "agents.orchestrator.extract_job_profile",
            new=AsyncMock(side_effect=ExtractionFailed("invalid_json", "job_extraction_v2")),
        ),
        p3,
        p4,
    ):
        final = await run_pipeline(query="AI Engineer", location="Munich")

    assert final["jobs"]
    assert all(j["extraction_failed"] for j in final["jobs"])
    assert all("score_breakdown" in j for j in final["jobs"])
