"""Tests for the LangGraph orchestrator — runs the full pipeline with mocks."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from agents.orchestrator import run_pipeline
from database import init_db


@pytest.fixture(autouse=True)
def _bootstrap_db() -> None:
    """Create tables before each orchestrator test."""
    init_db()


@pytest.mark.asyncio
async def test_full_pipeline_with_mocked_analyst() -> None:
    """Pipeline must run end-to-end with mocks, producing scored + analyzed jobs."""

    # Return each input job with a deterministic Claude annotation.
    async def _fake_analyze(jobs):
        return [
            {
                **j,
                "claude_score": 7,
                "claude_reasoning": "Test reasoning.",
                "skill_gaps": ["Kubernetes"],
                "highlights": ["Python"],
                "apply_recommendation": True,
            }
            for j in jobs
        ]

    # We replace the Claude calls to keep tests offline + deterministic.
    with patch("agents.orchestrator.analyze_jobs", new=AsyncMock(side_effect=_fake_analyze)), patch(
        "agents.delivery_agent.generate_motivation", new=AsyncMock(return_value="Keep going.")
    ):
        final = await run_pipeline(query="AI Engineer", location="Munich")

    assert final["run_id"]
    assert final["raw_jobs"], "scraper should have returned mock jobs"
    assert final["scored_jobs"], "RAG agent should have produced scored_jobs"
    assert final["analyzed_jobs"], "analyst should have produced analyzed_jobs"
    assert all("raw_score" in j for j in final["scored_jobs"])
    assert all("claude_score" in j for j in final["analyzed_jobs"])
    assert "delivery_result" in final and final["delivery_result"].get("path"), (
        "digest markdown should be written"
    )
