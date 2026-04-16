"""Tests for job sources — mock + arbeitnow (httpx mocked)."""

from __future__ import annotations

import pytest

from sources.arbeitnow import ArbeitnowSource
from sources.base import REQUIRED_FIELDS
from sources.mock import MockSource


@pytest.mark.asyncio
async def test_mock_source_returns_required_fields() -> None:
    src = MockSource()
    jobs = await src.fetch(query="AI", location="Munich", limit=5)
    assert len(jobs) > 0, "mock_jobs.json should produce at least one match"
    for job in jobs:
        for field in REQUIRED_FIELDS:
            assert field in job, f"missing field {field} in mock job"
        assert job["source"] == "mock"


@pytest.mark.asyncio
async def test_mock_source_respects_limit() -> None:
    src = MockSource()
    jobs = await src.fetch(query="", location="", limit=3)
    assert len(jobs) == 3


@pytest.mark.asyncio
async def test_mock_source_soft_filters_location() -> None:
    src = MockSource()
    # An obviously non-matching filter should still return *something* (fallback),
    # per our soft-filter policy — the demo must never produce zero jobs.
    jobs = await src.fetch(query="banana", location="narnia", limit=5)
    assert len(jobs) > 0


@pytest.mark.asyncio
async def test_arbeitnow_handles_http_error(httpx_mock) -> None:
    # httpx-mock's fixture intercepts real requests; we return 500 to ensure
    # the source degrades gracefully to [] instead of raising.
    httpx_mock.add_response(status_code=500, json={"data": []})
    src = ArbeitnowSource()
    jobs = await src.fetch(query="python", location="", limit=5)
    assert jobs == []


@pytest.mark.asyncio
async def test_arbeitnow_parses_response(httpx_mock) -> None:
    httpx_mock.add_response(
        json={
            "data": [
                {
                    "title": "Senior AI Engineer",
                    "company_name": "ExampleAI GmbH",
                    "location": "Munich",
                    "description": "Work on LLMs.",
                    "url": "https://example.com/job/1",
                    "tags": ["python"],
                    "remote": False,
                }
            ],
            "meta": {"current_page": 1, "last_page": 1},
        }
    )
    src = ArbeitnowSource()
    jobs = await src.fetch(query="AI", location="Munich", limit=5)
    assert len(jobs) == 1
    assert jobs[0]["title"] == "Senior AI Engineer"
    assert jobs[0]["company"] == "ExampleAI GmbH"
    assert jobs[0]["source"] == "arbeitnow"
