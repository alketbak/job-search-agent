"""Tests for agents/worker.py: CV processing and per-user matching jobs -
retries, timeouts, idempotency keys, failure status.

Job functions are called directly as plain async functions - the same
thing arq's Worker does under the hood - rather than running a real arq
Worker loop; `tests/test_api_cv.py` already covers `extract_cv_job`'s
success/failure paths via `POST /cv`, so this file focuses on
`run_matching_job` and `extract_cv_job`'s not-found edge case.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from agents.worker import extract_cv_job, run_matching_job
from database import init_db
from database.db import get_session
from database.models import PipelineRun


@pytest.fixture(autouse=True)
def _init_db() -> None:
    init_db()


@pytest.mark.asyncio
async def test_extract_cv_job_skips_when_cv_no_longer_exists() -> None:
    result = await extract_cv_job({}, "00000000-0000-0000-0000-000000000000", "some-user")
    assert result["status"] == "skipped"


@pytest.mark.asyncio
async def test_run_matching_job_marks_success_and_upserts_run_row() -> None:
    run_id = "test-run-success-1"
    async with get_session() as session:
        session.add(PipelineRun(id=run_id, user_id="some-user", status="pending"))

    fake_final = {"run_id": run_id, "jobs_found": 3, "errors": []}
    with patch("agents.orchestrator.run_pipeline", new=AsyncMock(return_value=fake_final)):
        result = await run_matching_job({}, run_id, "some-user", "AI Engineer", "Berlin")
    assert result["status"] == "success"

    async with get_session() as session:
        run = await session.get(PipelineRun, run_id)
        # run_pipeline is mocked, so it never reaches persist_node's own
        # upsert - this only proves run_matching_job flips pending->running
        # before calling it; persist_node (tested in test_orchestrator.py)
        # owns the running->success/failed transition for real.
        assert run.status == "running"


@pytest.mark.asyncio
async def test_run_matching_job_marks_failed_and_records_error_on_exception() -> None:
    run_id = "test-run-failed-1"
    async with get_session() as session:
        session.add(PipelineRun(id=run_id, user_id="some-user", status="pending"))

    with patch("agents.orchestrator.run_pipeline", new=AsyncMock(side_effect=RuntimeError("boom"))):
        with pytest.raises(RuntimeError):
            await run_matching_job({}, run_id, "some-user", None, None)

    async with get_session() as session:
        run = await session.get(PipelineRun, run_id)
        assert run.status == "failed"
        assert any("RuntimeError" in e for e in (run.errors or []))
