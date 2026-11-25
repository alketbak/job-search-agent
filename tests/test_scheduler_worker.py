"""Tests for scheduler_worker.py.

The process/signal-handling plumbing in `main()` isn't unit-testable in a
meaningful way (it was instead verified by actually running the process -
see docs/DEVELOPMENT.md: heartbeat file written
immediately, graceful SIGTERM shutdown confirmed via logs). These tests
cover the two units of real logic: the heartbeat write and the scheduled
job's failure isolation.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, patch

import pytest

from config import settings
from scheduler_worker import _scheduled_run, _write_heartbeat


def test_write_heartbeat_creates_parseable_timestamp_file(tmp_path) -> None:
    heartbeat_path = tmp_path / "nested" / "scheduler_heartbeat"
    original = settings.scheduler_heartbeat_file
    # `Settings` is a frozen dataclass (see config.py) - `patch.object`'s
    # plain `setattr` can't touch it; same `object.__setattr__` pattern
    # used to override frozen fields elsewhere in tests (e.g.
    # tests/test_demo_mode.py).
    object.__setattr__(settings, "scheduler_heartbeat_file", str(heartbeat_path))
    try:
        _write_heartbeat()
    finally:
        object.__setattr__(settings, "scheduler_heartbeat_file", original)

    assert heartbeat_path.exists()
    # Must round-trip through datetime.fromisoformat - the container
    # healthcheck in docker-compose.yml parses this via os.path.getmtime
    # instead, but scheduler_worker.py's own contract is an ISO timestamp.
    datetime.fromisoformat(heartbeat_path.read_text(encoding="utf-8"))


@pytest.mark.asyncio
async def test_scheduled_run_swallows_pipeline_failure() -> None:
    """A failed pipeline run must never crash the scheduler's event loop -
    APScheduler would otherwise silently drop the daily job from then on."""
    with patch("scheduler_worker.run_pipeline", new=AsyncMock(side_effect=RuntimeError("boom"))):
        await _scheduled_run()  # must not raise


@pytest.mark.asyncio
async def test_scheduled_run_calls_pipeline() -> None:
    mock_pipeline = AsyncMock(return_value={"jobs_found": 0})
    with patch("scheduler_worker.run_pipeline", new=mock_pipeline):
        await _scheduled_run()
    mock_pipeline.assert_awaited_once_with()
