"""Tests for api/rate_limit.py (rate limiting on the upload and
matching endpoints).

tests/conftest.py disables rate limiting globally (RATE_LIMITING_ENABLED=
false) so the rest of the suite isn't subject to a real, persistent-across-
runs Redis counter shared by every TestClient request (they all report the
same "testclient" IP - see config.py's `rate_limiting_enabled` docstring).
This file explicitly re-enables it to test the real boundary - skips, not
fails, if Redis is unreachable, same pattern as tests/test_quota.py.
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest

from api.rate_limit import _strategy, rate_limiter
from config import settings


async def _redis_reachable() -> bool:
    try:
        from observability import quota

        quota._redis = None
        client = quota._client()
        await client.ping()
        return True
    except Exception:
        return False


@pytest.fixture(autouse=True)
async def _enable_rate_limiting() -> None:
    if not await _redis_reachable():
        pytest.skip("Redis not reachable - see docker compose up -d redis")
    object.__setattr__(settings, "rate_limiting_enabled", True)
    try:
        yield
    finally:
        object.__setattr__(settings, "rate_limiting_enabled", False)


def _fake_request(ip: str) -> SimpleNamespace:
    return SimpleNamespace(client=SimpleNamespace(host=ip))


@pytest.mark.asyncio
async def test_allows_requests_under_the_limit() -> None:
    check = rate_limiter("3/minute", f"test-{uuid.uuid4().hex[:8]}")
    request = _fake_request(f"1.2.3.{uuid.uuid4().int % 255}")
    await check(request)
    await check(request)
    await check(request)  # 3rd is still within "3/minute"


@pytest.mark.asyncio
async def test_rejects_once_over_the_limit() -> None:
    from fastapi import HTTPException

    action = f"test-{uuid.uuid4().hex[:8]}"
    ip = f"1.2.3.{uuid.uuid4().int % 255}"
    check = rate_limiter("2/minute", action)
    request = _fake_request(ip)
    await check(request)
    await check(request)
    with pytest.raises(HTTPException) as exc_info:
        await check(request)
    assert exc_info.value.status_code == 429


@pytest.mark.asyncio
async def test_different_ips_have_independent_budgets() -> None:
    action = f"test-{uuid.uuid4().hex[:8]}"
    check = rate_limiter("1/minute", action)
    await check(_fake_request("10.0.0.1"))
    await check(_fake_request("10.0.0.2"))  # different IP, must not be blocked


@pytest.mark.asyncio
async def test_disabled_setting_bypasses_the_check_entirely() -> None:
    object.__setattr__(settings, "rate_limiting_enabled", False)
    try:
        action = f"test-{uuid.uuid4().hex[:8]}"
        check = rate_limiter("1/minute", action)
        ip = f"1.2.3.{uuid.uuid4().int % 255}"
        for _ in range(5):
            await check(_fake_request(ip))  # would 429 by the 2nd call if enabled
    finally:
        object.__setattr__(settings, "rate_limiting_enabled", True)


@pytest.mark.asyncio
async def test_fails_open_when_redis_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    def _broken_hit(*args, **kwargs):
        raise ConnectionError("simulated redis outage")

    monkeypatch.setattr(_strategy, "hit", _broken_hit)
    check = rate_limiter("1/minute", f"test-{uuid.uuid4().hex[:8]}")
    await check(_fake_request("1.2.3.4"))  # must not raise
