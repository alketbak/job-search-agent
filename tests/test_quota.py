"""Tests for observability/quota.py (per-user daily LLM quota).

Uses the real local Redis (REDIS_URL, started via `docker compose up -d
redis`) - skips, not fails, if it's unreachable, same pattern as the
pgvector-only tests in tests/pg_helpers.py. The fail-open behavior when
Redis is down is tested separately by actually breaking the connection,
not by mocking - it's the one case we WANT to see fail without a real
Redis at all.
"""

from __future__ import annotations

import uuid

import pytest

from observability import quota


async def _redis_reachable() -> bool:
    try:
        client = quota._client()
        await client.ping()
        return True
    except Exception:
        return False


@pytest.fixture(autouse=True)
async def _skip_if_no_redis() -> None:
    # redis-asyncio's connection pool binds to whatever event loop is
    # running when the client is first used - pytest-asyncio gives each
    # async test its own fresh loop by default, so the module-level
    # cached client from a PRIOR test is bound to a now-closed loop here.
    # Reset it so this test's reachability check (and the test itself)
    # gets a client bound to ITS OWN current loop, not a stale one.
    quota._redis = None
    if not await _redis_reachable():
        pytest.skip("Redis not reachable - see docker compose up -d redis")


@pytest.mark.asyncio
async def test_check_and_increment_counts_up() -> None:
    user_id = f"quota-test-{uuid.uuid4().hex[:8]}"
    assert await quota.check_and_increment(user_id) == 1
    assert await quota.check_and_increment(user_id) == 2
    assert await quota.current_usage(user_id) == 2


@pytest.mark.asyncio
async def test_check_and_increment_raises_once_over_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    from config import settings

    object.__setattr__(settings, "llm_daily_quota_per_user", 2)
    try:
        user_id = f"quota-test-{uuid.uuid4().hex[:8]}"
        await quota.check_and_increment(user_id)
        await quota.check_and_increment(user_id)
        with pytest.raises(quota.LLMQuotaExceeded) as exc_info:
            await quota.check_and_increment(user_id)
        assert exc_info.value.user_id == user_id
        assert exc_info.value.limit == 2
    finally:
        object.__setattr__(settings, "llm_daily_quota_per_user", 50)


@pytest.mark.asyncio
async def test_two_users_have_independent_quotas() -> None:
    alice = f"quota-alice-{uuid.uuid4().hex[:8]}"
    bob = f"quota-bob-{uuid.uuid4().hex[:8]}"
    await quota.check_and_increment(alice)
    assert await quota.current_usage(bob) == 0


@pytest.mark.asyncio
async def test_fails_open_when_redis_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    """The one case tested with a deliberately broken connection, not the
    real local Redis - this is specifically testing what happens when
    Redis is NOT there."""
    from redis.asyncio import Redis

    broken_client = Redis.from_url("redis://localhost:1/0", socket_connect_timeout=1)
    monkeypatch.setattr(quota, "_redis", broken_client)

    user_id = f"quota-broken-{uuid.uuid4().hex[:8]}"
    count = await quota.check_and_increment(user_id)  # must not raise
    assert count == 0
    assert await quota.current_usage(user_id) == 0  # must not raise either
