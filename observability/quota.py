"""Per-user daily LLM quota.

Redis-backed, not DB-backed - one simple counter per (user, UTC date),
incremented atomically and self-expiring. Deliberately a *call-count*
cap, not a dollar cap: `ANTHROPIC_PRICING_JSON` (observability/tracing.py)
is often left unset on purpose (a cost we can't verify is never
guessed), so a dollar-denominated quota would either be fabricated
or silently inert. Counting real calls works regardless of whether
pricing is known - `estimate_cost` is still used for reporting *alongside*
the count, never for the enforcement decision itself.

Scope: only the LLM calls that are this user's own - CV
extraction (their CV) and scoring a single job on request (a job they
added, scored against their CV). Extraction of shared scraped jobs is NOT
charged: it's cached by content and benefits every user once done.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from redis.asyncio import Redis

from config import settings

log = logging.getLogger(__name__)

_redis: Redis | None = None

# Comfortably past a day, in case a process starts just before UTC
# midnight - the key still expires eventually, just not razor-precise.
_KEY_TTL_SECONDS = 60 * 60 * 48


def _client() -> Redis:
    global _redis
    if _redis is None:
        _redis = Redis.from_url(settings.redis_url, decode_responses=True)
    return _redis


def _key(user_id: str) -> str:
    day = datetime.now(UTC).strftime("%Y-%m-%d")
    return f"llm_quota:{user_id}:{day}"


class LLMQuotaExceeded(RuntimeError):
    """Raised by `check_and_increment` - callers turn this into a 429."""

    def __init__(self, user_id: str, count: int, limit: int) -> None:
        super().__init__(f"Daily LLM quota exceeded for user {user_id}: {count}/{limit}")
        self.user_id = user_id
        self.count = count
        self.limit = limit


async def current_usage(user_id: str) -> int:
    try:
        value = await _client().get(_key(user_id))
    except Exception as e:
        log.warning("LLM quota read failed (%s) - reporting 0", type(e).__name__)
        return 0
    return int(value) if value else 0


async def check_and_increment(user_id: str) -> int:
    """Raises `LLMQuotaExceeded` if the user is already at or over the
    daily limit; otherwise atomically increments and returns the new
    count. Call this BEFORE making the real LLM call - the point is to
    prevent the call that would exceed the quota, not just count it
    after the money's already spent.

    Fails OPEN, not closed, if Redis itself is unreachable - same
    pattern as memory/mcp_client.py and observability/tracing.py's
    graceful fallbacks. A transient Redis outage degrading quota
    *enforcement* is an acceptable tradeoff; a transient Redis outage
    taking down CV uploads/explanations entirely is not. Also keeps
    `pytest -q`'s fast path (SQLite only, no services) from requiring a real Redis just to test file upload.
    """
    key = _key(user_id)
    try:
        client = _client()
        count = await client.incr(key)
        if count == 1:
            await client.expire(key, _KEY_TTL_SECONDS)
    except Exception as e:
        log.warning("LLM quota check failed (%s) - allowing the call", type(e).__name__)
        return 0
    if count > settings.llm_daily_quota_per_user:
        raise LLMQuotaExceeded(user_id, count, settings.llm_daily_quota_per_user)
    return count
