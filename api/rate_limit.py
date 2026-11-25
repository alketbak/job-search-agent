"""Rate limiting for the upload and matching endpoints.

Uses the `limits` library directly (`FixedWindowRateLimiter` + its Redis
storage backend), not its FastAPI wrapper `slowapi` - slowapi's
`@limiter.limit(...)` decorator is incompatible with `from __future__
import annotations` (used throughout this codebase): wrapping a route
function with it breaks FastAPI/Pydantic's forward-ref resolution for
that function's own type-hinted parameters/return type, reproduced
independently of anything else in this file
(`pydantic.errors.PydanticUndefinedAnnotation` at route-registration
time). Calling `limits` directly as a plain FastAPI dependency sidesteps
the incompatibility entirely - no decorator, no wrapping.

IP-based, not per-user - deliberately a different concern from
`observability/quota.py`'s per-user *daily* LLM call quota. Rate limiting
bounds burst request rate (abuse/brute-force protection, independent of
whether the caller is even authenticated), quota bounds total daily usage
per account. Backed by the same Redis instance as the quota counter and
arq (one piece of infra, multiple uses).
"""

from __future__ import annotations

import logging

from fastapi import HTTPException, Request
from limits import RateLimitItem, parse
from limits.storage import storage_from_string
from limits.strategies import FixedWindowRateLimiter

from config import settings

log = logging.getLogger(__name__)

_storage = storage_from_string(settings.redis_url)
_strategy = FixedWindowRateLimiter(_storage)


def _client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def rate_limiter(limit: str, action: str):
    """Returns a FastAPI dependency enforcing `limit` (e.g. `"5/minute"`)
    per client IP for the given `action` name (just a Redis key
    namespace - lets the same IP have independent budgets for upload vs.
    matching, rather than one shared counter)."""
    item: RateLimitItem = parse(limit)

    async def _check(request: Request) -> None:
        if not settings.rate_limiting_enabled:
            return
        key = f"{action}:{_client_ip(request)}"
        try:
            allowed = _strategy.hit(item, key)
        except Exception as e:
            # Fails OPEN, not closed - same reasoning as
            # observability/quota.py: a Redis outage degrading rate-limit
            # *enforcement* is acceptable; a Redis outage taking down CV
            # uploads/matching entirely is not.
            log.warning("Rate limit check failed (%s) - allowing the request", type(e).__name__)
            return
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded: {limit} for {action}. Try again shortly.",
            )

    return _check
