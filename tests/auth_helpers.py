"""Shared test helper for auth-protected routes.

Every endpoint in api/router.py now requires `Depends(current_active_user)`.
Tests that exercise business logic, not the auth library itself, use
FastAPI's `dependency_overrides` to swap that dependency for a fake user -
the standard pattern for testing protected routes without going through a
real register/login/JWT round trip for every test, keeping tests
deterministic and fast. The fake user only needs `.id` - nothing in
api/router.py reads any other attribute off `current_user`.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from fastapi import FastAPI

from auth.users import current_active_user
from database.seed import ensure_default_user


def override_auth(app: FastAPI, user_id: str | None = None) -> None:
    """Make every protected route in `app` see a fake authenticated user.

    Without `user_id`, uses the real bootstrap default user's id (so
    `Application.user_id`-style foreign keys resolve against a real row).
    Pass a specific `user_id` to simulate a *different*, already-created
    user - e.g. to test that user A can't read user B's data.
    """

    async def _fake_current_user() -> Any:
        uid = user_id if user_id is not None else await ensure_default_user()
        return SimpleNamespace(id=uid)

    app.dependency_overrides[current_active_user] = _fake_current_user
