"""Seed data that must exist before anything user-scoped can be written.

Every user-owned table carries a `user_id`, so a default user must exist from the very first boot - not only after
running `scripts/migrate_sqlite_to_postgres.py`. `ensure_default_user` is
idempotent and safe to call on every startup.

`User` carries fastapi-users' auth columns
(`database/models.py`), including a NOT NULL `hashed_password`. The
default user is an internal bootstrap row, not a real account - nobody
should be able to log in as it - so it gets the same unguessable `'!'`
placeholder the auth migration backfilled onto the pre-existing row
(never a valid bcrypt hash, so it can't match any real password check).
"""

from __future__ import annotations

import logging

from sqlalchemy import select

from database.db import get_session
from database.models import User

log = logging.getLogger(__name__)

DEFAULT_USER_EMAIL = "default@jobradar.local"


async def ensure_default_user() -> str:
    """Return the default user's id, creating it on first call."""
    async with get_session() as session:
        existing = (
            await session.execute(select(User).where(User.is_default.is_(True)))
        ).scalar_one_or_none()
        if existing:
            return existing.id

        user = User(email=DEFAULT_USER_EMAIL, is_default=True, hashed_password="!")
        session.add(user)
        await session.flush()
        log.info("Created default user %s (%s)", user.id, user.email)
        return user.id
