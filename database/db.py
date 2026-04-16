"""Async SQLAlchemy engine + session factory, plus `init_db()`.

We expose both a sync engine (only used for `init_db()` / CREATE TABLE on startup)
and an async engine+session for runtime use. In 2.x you can do DDL on an async
engine too, but using the sync one keeps startup simple and avoids nested event
loops when the API server is booting.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from config import settings

log = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


# Sync engine — used only for `init_db()`. Not exposed for runtime queries.
_sync_engine = create_engine(settings.database_url, future=True)

# Async engine + session factory — the runtime primitives.
_async_engine = create_async_engine(settings.async_database_url, future=True, echo=False)
_async_session_factory: async_sessionmaker[AsyncSession] = async_sessionmaker(
    _async_engine, expire_on_commit=False, class_=AsyncSession
)


def init_db() -> None:
    """Create all tables. Safe to call on every startup — CREATE IF NOT EXISTS."""
    # Import models so SQLAlchemy registers them on Base.metadata before create_all.
    # Local import avoids a circular import at module load time.
    from database import models  # noqa: F401

    log.info("Initializing database at %s", settings.database_url)
    Base.metadata.create_all(_sync_engine)


@asynccontextmanager
async def get_session() -> AsyncIterator[AsyncSession]:
    """Async context manager yielding a session with auto-commit on exit.

    Usage:
        async with get_session() as session:
            session.add(obj)
    """
    session = _async_session_factory()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


async def dispose_engine() -> None:
    """Dispose the async engine — call on shutdown."""
    await _async_engine.dispose()
