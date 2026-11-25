"""Test helpers for pgvector-dependent code.

`tests/conftest.py` forces `DATABASE_URL` to a temp SQLite file so most of
the suite stays fast and dependency-free. But `.with_variant(JSON(), "sqlite")` means
SQLite can't actually run `<=>` similarity queries - anything testing real
pgvector behavior (agents/retrieval.py, agents/dedup.py's embedding stage)
needs a real Postgres. These tests connect directly to a dedicated test
database instead of `settings.database_url`, and skip (not fail) if it's
unreachable - that's the "DB unavailable" case, reported explicitly
rather than silently.

One-time local setup:
    docker compose up -d postgres
    docker exec jobradar-postgres psql -U jobradar -d jobradar -c \
        "CREATE DATABASE jobradar_test;"
    docker exec jobradar-postgres psql -U jobradar -d jobradar_test -c \
        "CREATE EXTENSION IF NOT EXISTS vector;"
    DATABASE_URL=postgresql+psycopg://jobradar:jobradar@localhost:5432/jobradar_test \
        alembic upgrade head
"""

from __future__ import annotations

import os

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

TEST_POSTGRES_URL = os.getenv(
    "TEST_DATABASE_URL", "postgresql+psycopg://jobradar:jobradar@localhost:5432/jobradar_test"
)

_engine = create_async_engine(TEST_POSTGRES_URL, future=True)
_session_factory: async_sessionmaker[AsyncSession] = async_sessionmaker(
    _engine, expire_on_commit=False, class_=AsyncSession
)


async def pg_available() -> bool:
    try:
        async with _engine.connect():
            return True
    except Exception:
        return False


def pg_test_session() -> AsyncSession:
    return _session_factory()


async def truncate_all() -> None:
    """Wipe all app tables between tests - cheap isolation for a test DB."""
    tables = [
        "feedback",
        "interviews",
        "match_results",
        "job_requirements",
        "job_embeddings",
        "applications",
        "jobs",
        "candidate_skills",
        "candidate_profiles",
        "cv_chunks",
        "cvs",
        "pipeline_runs",
        "skill_gaps",
        "users",
    ]
    async with _engine.begin() as conn:
        from sqlalchemy import text

        await conn.execute(text(f"TRUNCATE {', '.join(tables)} CASCADE"))
