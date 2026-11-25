"""Tests for agents/retrieval.py - the pgvector replacement for the old
FAISS-based agents/rag_agent.py.

Chunking is a pure function, tested without any DB. Scoring needs real
pgvector similarity search, so it runs against the dedicated Postgres test
database and skips (doesn't fail) if that's unreachable - see
tests/pg_helpers.py for why SQLite can't stand in here.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from agents.retrieval import _chunk_text, embed
from tests.pg_helpers import pg_available


def test_chunking_respects_overlap_and_size() -> None:
    text = " ".join(f"w{i}" for i in range(500))
    chunks = _chunk_text(text, chunk_tokens=100, overlap=20)
    assert len(chunks) >= 5
    assert all(len(c.split()) <= 100 for c in chunks)
    a = chunks[0].split()
    b = chunks[1].split()
    assert a[-20:] == b[:20]


def test_chunking_empty_text() -> None:
    assert _chunk_text("", chunk_tokens=100, overlap=20) == []


def test_embed_empty_input_returns_empty_array() -> None:
    vecs = embed([])
    assert vecs.shape == (0, 768)


@pytest.mark.asyncio
async def test_max_similarity_to_cv_against_real_postgres() -> None:
    """Exercises the actual pgvector query (`_max_similarity_to_cv`) that
    replaced FAISS's `idx.search` - real CV chunk embeddings, real `<=>`
    cosine-distance search, real ordering assertion.

    Goes straight at the DB-facing function with our own Postgres test
    session rather than through `score_jobs`/`get_session()`, because
    `database/db.py` builds its engine from `settings.database_url` once at
    import time (bound to the sqlite test DB by `tests/conftest.py`) -
    monkeypatching `settings` later wouldn't redirect an already-built
    engine. `main.py` against the real `postgres` compose service is the
    end-to-end check for the full `score_jobs` path.
    """
    if not await pg_available():
        pytest.skip("jobradar_test Postgres not reachable - see tests/pg_helpers.py")

    from database.models import CVChunk, User
    from tests.pg_helpers import pg_test_session, truncate_all

    await truncate_all()

    from agents.retrieval import _max_similarity_to_cv, embed

    async with pg_test_session() as session:
        user = User(email="default@jobradar.local", is_default=True, hashed_password="!")
        session.add(user)
        await session.flush()

        from database.models import CV

        cv = CV(
            user_id=user.id,
            raw_text="placeholder",
            content_hash="abc123",
            file_type="text",
            consent_given_at=datetime.utcnow(),
        )
        session.add(cv)
        await session.flush()

        cv_chunk_texts = [
            "Experienced AI engineer building LLM agent systems with LangGraph and Claude.",
            "Also enjoys long walks and gardening on weekends.",
        ]
        cv_vecs = embed(cv_chunk_texts, kind="passage")
        for idx, (text, vec) in enumerate(zip(cv_chunk_texts, cv_vecs)):
            session.add(
                CVChunk(
                    cv_id=cv.id,
                    user_id=user.id,
                    chunk_index=idx,
                    chunk_text=text,
                    embedding_model="test-model",
                    embedding=vec.tolist(),
                )
            )
        await session.flush()

        job_vecs = embed(
            [
                "Build production agent systems using LangGraph and Anthropic Claude.",
                "Pruning roses and weeding flower beds all day.",
            ],
            kind="passage",
        )
        llm_sim = await _max_similarity_to_cv(session, user.id, job_vecs[0])
        garden_sim = await _max_similarity_to_cv(session, user.id, job_vecs[1])

        assert -1.0 <= llm_sim <= 1.0
        assert -1.0 <= garden_sim <= 1.0
        assert llm_sim > garden_sim


@pytest.mark.asyncio
async def test_score_jobs_empty_input() -> None:
    from agents.retrieval import score_jobs

    assert await score_jobs([]) == []
