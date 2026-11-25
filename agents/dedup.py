"""Job deduplication - canonical URL -> normalized triple -> embedding
similarity above a configurable threshold.

Wired into `agents/orchestrator.py::dedupe_node` so repeated pipeline runs
update existing job rows instead of accumulating duplicates - a gap flagged
in the v1 audit (a comment claimed "the UI layer" deduped across runs; it
didn't).
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Iterable
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from database.models import Job, JobEmbedding, job_visible_to

log = logging.getLogger(__name__)


def _to_float_list(embedding: Iterable[float]) -> list[float]:
    """Normalize numpy arrays or plain lists to a JSON/pgvector-safe
    list[float] - state passed through LangGraph is already a plain list
    (see agents/retrieval.py), but direct callers (tests) may pass ndarray."""
    return [float(x) for x in embedding]


def canonical_url(url: str | None) -> str | None:
    url = (url or "").strip()
    return url or None


def content_hash(title: str, company: str, location: str) -> str:
    """Hash of the normalized (title, company, location) triple - the
    second dedup stage, used when two sources post the same role without
    sharing a URL."""
    norm = re.sub(r"\s+", " ", f"{title}|{company}|{location}".strip().lower())
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


async def _find_by_embedding(
    session: AsyncSession, embedding: list[float], owner_user_id: str | None = None
) -> Job | None:
    """Third dedup stage: an existing job whose description embedding is
    within `settings.job_dedup_embedding_threshold` cosine similarity.

    Returns None (rather than raising) if the embedding backend can't run
    the query - e.g. the SQLite JSON fallback in tests has no `<=>`
    operator. Dedup degrades to the first two (exact) stages in that case.
    """
    max_distance = 1.0 - settings.job_dedup_embedding_threshold
    stmt = (
        select(Job)
        .join(JobEmbedding, JobEmbedding.job_id == Job.id)
        .where(JobEmbedding.embedding.cosine_distance(embedding) <= max_distance)
        .where(job_visible_to(owner_user_id))
        .order_by(JobEmbedding.embedding.cosine_distance(embedding))
        .limit(1)
    )
    try:
        return (await session.execute(stmt)).scalars().first()
    except Exception as e:
        log.debug("Embedding-similarity dedup stage unavailable (%s) - skipping", e)
        return None


async def find_duplicate(
    session: AsyncSession,
    job: dict[str, Any],
    embedding: Iterable[float] | None = None,
    owner_user_id: str | None = None,
) -> Job | None:
    """Return the existing `Job` row that duplicates `job`, or None.

    Only shared jobs are candidates, plus `owner_user_id`'s own private
    jobs when given (a user adding a job by hand). The pipeline passes
    None, so a scraped job is never merged into (and never overwrites) a
    job someone added privately.

    Stages, cheapest/most-precise first:
      1. canonical URL match
      2. normalized (title, company, location) hash match
      3. embedding cosine similarity above the configured threshold
    """
    url = canonical_url(job.get("url"))
    if url:
        stmt = select(Job).where(Job.canonical_url == url, job_visible_to(owner_user_id))
        existing = (await session.execute(stmt)).scalars().first()
        if existing:
            return existing

    c_hash = content_hash(job.get("title", ""), job.get("company", ""), job.get("location", ""))
    stmt = select(Job).where(Job.content_hash == c_hash, job_visible_to(owner_user_id))
    existing = (await session.execute(stmt)).scalars().first()
    if existing:
        return existing

    if embedding is None:
        return None
    return await _find_by_embedding(session, _to_float_list(embedding), owner_user_id)
