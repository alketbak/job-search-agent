"""Scraper agent — fan out to all enabled sources concurrently, dedupe.

Dedup key: (title + company), lowercased and whitespace-collapsed. This is
intentionally simple — different boards post the same role with slightly
different titles, and collapsing on just company can merge distinct roles.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections import Counter
from typing import Any

from config import settings
from sources import get_enabled_sources
from sources.base import JobSource

log = logging.getLogger(__name__)

MAX_TOTAL_JOBS = 50


def _dedup_key(job: dict[str, Any]) -> str:
    """Stable dedup key based on title + company."""
    title = re.sub(r"\s+", " ", (job.get("title") or "")).strip().lower()
    company = re.sub(r"\s+", " ", (job.get("company") or "")).strip().lower()
    return f"{title}::{company}"


async def _fetch_one(
    source: JobSource, query: str, location: str, limit: int
) -> tuple[str, list[dict[str, Any]]]:
    """Fetch from one source. Catches all exceptions to keep the gather alive."""
    try:
        jobs = await source.fetch(query=query, location=location, limit=limit)
        return source.name, jobs
    except Exception as e:
        # Sources should swallow their own errors, but belt-and-braces.
        log.exception("Source %s raised during fetch: %s", source.name, e)
        return source.name, []


async def scrape(
    query: str,
    location: str,
    sources: list[JobSource] | None = None,
) -> list[dict[str, Any]]:
    """Scrape all enabled sources concurrently. Returns deduped, truncated list."""
    sources = sources or get_enabled_sources()
    if not sources:
        log.warning("No sources enabled — returning empty job list")
        return []

    per_source_limit = settings.max_jobs_per_source
    results = await asyncio.gather(
        *(_fetch_one(s, query, location, per_source_limit) for s in sources)
    )

    # Flatten and dedupe.
    all_jobs: list[dict[str, Any]] = []
    seen: set[str] = set()
    breakdown: Counter = Counter()

    for src_name, jobs in results:
        breakdown[src_name] = len(jobs)
        for job in jobs:
            key = _dedup_key(job)
            if not key or key in seen:
                continue
            seen.add(key)
            all_jobs.append(job)

    log.info(
        "Scraper breakdown: %s — %s unique jobs after dedup (cap %s)",
        dict(breakdown),
        len(all_jobs),
        MAX_TOTAL_JOBS,
    )
    return all_jobs[:MAX_TOTAL_JOBS]
