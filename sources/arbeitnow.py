"""Arbeitnow job board - free, no auth, Germany-focused.

Docs: https://www.arbeitnow.com/api/job-board-api
The API returns results in `data`, with `meta.current_page` / `meta.last_page`
for pagination. We'll stop once we've gathered `limit` results or exhausted pages.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from sources.base import JobSource

log = logging.getLogger(__name__)

BASE_URL = "https://www.arbeitnow.com/api/job-board-api"
TIMEOUT = httpx.Timeout(15.0, connect=5.0)
MAX_PAGES = 5  # hard cap so a bad loop can't hammer the API


class ArbeitnowSource(JobSource):
    name = "arbeitnow"

    async def fetch(self, query: str, location: str, limit: int) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        q = (query or "").strip()
        loc = (location or "").strip()

        # Arbeitnow accepts ?search= as the free-text param; their `location`
        # filtering is limited, so we do a client-side substring match.
        try:
            async with httpx.AsyncClient(timeout=TIMEOUT) as client:
                for page in range(1, MAX_PAGES + 1):
                    params = {"search": q, "page": page} if q else {"page": page}
                    r = await client.get(BASE_URL, params=params)
                    if r.status_code != 200:
                        log.warning("Arbeitnow HTTP %s on page %s", r.status_code, page)
                        break
                    payload = r.json() or {}
                    data = payload.get("data", []) or []
                    if not data:
                        break
                    for item in data:
                        normalized = self._normalize(item)
                        if loc and loc.lower() not in (normalized["location"] or "").lower():
                            # Accept "remote" tag regardless of location filter.
                            if not item.get("remote"):
                                continue
                        results.append(normalized)
                        if len(results) >= limit:
                            return results
                    meta = payload.get("meta") or {}
                    if page >= (meta.get("last_page") or page):
                        break
        except (httpx.HTTPError, ValueError) as e:
            log.error("Arbeitnow fetch failed: %s", e)
            return results

        return results[:limit]

    def _normalize(self, item: dict[str, Any]) -> dict[str, Any]:
        """Map Arbeitnow fields onto our contract."""
        return self.validate(
            {
                "title": item.get("title", "") or "",
                "company": item.get("company_name", "") or "",
                "location": item.get("location", "") or "",
                "description": item.get("description", "") or "",
                "url": item.get("url", "") or "",
                # Pass-through extras - downstream may or may not use these.
                "tags": item.get("tags") or [],
                "remote": bool(item.get("remote")),
            }
        )
