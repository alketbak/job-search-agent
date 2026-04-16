"""Adzuna job search — free tier with app_id + app_key.

Docs: https://developer.adzuna.com/docs/search
Endpoint pattern: /v1/api/jobs/{country}/search/{page}
We target Germany (de) per the spec.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from config import settings
from sources.base import JobSource

log = logging.getLogger(__name__)

BASE_URL = "https://api.adzuna.com/v1/api/jobs/de/search/1"
TIMEOUT = httpx.Timeout(15.0, connect=5.0)


class AdzunaSource(JobSource):
    name = "adzuna"

    def __init__(
        self, app_id: str | None = None, app_key: str | None = None
    ) -> None:
        self.app_id = app_id or settings.adzuna_app_id
        self.app_key = app_key or settings.adzuna_app_key

    def enabled(self) -> bool:
        return bool(self.app_id and self.app_key)

    async def fetch(self, query: str, location: str, limit: int) -> list[dict[str, Any]]:
        if not self.enabled():
            log.info("Adzuna skipped — credentials not set")
            return []

        params = {
            "app_id": self.app_id,
            "app_key": self.app_key,
            "what": query or "",
            "where": location or "",
            "results_per_page": min(max(limit, 1), 50),
            "content-type": "application/json",
        }
        try:
            async with httpx.AsyncClient(timeout=TIMEOUT) as client:
                r = await client.get(BASE_URL, params=params)
                if r.status_code != 200:
                    log.warning("Adzuna HTTP %s: %s", r.status_code, r.text[:200])
                    return []
                payload = r.json() or {}
        except (httpx.HTTPError, ValueError) as e:
            log.error("Adzuna fetch failed: %s", e)
            return []

        items = payload.get("results", []) or []
        return [self._normalize(i) for i in items[:limit]]

    def _normalize(self, item: dict[str, Any]) -> dict[str, Any]:
        company = (item.get("company") or {}).get("display_name", "") or ""
        loc_obj = item.get("location") or {}
        location = loc_obj.get("display_name", "") or ""
        salary_min = item.get("salary_min")
        salary_max = item.get("salary_max")
        return self.validate(
            {
                "title": item.get("title", "") or "",
                "company": company,
                "location": location,
                "description": item.get("description", "") or "",
                "url": item.get("redirect_url", "") or "",
                "salary_min": salary_min,
                "salary_max": salary_max,
                "contract_type": item.get("contract_type"),
            }
        )
