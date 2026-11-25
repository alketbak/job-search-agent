"""Bundesagentur für Arbeit - official German government jobs API.

Endpoint: https://rest.arbeitsagentur.de/jobboerse/jobsuche-service/pc/v4/jobs

Auth: a static client-id header is required for public access. The value
below is the widely-documented public demo id used by the official Jobboerse
mobile app; if the API changes its auth model, set JOBBOERSE_CLIENT_ID in env.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

from sources.base import JobSource

log = logging.getLogger(__name__)

BASE_URL = "https://rest.arbeitsagentur.de/jobboerse/jobsuche-service/pc/v4/jobs"
TIMEOUT = httpx.Timeout(15.0, connect=5.0)
# Public client-id documented in the API portal for unauthenticated access.
DEFAULT_CLIENT_ID = "jobboerse-jobsuche"


class BundesagenturSource(JobSource):
    name = "bundesagentur"

    def __init__(self, client_id: str | None = None) -> None:
        self.client_id = client_id or os.getenv("JOBBOERSE_CLIENT_ID", DEFAULT_CLIENT_ID)

    async def fetch(self, query: str, location: str, limit: int) -> list[dict[str, Any]]:
        params: dict[str, Any] = {
            "was": query or "",
            "wo": location or "",
            "angebotsart": 1,  # 1 = Arbeit (regular jobs), per API docs
            "size": min(max(limit, 1), 50),
            "page": 1,
        }
        headers = {"X-API-Key": self.client_id, "Accept": "application/json"}

        try:
            async with httpx.AsyncClient(timeout=TIMEOUT) as client:
                r = await client.get(BASE_URL, params=params, headers=headers)
                if r.status_code != 200:
                    log.warning(
                        "Bundesagentur HTTP %s: %s", r.status_code, r.text[:200]
                    )
                    return []
                payload = r.json() or {}
        except (httpx.HTTPError, ValueError) as e:
            log.error("Bundesagentur fetch failed: %s", e)
            return []

        items = payload.get("stellenangebote") or payload.get("jobs") or []
        return [self._normalize(i) for i in items[:limit]]

    def _normalize(self, item: dict[str, Any]) -> dict[str, Any]:
        arbeitsort = item.get("arbeitsort") or {}
        location = ", ".join(
            filter(None, [arbeitsort.get("ort"), arbeitsort.get("region")])
        )
        ref = item.get("refnr") or item.get("hashId") or ""
        url = f"https://www.arbeitsagentur.de/jobsuche/jobdetail/{ref}" if ref else ""
        return self.validate(
            {
                "title": item.get("titel", "") or item.get("beruf", "") or "",
                "company": item.get("arbeitgeber", "") or "",
                "location": location,
                "description": item.get("stellenbeschreibung", "") or "",
                "url": url,
                "aktuelleVeroeffentlichungsdatum": item.get(
                    "aktuelleVeroeffentlichungsdatum"
                ),
            }
        )
