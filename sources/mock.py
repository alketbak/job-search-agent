"""Mock job source — reads from data/mock_jobs.json.

Used when USE_MOCK_DATA=true, and also as a fallback for test runs.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from sources.base import JobSource

log = logging.getLogger(__name__)

MOCK_FILE = Path("data/mock_jobs.json")


class MockSource(JobSource):
    name = "mock"

    def __init__(self, path: Path | str = MOCK_FILE) -> None:
        self.path = Path(path)

    async def fetch(self, query: str, location: str, limit: int) -> list[dict[str, Any]]:
        # File I/O is cheap, but we run it in a thread to stay fully async.
        try:
            data = await asyncio.to_thread(self._read_file)
        except FileNotFoundError:
            log.warning("Mock file not found at %s — returning empty list", self.path)
            return []
        except json.JSONDecodeError as e:
            log.error("Mock file %s is malformed JSON: %s", self.path, e)
            return []

        # Soft filter — if the caller passed a query/location, prefer matches,
        # but don't filter so aggressively that the demo returns nothing.
        q = (query or "").lower().strip()
        loc = (location or "").lower().strip()

        def _matches(job: dict[str, Any]) -> bool:
            if not q and not loc:
                return True
            text = f"{job.get('title','')} {job.get('company','')} {job.get('description','')}".lower()
            hit_q = q in text if q else True
            hit_loc = loc in (job.get("location", "") or "").lower() if loc else True
            return hit_q and hit_loc

        filtered = [j for j in data if _matches(j)] or data  # fall back to all
        return [self.validate(j) for j in filtered[:limit]]

    def _read_file(self) -> list[dict[str, Any]]:
        with self.path.open("r", encoding="utf-8") as f:
            return json.load(f)
