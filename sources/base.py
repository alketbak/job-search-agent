"""Abstract base class for job sources.

Every concrete source must return a list of dicts with the standardized keys:
    title, company, location, description, url, source

Additional keys are allowed and will be passed through, but these six are
contract-level — downstream agents rely on them.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

log = logging.getLogger(__name__)

REQUIRED_FIELDS = ("title", "company", "location", "description", "url", "source")


class JobSource(ABC):
    """Abstract base class for a job board / API integration."""

    #: Short, lowercase identifier — used in log output and persisted to DB.
    name: str = "base"

    @abstractmethod
    async def fetch(self, query: str, location: str, limit: int) -> list[dict[str, Any]]:
        """Fetch jobs from this source.

        Args:
            query: Free-text query, e.g. "AI Engineer".
            location: City / region, e.g. "Munich".
            limit: Maximum number of results to return.

        Returns:
            A list of normalized job dicts. Must never raise — on error return [].
        """

    def validate(self, job: dict[str, Any]) -> dict[str, Any]:
        """Ensure the job dict has all required fields; fill blanks with empty string."""
        out = dict(job)
        for field in REQUIRED_FIELDS:
            out.setdefault(field, "")
        out["source"] = self.name  # always stamp our own name
        return out
