"""Source registry - decides which job sources to enable based on config.

We keep this separate from `__init__.py` so the decision logic is easy to
unit-test: call `get_enabled_sources()` with different settings and assert
the returned list.
"""

from __future__ import annotations

import logging

from config import settings
from sources.adzuna import AdzunaSource
from sources.arbeitnow import ArbeitnowSource
from sources.base import JobSource
from sources.bundesagentur import BundesagenturSource
from sources.mock import MockSource

log = logging.getLogger(__name__)


def get_enabled_sources() -> list[JobSource]:
    """Return the list of JobSource instances to query on this run."""
    if settings.use_mock_data:
        log.info("USE_MOCK_DATA=true - using MockSource only")
        return [MockSource()]

    sources: list[JobSource] = [ArbeitnowSource(), BundesagenturSource()]
    adzuna = AdzunaSource()
    if adzuna.enabled():
        sources.append(adzuna)
    else:
        log.info("Adzuna credentials not set - skipping")

    log.info("Enabled sources: %s", [s.name for s in sources])
    return sources
