"""Job sources package. Import `get_enabled_sources()` to build the source list."""

from sources.base import JobSource  # noqa: F401
from sources.registry import get_enabled_sources  # noqa: F401
