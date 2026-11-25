"""Memory package - MCP knowledge-graph client and high-level helpers."""

from memory.knowledge_graph import (  # noqa: F401
    get_company_history,
    get_full_graph,
    store_application,
    store_job_run,
    store_skill_gap,
)
from memory.mcp_client import MCPMemoryClient, get_memory_client  # noqa: F401
