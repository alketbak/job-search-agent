"""MCP memory client: the two causes of the always-empty knowledge graph.

1. The memory server resolves a relative MEMORY_FILE_PATH against its own
   install directory, so the client must hand it an absolute path.
2. Tool failures arrive as a normal result flagged `isError`, which the
   client used to return as if it were data - every failed write looked
   like a success.

The real server (spawned via npx) isn't started here: tests must not depend
on Node or network access.
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock

import pytest

from memory.mcp_client import MCPMemoryClient, MCPToolError


def test_memory_file_path_is_made_absolute() -> None:
    client = MCPMemoryClient(command="npx -y x", memory_file="data/memory.json")
    assert os.path.isabs(client._memory_file)
    assert client._memory_file == os.path.join(os.getcwd(), "data", "memory.json")


@pytest.mark.asyncio
async def test_tool_error_result_raises_instead_of_returning_text() -> None:
    client = MCPMemoryClient(command="npx -y x")
    client._rpc = AsyncMock(  # type: ignore[method-assign]
        return_value={
            "content": [{"type": "text", "text": "ENOENT: no such file or directory, open '/x/memory.json'"}],
            "isError": True,
        }
    )
    with pytest.raises(MCPToolError, match="ENOENT"):
        await client.create_entities([{"name": "A", "entityType": "T", "observations": []}])


@pytest.mark.asyncio
async def test_successful_tool_result_is_parsed() -> None:
    client = MCPMemoryClient(command="npx -y x")
    client._rpc = AsyncMock(  # type: ignore[method-assign]
        return_value={"content": [{"type": "text", "text": '{"entities": [{"name": "A"}], "relations": []}'}]}
    )
    graph = await client.read_graph()
    assert graph == {"entities": [{"name": "A"}], "relations": []}
