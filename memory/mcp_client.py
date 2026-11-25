"""MCP memory-server client over stdio (JSON-RPC 2.0).

We spawn `@modelcontextprotocol/server-memory` as a subprocess and talk to it
over stdin/stdout. The MCP protocol is JSON-RPC 2.0 with newline-delimited
messages. We implement the minimum needed: initialize, tools/call.

Fallback: if Node/npx isn't available or the server fails to start, the
client degrades gracefully into an in-memory stub so the rest of the pipeline
still works (the knowledge graph will be empty but nothing crashes).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import shlex
import shutil
from typing import Any

from config import settings

log = logging.getLogger(__name__)


class MCPToolError(RuntimeError):
    """The memory server ran a tool and reported that it failed."""


class MCPMemoryClient:
    """Async client for the MCP memory server.

    Usage:
        client = MCPMemoryClient()
        await client.start()
        await client.create_entities([...])
        graph = await client.read_graph()
        await client.stop()
    """

    def __init__(self, command: str | None = None, memory_file: str | None = None) -> None:
        self._command = command or settings.mcp_memory_cmd
        # Absolute on purpose: the memory server resolves a relative
        # MEMORY_FILE_PATH against its own install directory (inside the npx
        # cache), not our working directory - every write then failed with
        # ENOENT and the graph stayed empty.
        self._memory_file = os.path.abspath(memory_file or settings.mcp_memory_file)
        self._proc: asyncio.subprocess.Process | None = None
        self._req_id = 0
        self._lock = asyncio.Lock()
        self._in_memory_fallback: dict[str, Any] | None = None
        self._started = False

    # ---------- lifecycle ----------

    async def start(self) -> None:
        """Spawn the MCP server subprocess and perform the `initialize` handshake.

        Safe to call multiple times; only the first call starts the process.
        On failure, we fall back to an in-memory graph so the app keeps working.
        """
        if self._started:
            return
        self._started = True

        # The memory server is launched through `npx`, so it must be on PATH.
        if not self._has_npx():
            log.warning("npx not on PATH - MCP memory server disabled, using in-memory fallback")
            self._enable_fallback()
            return

        cmd_parts = shlex.split(self._command)
        env = os.environ.copy()
        env["MEMORY_FILE_PATH"] = self._memory_file
        os.makedirs(os.path.dirname(self._memory_file) or ".", exist_ok=True)

        try:
            self._proc = await asyncio.create_subprocess_exec(
                *cmd_parts,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
        except (FileNotFoundError, OSError) as e:
            log.warning("Failed to spawn MCP memory server (%s); falling back to in-memory", e)
            self._enable_fallback()
            return

        # MCP handshake - send `initialize` then `notifications/initialized`.
        try:
            init_resp = await self._rpc(
                "initialize",
                {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "jobradar", "version": "1.0.0"},
                },
                timeout=10.0,
            )
            log.info(
                "MCP memory server ready: %s",
                (init_resp.get("serverInfo") or {}).get("name", "memory"),
            )
            # Required by MCP spec after successful initialize.
            await self._notify("notifications/initialized", {})
        except Exception as e:
            log.warning("MCP initialize failed (%s); falling back to in-memory", e)
            await self.stop()
            self._enable_fallback()

    async def stop(self) -> None:
        """Terminate the subprocess. Idempotent."""
        if self._proc and self._proc.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                self._proc.terminate()
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self._proc.wait(), timeout=3.0)
            if self._proc.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    self._proc.kill()
        self._proc = None

    def _has_npx(self) -> bool:
        # Works with `npx ...` at the start of the command, which is the default.
        first = shlex.split(self._command)[0] if self._command else ""
        return bool(shutil.which(first))

    def _enable_fallback(self) -> None:
        """Initialize the in-memory graph stub."""
        self._in_memory_fallback = {"entities": [], "relations": []}

    # ---------- low-level JSON-RPC ----------

    async def _rpc(self, method: str, params: dict[str, Any], timeout: float = 30.0) -> Any:
        """Send a request and await the matching response.

        MCP uses line-delimited JSON, one message per line on stdin/stdout.
        We serialize requests with a lock - stdin is a single stream.
        """
        if not self._proc or not self._proc.stdin or not self._proc.stdout:
            raise RuntimeError("MCP subprocess not running")

        async with self._lock:
            self._req_id += 1
            msg = {"jsonrpc": "2.0", "id": self._req_id, "method": method, "params": params}
            data = (json.dumps(msg) + "\n").encode("utf-8")
            self._proc.stdin.write(data)
            await self._proc.stdin.drain()

            # Wait for a matching response. We may need to skip unrelated
            # server-initiated messages; we do so in a bounded loop.
            deadline = asyncio.get_event_loop().time() + timeout
            while True:
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    raise TimeoutError(f"MCP {method} timed out")
                line = await asyncio.wait_for(self._proc.stdout.readline(), timeout=remaining)
                if not line:
                    raise RuntimeError("MCP server closed stdout")
                try:
                    response = json.loads(line.decode("utf-8").strip())
                except json.JSONDecodeError:
                    continue
                if response.get("id") == self._req_id:
                    if "error" in response:
                        raise RuntimeError(f"MCP error: {response['error']}")
                    return response.get("result")
                # otherwise it's a notification - ignore

    async def _notify(self, method: str, params: dict[str, Any]) -> None:
        """Fire-and-forget notification. No response expected."""
        if not self._proc or not self._proc.stdin:
            return
        msg = {"jsonrpc": "2.0", "method": method, "params": params}
        data = (json.dumps(msg) + "\n").encode("utf-8")
        self._proc.stdin.write(data)
        await self._proc.stdin.drain()

    async def _call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """Call an MCP tool. Server returns a list of content blocks.

        Raises `MCPToolError` when the server flags the result `isError` -
        MCP reports tool failures (e.g. the memory file can't be written)
        as a normal result, not a JSON-RPC error, so without this check a
        failed write looks like a success."""
        result = await self._rpc("tools/call", {"name": name, "arguments": arguments})
        content = (result or {}).get("content") or []
        if (result or {}).get("isError"):
            texts = [c.get("text", "") for c in content if isinstance(c, dict)]
            raise MCPToolError(f"MCP tool {name} failed: {' '.join(texts)[:300]}")
        # Most memory-server tools return JSON text in content[0].text.
        if content and isinstance(content, list) and content[0].get("type") == "text":
            text = content[0].get("text", "")
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return text
        return result

    # ---------- high-level MCP memory-server API ----------

    async def create_entities(self, entities: list[dict[str, Any]]) -> Any:
        """entities: [{"name": str, "entityType": str, "observations": [str]}]"""
        if self._in_memory_fallback is not None:
            existing = {e["name"] for e in self._in_memory_fallback["entities"]}
            for ent in entities:
                if ent["name"] in existing:
                    # Merge observations
                    for e in self._in_memory_fallback["entities"]:
                        if e["name"] == ent["name"]:
                            e["observations"] = list(
                                dict.fromkeys((e.get("observations") or []) + (ent.get("observations") or []))
                            )
                else:
                    self._in_memory_fallback["entities"].append(
                        {
                            "name": ent["name"],
                            "entityType": ent.get("entityType", "Unknown"),
                            "observations": list(ent.get("observations") or []),
                        }
                    )
            return {"created": len(entities)}
        return await self._call_tool("create_entities", {"entities": entities})

    async def create_relations(self, relations: list[dict[str, Any]]) -> Any:
        """relations: [{"from": str, "to": str, "relationType": str}]"""
        if self._in_memory_fallback is not None:
            existing = {(r["from"], r["to"], r["relationType"]) for r in self._in_memory_fallback["relations"]}
            for rel in relations:
                key = (rel["from"], rel["to"], rel["relationType"])
                if key not in existing:
                    self._in_memory_fallback["relations"].append(rel)
                    existing.add(key)
            return {"created": len(relations)}
        return await self._call_tool("create_relations", {"relations": relations})

    async def add_observations(self, observations: list[dict[str, Any]]) -> Any:
        """observations: [{"entityName": str, "contents": [str]}]"""
        if self._in_memory_fallback is not None:
            by_name = {e["name"]: e for e in self._in_memory_fallback["entities"]}
            for o in observations:
                ent = by_name.get(o["entityName"])
                if ent:
                    ent["observations"] = list(
                        dict.fromkeys((ent.get("observations") or []) + list(o.get("contents") or []))
                    )
            return {"updated": len(observations)}
        return await self._call_tool("add_observations", {"observations": observations})

    async def search_nodes(self, query: str) -> Any:
        if self._in_memory_fallback is not None:
            q = query.lower()
            ents = [
                e
                for e in self._in_memory_fallback["entities"]
                if q in e["name"].lower()
                or q in e.get("entityType", "").lower()
                or any(q in (o or "").lower() for o in e.get("observations", []))
            ]
            names = {e["name"] for e in ents}
            rels = [
                r
                for r in self._in_memory_fallback["relations"]
                if r["from"] in names or r["to"] in names
            ]
            return {"entities": ents, "relations": rels}
        return await self._call_tool("search_nodes", {"query": query})

    async def read_graph(self) -> dict[str, Any]:
        if self._in_memory_fallback is not None:
            # Return a shallow copy so callers can mutate without corrupting state.
            return {
                "entities": list(self._in_memory_fallback["entities"]),
                "relations": list(self._in_memory_fallback["relations"]),
            }
        result = await self._call_tool("read_graph", {})
        # The MCP server returns {"entities": [...], "relations": [...]}
        if isinstance(result, dict):
            return result
        return {"entities": [], "relations": []}

    async def delete_entity(self, entity_names: list[str]) -> Any:
        if self._in_memory_fallback is not None:
            self._in_memory_fallback["entities"] = [
                e for e in self._in_memory_fallback["entities"] if e["name"] not in entity_names
            ]
            self._in_memory_fallback["relations"] = [
                r
                for r in self._in_memory_fallback["relations"]
                if r["from"] not in entity_names and r["to"] not in entity_names
            ]
            return {"deleted": len(entity_names)}
        return await self._call_tool("delete_entities", {"entityNames": entity_names})


# ---------- singleton accessor ----------

_client: MCPMemoryClient | None = None


async def get_memory_client() -> MCPMemoryClient:
    """Return the process-wide MCP client, starting it on first use."""
    global _client
    if _client is None:
        _client = MCPMemoryClient()
        await _client.start()
    return _client


async def shutdown_memory_client() -> None:
    """Stop the singleton - call on app shutdown."""
    global _client
    if _client is not None:
        await _client.stop()
        _client = None
