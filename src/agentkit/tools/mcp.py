"""MCP tool adapter — load_mcp_tools (block 7).

Design note: the book spins up a fresh npx process for every tool call.
We reuse the persistent McpToolset session from block 5 instead.
See experiments/ch04_layer.py section (e) for the timing comparison.
"""
from __future__ import annotations

from typing import Any

from agentkit.context import ExecutionContext
from agentkit.mcp_client import McpToolset
from agentkit.tools.base import BaseTool


class _McpTool(BaseTool):
    """Thin wrapper that forwards execute() to a live McpToolset session."""

    def __init__(self, toolset: McpToolset, defn: dict[str, Any]) -> None:
        fn = defn["function"]
        super().__init__(
            name=fn["name"],
            description=fn.get("description", ""),
            tool_definition=defn,
        )
        self._toolset = toolset

    async def execute(self, context: ExecutionContext, **kwargs: Any) -> Any:
        return await self._toolset.call(self.name, kwargs)


def load_mcp_tools(toolset: McpToolset) -> list[BaseTool]:
    """Create a BaseTool for each tool exposed by *toolset*.

    The toolset must already be connected (used as an async context manager).
    Tool definitions are taken from the server — not regenerated locally.
    The persistent session is reused for every call; no new subprocess is
    spawned per invocation.
    """
    return [_McpTool(toolset, defn) for defn in toolset.tool_definitions()]
