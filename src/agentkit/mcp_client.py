"""MCP client — McpToolset async context manager (block 5)."""
from __future__ import annotations

import contextlib
import logging
from typing import Any, Self

from agentkit.schema import build_tool_definition

logger = logging.getLogger(__name__)


class McpToolset:
    """Async context manager wrapping an MCP stdio server connection.

    Connects to a server process, fetches the tool list, and exposes
    tool_definitions() (OpenAI format) and async call().

    Usage::

        async with McpToolset("npx", ["-y", "tavily-mcp@latest"]) as ts:
            defns = ts.tool_definitions()   # list of OpenAI tool dicts
            text  = await ts.call("tavily_search", {"query": "..."})
    """

    def __init__(
        self,
        command: str,
        args: list[str],
        env: dict[str, str] | None = None,
    ) -> None:
        self._command = command
        self._args = args
        self._env = env
        self._session: Any = None
        self._tools: list[Any] = []
        self._stack = contextlib.AsyncExitStack()

    async def __aenter__(self) -> Self:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        params = StdioServerParameters(
            command=self._command,
            args=self._args,
            env=self._env,
        )
        try:
            read, write = await self._stack.enter_async_context(stdio_client(params))
        except (FileNotFoundError, OSError) as exc:
            await self._stack.aclose()
            cmd = f"{self._command} {' '.join(self._args)}"
            raise RuntimeError(
                f"Could not start MCP server '{cmd}': {exc}. "
                "Check that the command is installed and available on PATH "
                "(e.g. 'npx' for Node-based servers, 'uv' for Python servers)."
            ) from exc

        self._session = await self._stack.enter_async_context(ClientSession(read, write))
        await self._session.initialize()
        result = await self._session.list_tools()
        self._tools = result.tools
        logger.info("McpToolset connected: %d tool(s) available", len(self._tools))
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self._stack.aclose()

    @staticmethod
    def _get_input_schema(tool: Any) -> dict[str, Any]:
        """Return tool's input schema, handling mcp 1.x (inputSchema) and 2.x (input_schema)."""
        schema = getattr(tool, "input_schema", None) or getattr(tool, "inputSchema", None)
        return schema if isinstance(schema, dict) else {}

    def tool_definitions(self) -> list[dict[str, Any]]:
        """Return connected tools as OpenAI function-calling definitions."""
        return [
            build_tool_definition(
                t.name,
                t.description or "",
                self._get_input_schema(t),
            )
            for t in self._tools
        ]

    @staticmethod
    def _extract_text(content: list[Any]) -> str:
        """Extract text from a list of MCP content blocks."""
        parts = [block.text for block in content if hasattr(block, "text")]
        return "\n".join(parts) if parts else "(empty response)"

    async def call(self, name: str, arguments: dict[str, Any]) -> str:
        """Invoke an MCP tool by name and return its output as a string."""
        try:
            result = await self._session.call_tool(name, arguments)
            return self._extract_text(result.content)
        except Exception as exc:  # noqa: BLE001
            return f"Error: {exc}"
