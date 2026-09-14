"""Custom MCP server: search_web + calculator via FastMCP (block 5).

Run as stdio server::

    uv run python -m agentkit.servers.tavily_server
"""
from __future__ import annotations

try:
    from mcp.server.fastmcp import FastMCP as _Server  # mcp 1.x
except ModuleNotFoundError:
    from mcp.server.mcpserver import (
        MCPServer as _Server,  # type: ignore[no-redef]  # mcp 2.x
    )

from agentkit.tools.web import format_results
from agentkit.tools.web import search_web as _search_web
from agentkit.tools_manual import calculator as _calculator

mcp = _Server("custom-tavily-search")


@mcp.tool()
def search_web(
    query: str,
    max_results: int = 5,
    topic: str = "general",
    time_range: str | None = None,
) -> str:
    """Search the web for current information using the Tavily search API.

    Use this tool when you need recent events, live data, current statistics,
    or any fact that may have changed since your training cutoff.
    Do NOT call this for well-established facts, definitions, mathematical
    concepts, or historical events you can answer from memory.

    Parameters
    ----------
    query:
        The search query string.
    max_results:
        Number of results to return (default 5, max 10).
    topic:
        'general' for most queries; 'news' for recent news articles.
    time_range:
        Optional recency filter: 'day', 'week', 'month', or 'year'.
        Omit for no filter.

    Returns formatted text with title, URL, and snippet for each result.
    """
    results = _search_web(query, max_results=max_results, topic=topic, time_range=time_range)
    if isinstance(results, list):
        return format_results(results)
    return str(results)


@mcp.tool()
def calculator(
    operator: str,
    first_number: float,
    second_number: float,
) -> float:
    """Perform basic arithmetic on two numbers.

    Use this whenever you need precise numeric computation: addition,
    subtraction, multiplication, or division.
    Do NOT use for trivial mental arithmetic.

    Parameters
    ----------
    operator:
        The operation to perform. Must be one of:
        'add', 'subtract', 'multiply', 'divide'.
    first_number:
        The left-hand operand.
    second_number:
        The right-hand operand. Must not be zero when operator is 'divide'.
    """
    return _calculator(operator, first_number, second_number)


if __name__ == "__main__":
    mcp.run(transport="stdio")
