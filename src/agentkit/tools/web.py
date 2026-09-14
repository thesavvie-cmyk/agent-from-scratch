"""Web search tool via Tavily."""
from __future__ import annotations

import logging
from typing import Any

from agentkit.config import require_env

logger = logging.getLogger(__name__)

_client: Any = None


def _get_client() -> Any:
    global _client
    if _client is None:
        from tavily import TavilyClient

        _client = TavilyClient(api_key=require_env("TAVILY_API_KEY"))
    return _client


def search_web(
    query: str,
    max_results: int = 5,
    topic: str = "general",
    time_range: str | None = None,
) -> list | str:
    """Search the web for current information using the Tavily search API.
    Use this tool when you need recent events, live data, current statistics,
    or any fact that may have changed since your training cutoff.
    Do NOT call this for well-established facts, definitions, or concepts
    you can answer confidently from memory.

    topic: "general" for most queries, "news" for recent news articles.
    time_range: optional filter — "day", "week", "month", or "year".
    Returns a list of result dicts with title, url, and content.
    """
    try:
        client = _get_client()
        kwargs: dict[str, Any] = {
            "query": query,
            "max_results": max_results,
            "topic": topic,
        }
        if time_range is not None:
            kwargs["time_range"] = time_range
        response = client.search(**kwargs)
        return response.get("results", [])
    except Exception as exc:
        msg = str(exc).lower()
        if "401" in msg or "unauthorized" in msg or "invalid api key" in msg or "invalid_api_key" in msg:
            raise
        return f"Error: {exc}"


def format_results(results: list) -> str:
    """Format Tavily results into a compact string for model context.

    Includes title, url, and content. Omits score and raw_content to
    keep the context concise.
    """
    if not results:
        return "(no results)"
    parts: list[str] = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "")
        url = r.get("url", "")
        content = r.get("content", "")
        parts.append(f"[{i}] {title}\n{url}\n{content}")
    return "\n\n".join(parts)
