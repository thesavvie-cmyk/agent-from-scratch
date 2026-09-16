"""Built-in callback factories (block 11).

make_approval_callback  — prompts the user before dangerous tool calls
make_search_compressor  — replaces long search results with relevant chunks
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


def make_approval_callback(
    dangerous_tools: set[str],
    prompt_fn: Callable[[str], str] | None = None,
) -> Callable[..., Any]:
    """Return a before_tool callback that requires user confirmation.

    For tools in *dangerous_tools*, *prompt_fn* is called with a description
    of the call. Any answer starting with 'y' (case-insensitive) allows the
    call; anything else returns SkipTool("Declined by user").

    prompt_fn defaults to input() for CLI use. Pass a mock in tests.
    """
    from agentkit.callbacks import SkipTool

    _prompt = prompt_fn if prompt_fn is not None else input

    def callback(
        context: Any,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any] | SkipTool | None:
        if tool_name not in dangerous_tools:
            return None
        desc = f"Tool {tool_name!r} with arguments: {arguments}"
        answer = _prompt(f"\n[approval required] {desc}\nAllow? (y/N): ")
        if answer.strip().lower().startswith("y"):
            return None
        return SkipTool("Declined by user")

    return callback


def make_search_compressor(
    index_factory: Callable[[], Any],
    top_k: int = 3,
    min_tokens: int = 2000,
) -> Callable[..., Any]:
    """Return an after_tool callback that compresses long search results.

    When a tool result exceeds *min_tokens* tokens, the content is chunked,
    embedded, and the *top_k* most relevant chunks are substituted back.

    The query for retrieval comes from the originating ToolCall's arguments:
    it looks for a key named 'query', 'q', or 'search', then falls back to
    the first string value found.

    index_factory: zero-argument callable returning a fresh VectorIndex, e.g.
        ``lambda: VectorIndex(LocalEmbeddings())``

    The substituted result is prefixed with a line showing the token savings.
    Results shorter than *min_tokens* pass through unchanged.
    """
    from agentkit.chunking import sentence_aware_chunking
    from agentkit.tokens import count_tokens
    from agentkit.types import ToolResult

    def callback(
        context: Any,
        tool_name: str,
        result: ToolResult,
    ) -> ToolResult | None:
        if not result.content:
            return None

        content_str = str(result.content[0])
        original_tokens = count_tokens(content_str)
        if original_tokens < min_tokens:
            return None

        # Recover the query from the originating ToolCall
        original_call = context.find_tool_call(result.tool_call_id)
        query: str | None = None
        if original_call is not None:
            args = original_call.arguments
            query = args.get("query") or args.get("q") or args.get("search")
            if query is None:
                for v in args.values():
                    if isinstance(v, str):
                        query = v
                        break

        if not query:
            logger.debug(
                "search_compressor: no query found for tool %r — skipping", tool_name
            )
            return None

        try:
            chunks = sentence_aware_chunking(content_str, chunk_size=500, overlap=50)
            if not chunks:
                return None
            index = index_factory()
            index.add(chunks)
            top = index.search(query, top_k=top_k)
            if not top:
                return None

            compressed = "\n\n---\n\n".join(r["text"] for r in top)
            compressed_tokens = count_tokens(compressed)
            saved = original_tokens - compressed_tokens

            header = (
                f"[Compressed: {original_tokens} → {compressed_tokens} tokens "
                f"({saved} saved). Top {len(top)} chunks for query {query!r}.]\n\n"
            )
            return ToolResult(
                tool_call_id=result.tool_call_id,
                name=result.name,
                status=result.status,
                content=[header + compressed],
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("search_compressor error for %r: %s", tool_name, exc)
            return None

    return callback
