"""Context budget management — token estimation and compaction (block 12).

estimate_context_tokens
-----------------------
Fast character-based approximation (~4 chars/token) with a module-level
cache keyed by the content fingerprint.  Using the Anthropic count_tokens
API on every step would be too slow; the approximation is accurate enough
for compaction decisions (±20%).

ContextBudget
-------------
Wraps one or more CompactionStrategies.  Before each LLM call the agent
calls fit(contents, context).  If the estimate exceeds max_tokens the
strategies are applied in order until the budget is met.
"""
from __future__ import annotations

import logging
from typing import Any

from agentkit.types import ContentItem, ToolCall, ToolResult

logger = logging.getLogger(__name__)

# ── Token estimation ──────────────────────────────────────────────────────────

# Module-level cache: fingerprint hash → token count
_token_cache: dict[int, int] = {}


def _item_fingerprint(item: ContentItem) -> str:
    """Return a stable string fingerprint for a ContentItem."""
    if isinstance(item, ToolCall):
        args_str = str(sorted(item.arguments.items()))
        return f"tc:{item.tool_call_id}:{item.name}:{args_str}"
    if isinstance(item, ToolResult):
        content_str = str(item.content)[:200]
        return f"tr:{item.tool_call_id}:{item.status}:{content_str}"
    # Message
    return f"msg:{item.role}:{item.content[:200]}"  # type: ignore[union-attr]


def _content_text(item: ContentItem) -> str:
    """Extract displayable text from a ContentItem for token estimation."""
    if isinstance(item, ToolCall):
        return f"{item.name}({item.arguments})"
    if isinstance(item, ToolResult):
        return " ".join(str(c) for c in item.content)
    return item.content  # type: ignore[union-attr]


def estimate_context_tokens(contents: list[ContentItem]) -> int:
    """Estimate token count for *contents* using a fast cached approximation.

    Uses ~4 characters per token, which is accurate to within ±20% for
    Claude models.  Results are cached by content fingerprint so repeated
    calls on the same (unchanged) contents are O(1).
    """
    key = hash(tuple(_item_fingerprint(item) for item in contents))
    if key in _token_cache:
        return _token_cache[key]

    total_chars = sum(len(_content_text(item)) for item in contents)
    # Add ~10 tokens overhead per item (role labels, JSON structure)
    tokens = max(1, total_chars // 4 + len(contents) * 10)
    _token_cache[key] = tokens
    return tokens


# ── ContextBudget ─────────────────────────────────────────────────────────────


class ContextBudget:
    """Apply compaction strategies to keep context within *max_tokens*.

    Parameters
    ----------
    max_tokens:
        Hard token limit for the LLM request.
    strategies:
        Ordered list of CompactionStrategy instances.  Applied in order
        until the estimate drops below max_tokens.
    """

    def __init__(
        self,
        max_tokens: int,
        strategies: list[Any],
    ) -> None:
        self.max_tokens = max_tokens
        self.strategies = strategies

    async def fit(
        self,
        contents: list[ContentItem],
        context: Any,
    ) -> tuple[list[ContentItem], dict[str, Any]]:
        """Return compacted *contents* and a report dict.

        The report contains:
        - ``original_tokens``: estimated tokens before compaction
        - ``final_tokens``: estimated tokens after compaction
        - ``applied``: list of strategy class names that were applied
        """
        original_tokens = estimate_context_tokens(contents)
        report: dict[str, Any] = {
            "original_tokens": original_tokens,
            "applied": [],
            "final_tokens": original_tokens,
        }

        if original_tokens <= self.max_tokens:
            return list(contents), report

        current = list(contents)
        for strategy in self.strategies:
            if estimate_context_tokens(current) <= self.max_tokens:
                break
            try:
                compacted = await strategy.apply(current, context)
            except Exception as exc:  # noqa: BLE001
                name = type(strategy).__name__
                logger.warning("Compaction strategy %s failed: %s", name, exc)
                continue
            report["applied"].append(type(strategy).__name__)
            current = compacted

        report["final_tokens"] = estimate_context_tokens(current)
        return current, report
