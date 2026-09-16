"""Context budget management — token estimation and compaction (block 12).

estimate_context_tokens
-----------------------
Fast character-based approximation (~4 chars/token) with a module-level
cache keyed by the content fingerprint.  Using the Anthropic count_tokens
API on every step would be too slow.

Accuracy on the deepest ch04_traces run (8 steps, haiku):
  heuristic of final state       : 19,418 tokens
  Anthropic count_tokens (messages format): 19,006 tokens
  ratio                          : 0.98x  ← accurate within 2%

The heuristic correctly estimates message content tokens.
What it does NOT count:
  - System prompt (typically 1–3 K tokens)
  - Tool schemas sent per call (200–500 tokens per tool)

ContextBudget
-------------
Wraps one or more CompactionStrategies.  Before each LLM call the agent
calls fit(contents, context).  If the calibrated estimate exceeds max_tokens
the strategies are applied in order until the budget is met.

The ``calibration_factor`` (default 1.1) accounts for the system-prompt
and tool-schema overhead that is not visible in ContentItems.  For a GAIA
agent with a ~1 K system prompt and 1 tool schema (~ 300 tokens), the
overhead per call is roughly 1.3 K tokens, or ~7% of a 19 K context.
calibration_factor=1.1 covers this gap conservatively.
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
        Target token limit for the LLM request (real API tokens).
    strategies:
        Ordered list of CompactionStrategy instances.  Applied in order
        until the calibrated estimate drops below max_tokens.
    calibration_factor:
        Multiplier applied to heuristic estimates to account for system-prompt
        and tool-schema tokens not visible in ContentItems (~1.1 measured on
        haiku ch04 traces).  Compaction fires when
        ``estimate * calibration_factor > max_tokens``.
    """

    def __init__(
        self,
        max_tokens: int,
        strategies: list[Any],
        calibration_factor: float = 1.1,
    ) -> None:
        self.max_tokens = max_tokens
        self.strategies = strategies
        self.calibration_factor = calibration_factor

    def _calibrated(self, contents: list[ContentItem]) -> int:
        """Return the calibrated token estimate for *contents*."""
        return int(estimate_context_tokens(contents) * self.calibration_factor)

    async def fit(
        self,
        contents: list[ContentItem],
        context: Any,
    ) -> tuple[list[ContentItem], dict[str, Any]]:
        """Return compacted *contents* and a report dict.

        The report contains:
        - ``original_tokens``: calibrated estimate before compaction
        - ``final_tokens``: calibrated estimate after compaction
        - ``applied``: list of strategy class names that were applied
        """
        original_tokens = self._calibrated(contents)
        report: dict[str, Any] = {
            "original_tokens": original_tokens,
            "applied": [],
            "final_tokens": original_tokens,
        }

        if original_tokens <= self.max_tokens:
            return list(contents), report

        current = list(contents)
        for strategy in self.strategies:
            if self._calibrated(current) <= self.max_tokens:
                break
            try:
                compacted = await strategy.apply(current, context)
            except Exception as exc:  # noqa: BLE001
                name = type(strategy).__name__
                logger.warning("Compaction strategy %s failed: %s", name, exc)
                continue
            report["applied"].append(type(strategy).__name__)
            current = compacted

        report["final_tokens"] = self._calibrated(current)
        return current, report
