"""Compaction strategies for managing context window size (block 12).

All strategies implement CompactionStrategy(Protocol) and are called by
ContextBudget when the context exceeds max_tokens.

Invariant
---------
ToolCall and the matching ToolResult are ALWAYS handled as a pair.
Orphaned ToolResults (without a preceding ToolCall) break the Anthropic
API — they are never produced by these strategies.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from agentkit.types import ContentItem, Message, ToolCall, ToolResult

if TYPE_CHECKING:
    from agentkit.context import ExecutionContext

logger = logging.getLogger(__name__)


# ── Protocol ──────────────────────────────────────────────────────────────────


@runtime_checkable
class CompactionStrategy(Protocol):
    """Interface for context compaction strategies."""

    async def apply(
        self,
        contents: list[ContentItem],
        context: ExecutionContext,
    ) -> list[ContentItem]:
        """Return a compacted copy of *contents*.

        Must never produce orphaned ToolResults (ToolResult without a
        preceding ToolCall with the same tool_call_id).
        """
        ...


# ── Helpers ───────────────────────────────────────────────────────────────────


def _find_tool_call_indices(contents: list[ContentItem]) -> dict[str, int]:
    """Map tool_call_id → index in *contents* for all ToolCalls."""
    return {
        item.tool_call_id: i
        for i, item in enumerate(contents)
        if isinstance(item, ToolCall)
    }


def _find_tool_result_indices(contents: list[ContentItem]) -> dict[str, int]:
    """Map tool_call_id → index in *contents* for all ToolResults."""
    return {
        item.tool_call_id: i
        for i, item in enumerate(contents)
        if isinstance(item, ToolResult)
    }


def _safe_cut_points(contents: list[ContentItem]) -> list[int]:
    """Return indices where it is safe to cut the beginning of *contents*.

    A safe cut point is a position i such that every ToolCall in
    contents[:i] has a matching ToolResult in contents[:i].  Cutting at i
    means keeping contents[i:] without any orphaned ToolResults.
    """
    call_ids_seen: set[str] = set()
    result_ids_seen: set[str] = set()
    safe: list[int] = [0]  # always safe to keep everything

    for i, item in enumerate(contents):
        if isinstance(item, ToolCall):
            call_ids_seen.add(item.tool_call_id)
        elif isinstance(item, ToolResult):
            result_ids_seen.add(item.tool_call_id)
        # Safe after this position if all seen calls have matching results
        if call_ids_seen == result_ids_seen:
            safe.append(i + 1)

    return safe


# ── TruncateOldToolResults ────────────────────────────────────────────────────


class TruncateOldToolResults:
    """Replace old ToolResult content with a placeholder.

    The ``keep_recent`` most recent ToolResults (by order in *contents*)
    are left intact.  Older ones have their ``content`` replaced with
    ``[placeholder]``.  ToolCalls and Messages are never removed, so no
    orphans are ever produced.

    Parameters
    ----------
    keep_recent:
        How many of the most recent ToolResults to preserve verbatim.
    placeholder:
        String to substitute for truncated content.
    """

    def __init__(
        self,
        keep_recent: int = 3,
        placeholder: str = "[tool result evicted]",
    ) -> None:
        self.keep_recent = keep_recent
        self.placeholder = placeholder

    async def apply(
        self,
        contents: list[ContentItem],
        context: ExecutionContext,
    ) -> list[ContentItem]:
        result_indices = [
            i for i, item in enumerate(contents) if isinstance(item, ToolResult)
        ]
        # The last `keep_recent` result indices are preserved
        if self.keep_recent > 0 and result_indices:
            preserved = set(result_indices[-self.keep_recent :])
        else:
            preserved = set()

        out: list[ContentItem] = []
        for i, item in enumerate(contents):
            if isinstance(item, ToolResult) and i not in preserved:
                out.append(
                    ToolResult(
                        tool_call_id=item.tool_call_id,
                        name=item.name,
                        status=item.status,
                        content=[self.placeholder],
                    )
                )
            else:
                out.append(item)
        return out


# ── DropOldest ────────────────────────────────────────────────────────────────


class DropOldest:
    """Drop the oldest complete interaction pairs to fit within *max_tokens*.

    Only drops at "safe cut points" — positions where every ToolCall
    before the cut already has its matching ToolResult.  This guarantees
    no orphaned ToolResults remain.

    Parameters
    ----------
    max_tokens:
        Target token budget (uses character approximation internally).
    """

    def __init__(self, max_tokens: int) -> None:
        self.max_tokens = max_tokens

    async def apply(
        self,
        contents: list[ContentItem],
        context: ExecutionContext,
    ) -> list[ContentItem]:
        from agentkit.memory.budget import estimate_context_tokens

        safe_points = _safe_cut_points(contents)
        # Try progressively larger cuts until we fit.
        # Never cut to position >= len(contents): contents[len:] == [] would send
        # an empty message list to the API and raise "no non-system message" errors.
        for cut in reversed(safe_points):
            if cut == 0 or cut >= len(contents):
                continue
            candidate = contents[cut:]
            if estimate_context_tokens(candidate) <= self.max_tokens:
                return candidate
        # No cut is small enough — keep as much as possible without returning empty
        non_empty = [sp for sp in safe_points if 0 < sp < len(contents)]
        if non_empty:
            return contents[non_empty[-1] :]
        return list(contents)


# ── SummarizeHistory ─────────────────────────────────────────────────────────


class SummarizeHistory:
    """Replace the old portion of history with a single summary Message.

    When ``estimate_context_tokens(contents) >= trigger_tokens``, the
    first ``len(contents) - keep_recent_interactions * 2`` items are
    summarized using FAST_MODEL (cheap) and replaced with a single
    ``Message(role="user", content="[History summary]: ...")``.

    The summary call uses the cheap FAST_MODEL regardless of which model
    the agent itself uses, to keep compaction cost low.

    Parameters
    ----------
    llm_client:
        An LlmClient instance used for summarization calls.
    keep_recent:
        Number of *full interactions* (ToolCall+ToolResult pairs or
        message exchanges) to preserve verbatim at the end.
    trigger_tokens:
        Only apply when the estimated context exceeds this threshold.
    """

    def __init__(
        self,
        llm_client: Any,
        keep_recent: int = 5,
        trigger_tokens: int = 20_000,
    ) -> None:
        self.llm_client = llm_client
        self.keep_recent = keep_recent
        self.trigger_tokens = trigger_tokens

    async def apply(
        self,
        contents: list[ContentItem],
        context: ExecutionContext,
    ) -> list[ContentItem]:
        from agentkit.memory.budget import estimate_context_tokens

        if estimate_context_tokens(contents) < self.trigger_tokens:
            return list(contents)

        # Find a safe cut point that leaves keep_recent full interactions
        safe_points = _safe_cut_points(contents)
        if len(safe_points) <= 1:
            return list(contents)

        # Pick the cut point that leaves approximately keep_recent interactions
        # from the end.  Count ToolResult items as interaction markers.
        result_positions = [
            i for i, item in enumerate(contents) if isinstance(item, ToolResult)
        ]
        if len(result_positions) <= self.keep_recent:
            return list(contents)

        # Cut before the N-th-from-last ToolResult interaction
        target_idx = result_positions[-self.keep_recent]
        # Find the largest safe_point that is <= target_idx
        cut = 0
        for sp in safe_points:
            if sp <= target_idx:
                cut = sp
        if cut == 0:
            return list(contents)

        old_part = contents[:cut]
        new_part = contents[cut:]

        summary_text = await self._summarize(old_part, context)
        summary_msg = Message(role="user", content=f"[History summary]: {summary_text}")

        return [summary_msg, *new_part]  # type: ignore[list-item]

    async def _summarize(
        self,
        contents: list[ContentItem],
        context: ExecutionContext,
    ) -> str:
        """Ask the LLM to produce a concise summary of *contents*."""
        from agentkit.llm import LlmRequest
        from agentkit.transcript import items_to_messages

        messages_text = items_to_messages(contents)
        history_str = "\n".join(
            f"[{m['role']}]: {str(m.get('content', ''))[:500]}"
            for m in messages_text
        )
        prompt = (
            "Summarize the following agent interaction history concisely. "
            "Focus on: what was asked, which tools were called and what they returned, "
            "what conclusions were reached. Be brief (3-6 sentences).\n\n"
            f"History:\n{history_str}"
        )
        request = LlmRequest(
            contents=[Message(role="user", content=prompt)],
        )
        try:
            response = await self.llm_client.generate(request)
            if response.error_message:
                logger.warning("SummarizeHistory LLM error: %s", response.error_message)
                return "[summary unavailable]"
            for item in response.content:
                if isinstance(item, Message):
                    return item.content
            return "[summary unavailable]"
        except Exception as exc:  # noqa: BLE001
            logger.warning("SummarizeHistory failed: %s", exc)
            return "[summary unavailable]"
