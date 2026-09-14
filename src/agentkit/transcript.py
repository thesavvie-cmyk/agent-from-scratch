"""Transcript utilities: ExecutionContext → LLM messages / debug render (block 6).

to_messages() is the boundary between the agent's internal trace and the
LiteLLM request. In chapter 6 we will insert context compression here —
trimming old tool results or summarising long conversations before the
messages list is sent to the model.

items_to_messages() is the shared conversion core used by both to_messages()
(block 6) and LlmClient._build_messages() (block 7). The split avoids
duplicating the grouping logic.
"""
from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

from agentkit.context import ExecutionContext
from agentkit.types import ContentItem, Message, ToolCall, ToolResult


def items_to_messages(items: Iterable[ContentItem]) -> list[dict[str, Any]]:
    """Convert a flat sequence of ContentItems to LiteLLM-compatible messages.

    Conversion rules
    ----------------
    * Message        → {"role": ..., "content": ...}
    * ToolCall(s)    → Consecutive ToolCall items are collected into ONE
                       assistant message with a "tool_calls" list.
    * ToolResult     → {"role": "tool", "tool_call_id": ..., "content": ...}
                       Each result becomes its own tool message.
    """
    messages: list[dict[str, Any]] = []
    pending_calls: list[ToolCall] = []

    for item in items:
        if isinstance(item, ToolCall):
            pending_calls.append(item)
        else:
            if pending_calls:
                messages.append(_calls_to_assistant(pending_calls))
                pending_calls = []
            if isinstance(item, Message):
                messages.append({"role": item.role, "content": item.content})
            elif isinstance(item, ToolResult):
                content_str = (
                    "\n".join(str(c) for c in item.content) if item.content else ""
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": item.tool_call_id,
                        "content": content_str,
                    }
                )

    if pending_calls:
        messages.append(_calls_to_assistant(pending_calls))

    return messages


def _calls_to_assistant(calls: list[ToolCall]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": tc.tool_call_id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": json.dumps(tc.arguments),
                },
            }
            for tc in calls
        ],
    }


def to_messages(ctx: ExecutionContext) -> list[dict[str, Any]]:
    """Convert an ExecutionContext into a LiteLLM-compatible messages list.

    Delegates to items_to_messages(ctx.iter_content()). Chapter 6 will
    insert compression/summarisation here before the items are converted.
    """
    return items_to_messages(ctx.iter_content())


def render(ctx: ExecutionContext) -> str:
    """Return a human-readable trace dump for debugging.

    Shows step counter, author, event id prefix, and a short content preview
    for each item. Safe to call on an empty context.
    """
    header = (
        f"ExecutionContext {ctx.execution_id[:8]}..."
        f"  step={ctx.current_step}"
        f"  events={len(ctx.events)}"
    )
    lines = [header]
    for i, event in enumerate(ctx.events):
        lines.append(f"\n  [{i}] author={event.author!r}  id={event.id[:8]}...")
        for item in event.content:
            if isinstance(item, Message):
                preview = item.content[:80].replace("\n", " ")
                lines.append(f"      Message({item.role}): {preview!r}")
            elif isinstance(item, ToolCall):
                args_preview = str(item.arguments)[:60]
                lines.append(f"      ToolCall({item.name}): {args_preview}")
            elif isinstance(item, ToolResult):
                content_preview = str(item.content)[:60]
                lines.append(
                    f"      ToolResult({item.name}, {item.status}): {content_preview}"
                )
    return "\n".join(lines)
