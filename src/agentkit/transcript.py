"""Transcript utilities: ExecutionContext → LLM messages / debug render (block 6).

to_messages() is the boundary between the agent's internal trace and the
LiteLLM request. In chapter 6 we will insert context compression here —
trimming old tool results or summarising long conversations before the
messages list is sent to the model.
"""
from __future__ import annotations

import json
from typing import Any

from agentkit.context import ExecutionContext
from agentkit.types import Message, ToolCall, ToolResult


def to_messages(ctx: ExecutionContext) -> list[dict[str, Any]]:
    """Convert an ExecutionContext into a LiteLLM-compatible messages list.

    Conversion rules
    ----------------
    * Message          → {"role": ..., "content": ...}
    * ToolCall(s)      → ONE assistant message with a "tool_calls" list.
                         All ToolCall items in a single Event are grouped
                         into the same assistant message to preserve the
                         "they were requested together" semantics.
    * ToolResult       → {"role": "tool", "tool_call_id": ..., "content": ...}
                         Each result becomes its own tool message.
    """
    messages: list[dict[str, Any]] = []

    for event in ctx.events:
        plain: list[Message] = []
        calls: list[ToolCall] = []
        results: list[ToolResult] = []

        for item in event.content:
            if isinstance(item, Message):
                plain.append(item)
            elif isinstance(item, ToolCall):
                calls.append(item)
            elif isinstance(item, ToolResult):
                results.append(item)

        for msg in plain:
            messages.append({"role": msg.role, "content": msg.content})

        if calls:
            messages.append(
                {
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
            )

        for result in results:
            content_str = (
                "\n".join(str(c) for c in result.content) if result.content else ""
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": result.tool_call_id,
                    "content": content_str,
                }
            )

    return messages


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
