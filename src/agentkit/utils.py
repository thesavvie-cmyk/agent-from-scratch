"""Display utilities for agent execution traces (block 8)."""
from __future__ import annotations

from agentkit.context import ExecutionContext
from agentkit.types import Message, ToolCall, ToolResult


def display_trace(context: ExecutionContext, *, max_content_len: int = 300) -> str:
    """Return a human-readable trace of an ExecutionContext.

    Shows execution id, step/event counts, then for each event: its index,
    author, id prefix, and a truncated preview of each content item.

    Differs from transcript.render() in that max_content_len is configurable
    (render() uses fixed 80/60-char previews) and the output format is more
    suited for printing full traces to a terminal.
    """
    header = (
        f"ExecutionContext {context.execution_id[:8]}..."
        f"  steps={context.current_step}"
        f"  events={len(context.events)}"
    )
    lines = [header]

    for i, event in enumerate(context.events):
        lines.append(f"\n[{i}] author={event.author!r}  id={event.id[:8]}...")
        for item in event.content:
            if isinstance(item, Message):
                preview = item.content[:max_content_len].replace("\n", " ")
                lines.append(f"  Message({item.role}): {preview!r}")
            elif isinstance(item, ToolCall):
                args_str = str(item.arguments)[:max_content_len]
                lines.append(f"  ToolCall({item.name}): {args_str}")
            elif isinstance(item, ToolResult):
                content_str = str(item.content)[:max_content_len]
                lines.append(f"  ToolResult({item.name}, {item.status}): {content_str}")

    return "\n".join(lines)
