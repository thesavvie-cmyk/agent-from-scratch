"""ExecutionContext — the running state of one agent invocation (block 6)."""
from __future__ import annotations

import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from agentkit.types import ContentItem, Event, Message, ToolCall, ToolResult


@dataclass
class ExecutionContext:
    """Mutable state accumulated during one agent invocation.

    Holds the ordered list of Events (the trace), the current step
    counter, arbitrary key/value state for tools, and the final result
    once the agent finishes.

    The add_* helpers stamp execution_id automatically so callers never
    have to pass it explicitly.
    """

    execution_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    events: list[Event] = field(default_factory=list)
    current_step: int = 0
    state: dict[str, Any] = field(default_factory=dict)
    final_result: str | BaseModel | None = None

    # ── Core mutation ─────────────────────────────────────────────────────────

    def add_event(self, event: Event) -> None:
        """Append an already-constructed event to the trace."""
        self.events.append(event)

    def increment_step(self) -> None:
        """Advance the step counter by one."""
        self.current_step += 1

    # ── Convenience builders ──────────────────────────────────────────────────

    def add_message(
        self,
        role: str,
        content: str,
        author: str,
    ) -> Event:
        """Append a single text message and return the new Event."""
        event = Event(
            execution_id=self.execution_id,
            author=author,
            content=[Message(role=role, content=content)],  # type: ignore[list-item]
        )
        self.add_event(event)
        return event

    def add_tool_calls(self, calls: list[ToolCall], author: str) -> Event:
        """Append all tool calls as one Event (mirrors one assistant turn)."""
        event = Event(
            execution_id=self.execution_id,
            author=author,
            content=list(calls),  # type: ignore[list-item]
        )
        self.add_event(event)
        return event

    def add_tool_results(self, results: list[ToolResult], author: str) -> Event:
        """Append tool results as one Event."""
        event = Event(
            execution_id=self.execution_id,
            author=author,
            content=list(results),  # type: ignore[list-item]
        )
        self.add_event(event)
        return event

    # ── Query helpers ─────────────────────────────────────────────────────────

    def last_event(self) -> Event | None:
        """Return the most recently appended event, or None."""
        return self.events[-1] if self.events else None

    def iter_content(self) -> Iterator[ContentItem]:
        """Yield every ContentItem across all events in chronological order."""
        for event in self.events:
            yield from event.content

    def find_tool_call(self, tool_call_id: str) -> ToolCall | None:
        """Return the ToolCall with the given id, or None if not found."""
        for item in self.iter_content():
            if isinstance(item, ToolCall) and item.tool_call_id == tool_call_id:
                return item
        return None
