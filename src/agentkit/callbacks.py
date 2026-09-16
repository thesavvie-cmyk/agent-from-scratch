"""Callback hooks for the agent loop (block 11).

Callbacks intercept four points in the ReAct loop:

  before_model(context, request)           — observe/log before LLM call
  after_model(context, response)           — observe/log after LLM call
  before_tool(context, name, arguments)    — modify args or cancel the call
  after_tool(context, name, result)        — replace the result

Return value semantics
----------------------
before_tool:
    dict        → replaces the tool arguments for this call
    SkipTool    → cancels execution; agent sees ToolResult(status='error')
    None        → arguments unchanged

after_tool:
    ToolResult  → replaces the result seen by the model
    None        → result unchanged

before_model / after_model:
    return value is ignored (observation only)

Exceptions in any callback are logged and suppressed — they cannot crash
the agent. Async callbacks (coroutines) are awaited automatically.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict


class SkipTool:
    """Sentinel returned from a before_tool callback to cancel a tool call.

    When returned, the tool function is not executed. The agent receives
    a ToolResult(status='error', content=[f'Tool skipped: {reason}']).
    """

    __slots__ = ("reason",)

    def __init__(self, reason: str) -> None:
        self.reason = reason

    def __repr__(self) -> str:
        return f"SkipTool({self.reason!r})"


class Callbacks(BaseModel):
    """Container for agent lifecycle callback lists.

    Each list is invoked in order. Multiple callbacks can be registered for
    the same hook — all run unless one raises (which is logged and skipped).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    before_model: list[Callable[..., Any]] = []
    after_model: list[Callable[..., Any]] = []
    before_tool: list[Callable[..., Any]] = []
    after_tool: list[Callable[..., Any]] = []
