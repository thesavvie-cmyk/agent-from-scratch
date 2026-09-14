"""Core data types for the agent framework (block 6).

These are the shared vocabulary that all framework components speak.
Everything that passes through an agent is represented as one of these types.
"""
from __future__ import annotations

import time
import uuid
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field


class Message(BaseModel):
    """A plain text message from system, user, or assistant."""

    type: Literal["message"] = "message"
    role: Literal["system", "user", "assistant"]
    content: str


class ToolCall(BaseModel):
    """A request from the assistant to invoke a tool."""

    type: Literal["tool_call"] = "tool_call"
    tool_call_id: str
    name: str
    arguments: dict[str, Any]


class ToolResult(BaseModel):
    """The outcome of executing a tool call."""

    type: Literal["tool_result"] = "tool_result"
    tool_call_id: str
    name: str
    status: Literal["success", "error"]
    content: list[Any]


# Discriminated union — Pydantic uses the 'type' field to pick the right class
# during JSON deserialization. Without the discriminator it would try each type
# in order and could silently pick the wrong one.
ContentItem = Annotated[
    Message | ToolCall | ToolResult,
    Field(discriminator="type"),
]


class Event(BaseModel):
    """One atomic unit of an agent's execution trace.

    content is a list (not a single item) because one logical step can
    produce several related pieces simultaneously. The clearest example:
    an LLM response in a single API call can contain both a text fragment
    AND two parallel tool calls — that is one assistant turn, so all three
    items belong in one Event. Splitting them across events would lose the
    "they happened together" information and make to_messages() harder to
    reconstruct correctly.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    execution_id: str
    timestamp: float = Field(default_factory=time.time)
    author: str
    content: list[ContentItem]
