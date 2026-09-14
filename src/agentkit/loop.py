"""Simple ReAct-style agent loop — block 4, extended with MCP in block 5.

Design decision: the core implementation is async (async_simple_agent_loop)
because MCP tool calls are inherently async. simple_agent_loop is kept as a
synchronous wrapper via asyncio.run() for backward compatibility in scripts
and unit tests. Callers in async contexts (e.g. when McpToolset is already
open) should await async_simple_agent_loop directly to avoid nested-event-loop
errors.
"""
from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import litellm

from agentkit.config import FAST_MODEL
from agentkit.schema import function_to_tool_definition

if TYPE_CHECKING:
    from agentkit.mcp_client import McpToolset

logger = logging.getLogger(__name__)


async def async_simple_agent_loop(
    system_prompt: str,
    question: str,
    tools: list[Callable[..., Any]],
    *,
    model: str = FAST_MODEL,
    max_rounds: int = 10,
    mcp_toolsets: list[McpToolset] | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Async ReAct-style think-act-observe loop.

    Combines local Python callables with optional MCP toolsets.
    On tool name collision, the local callable wins and a warning is logged.

    Returns (final_answer, full message history).
    Raises RuntimeError if max_rounds is exhausted.
    """
    tool_definitions = [function_to_tool_definition(t) for t in tools]
    toolbox: dict[str, Callable[..., Any]] = {t.__name__: t for t in tools}

    mcp_lookup: dict[str, Any] = {}
    if mcp_toolsets:
        for ts in mcp_toolsets:
            for defn in ts.tool_definitions():
                name = defn["function"]["name"]
                if name in toolbox:
                    logger.warning(
                        "Tool name collision: '%s' exists in local toolbox and MCP toolset"
                        " — local tool wins.",
                        name,
                    )
                elif name in mcp_lookup:
                    logger.warning(
                        "Tool name collision across MCP toolsets: '%s' — first toolset wins.",
                        name,
                    )
                else:
                    mcp_lookup[name] = ts
                    tool_definitions.append(defn)

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    for round_num in range(max_rounds):
        response = litellm.completion(
            model=model,
            messages=messages,
            tools=tool_definitions,
        )

        msg = response.choices[0].message

        assistant_msg: dict[str, Any] = {
            "role": "assistant",
            "content": msg.content,
        }
        if msg.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]
        messages.append(assistant_msg)

        if not msg.tool_calls:
            return msg.content or "", messages

        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args: dict[str, Any] = json.loads(tc.function.arguments)
                if name in toolbox:
                    raw = toolbox[name](**args)
                    result_str = str(raw)
                elif name in mcp_lookup:
                    result_str = await mcp_lookup[name].call(name, args)
                else:
                    result_str = f"Error: unknown tool '{name}'"
                    logger.info("Round %d | tool=%s — unknown", round_num + 1, name)
                logger.info(
                    "Round %d | tool=%s args=%s result_len=%d",
                    round_num + 1,
                    name,
                    args,
                    len(result_str),
                )
            except json.JSONDecodeError as exc:
                result_str = f"Error: invalid JSON in arguments — {exc}"
                logger.info("Round %d | tool=%s — JSON error: %s", round_num + 1, name, exc)
            except Exception as exc:  # noqa: BLE001
                result_str = f"Error: {exc}"
                logger.info("Round %d | tool=%s — raised: %s", round_num + 1, name, exc)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_str,
                }
            )

    raise RuntimeError(
        f"Agent loop did not produce a final answer after {max_rounds} rounds."
    )


def simple_agent_loop(
    system_prompt: str,
    question: str,
    tools: list[Callable[..., Any]],
    *,
    model: str = FAST_MODEL,
    max_rounds: int = 10,
    mcp_toolsets: list[McpToolset] | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Synchronous wrapper around async_simple_agent_loop.

    Creates a fresh event loop via asyncio.run(). Cannot be called from
    within a running event loop — use async_simple_agent_loop directly
    in async contexts (e.g. when McpToolset is already open).
    """
    return asyncio.run(
        async_simple_agent_loop(
            system_prompt,
            question,
            tools,
            model=model,
            max_rounds=max_rounds,
            mcp_toolsets=mcp_toolsets,
        )
    )
