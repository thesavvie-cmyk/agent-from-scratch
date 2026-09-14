"""Simple ReAct-style agent loop — block 4.

No abstractions yet; those arrive in block 7 (LlmClient / Agent).
"""
from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any

import litellm

from agentkit.config import FAST_MODEL
from agentkit.schema import function_to_tool_definition

logger = logging.getLogger(__name__)


def simple_agent_loop(
    system_prompt: str,
    question: str,
    tools: list[Callable[..., Any]],
    *,
    model: str = FAST_MODEL,
    max_rounds: int = 10,
) -> tuple[str, list[dict[str, Any]]]:
    """Run a simple think→act→observe cycle.

    Builds OpenAI tool definitions from *tools*, sends the conversation to
    the model, executes any tool calls (errors become strings in context),
    and repeats until the model returns a plain-text answer or max_rounds
    is exhausted.

    Returns (final_answer, full message history).
    """
    tool_definitions = [function_to_tool_definition(t) for t in tools]
    toolbox: dict[str, Callable[..., Any]] = {t.__name__: t for t in tools}

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
                func = toolbox.get(name)
                if func is None:
                    result_str = f"Error: unknown tool '{name}'"
                    logger.info("Round %d | tool=%s — unknown", round_num + 1, name)
                else:
                    raw = func(**args)
                    result_str = str(raw)
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
