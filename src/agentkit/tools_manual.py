"""Manual tool-calling loop — block 3.

Explicit implementation of the think→act→observe cycle without any
agent abstraction. The abstractions arrive in block 7 (LlmClient).
"""
from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import litellm

from agentkit.config import FAST_MODEL

# ── Tool definition (OpenAI schema) ──────────────────────────────────────────

CALCULATOR_TOOL_DEFINITION: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": (
            "Perform basic arithmetic on two numbers. "
            "Use this whenever you need to add, subtract, multiply, or divide."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "operator": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide"],
                    "description": "The arithmetic operation to perform.",
                },
                "first_number": {
                    "type": "number",
                    "description": "The first operand.",
                },
                "second_number": {
                    "type": "number",
                    "description": "The second operand.",
                },
            },
            "required": ["operator", "first_number", "second_number"],
        },
    },
}


# ── Calculator implementation ─────────────────────────────────────────────────

def calculator(
    operator: str,
    first_number: float,
    second_number: float,
) -> float:
    """Perform a basic arithmetic operation on two numbers."""
    if operator == "add":
        return first_number + second_number
    if operator == "subtract":
        return first_number - second_number
    if operator == "multiply":
        return first_number * second_number
    if operator == "divide":
        if second_number == 0:
            raise ValueError("Division by zero")
        return first_number / second_number
    raise ValueError(f"Unknown operator: {operator!r}")


# ── Tool registry and execution ───────────────────────────────────────────────

TOOL_REGISTRY: dict[str, Callable[..., Any]] = {
    "calculator": calculator,
}


def execute_tool_call(tool_call: Any) -> str:
    """Execute a single tool call object. Returns the result as a string.

    Never raises — any error is returned as an "Error: ..." string so the
    model can see what went wrong and decide what to do next. This is
    deliberate: the model must be able to recover from tool failures.
    """
    try:
        name: str = tool_call.function.name
        func = TOOL_REGISTRY.get(name)
        if func is None:
            return f"Error: unknown tool '{name}'"
        args: dict[str, Any] = json.loads(tool_call.function.arguments)
        result = func(**args)
        return str(result)
    except json.JSONDecodeError as exc:
        return f"Error: invalid JSON in arguments — {exc}"
    except Exception as exc:  # noqa: BLE001
        return f"Error: {exc}"


# ── Explicit tool-use loop ────────────────────────────────────────────────────

def run_with_tools(
    question: str,
    *,
    model: str = FAST_MODEL,
    max_rounds: int = 5,
) -> tuple[str, list[dict[str, Any]]]:
    """Run the think→act→observe cycle explicitly.

    Sends *question* with CALCULATOR_TOOL_DEFINITION attached.
    If the model returns tool_calls, executes each one, appends the results,
    and repeats. Stops when the model returns a plain text answer.

    Returns (final_answer, full message history).
    Raises RuntimeError if max_rounds is exhausted.
    """
    messages: list[dict[str, Any]] = [{"role": "user", "content": question}]

    for _round in range(max_rounds):
        response = litellm.completion(
            model=model,
            messages=messages,
            tools=[CALCULATOR_TOOL_DEFINITION],
        )

        msg = response.choices[0].message

        # Append assistant turn (serialisable dict)
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

        # Execute every tool call, append one tool message per call
        for tc in msg.tool_calls:
            result = execute_tool_call(tc)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                }
            )

    raise RuntimeError(
        f"Tool loop did not produce a final answer after {max_rounds} rounds. "
        "Increase max_rounds or check your tool definitions."
    )
