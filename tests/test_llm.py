"""Block 7 tests: FunctionTool, @tool decorator, LlmClient."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import litellm
import pytest

from agentkit.context import ExecutionContext
from agentkit.llm import LlmClient, LlmRequest
from agentkit.tools.base import FunctionTool, tool
from agentkit.types import Message, ToolCall, ToolResult

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_raw_response(
    content: str | None = None,
    tool_calls: list | None = None,
    input_tokens: int = 10,
    output_tokens: int = 5,
) -> MagicMock:
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls or []
    usage = MagicMock()
    usage.prompt_tokens = input_tokens
    usage.completion_tokens = output_tokens
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message = msg
    resp.usage = usage
    return resp


def _make_tool_call_obj(call_id: str, name: str, args: dict) -> MagicMock:
    tc = MagicMock()
    tc.id = call_id
    tc.function.name = name
    tc.function.arguments = json.dumps(args)
    return tc


# ── FunctionTool unit tests ───────────────────────────────────────────────────


async def test_function_tool_sync_execute():
    def add(x: int, y: int) -> int:
        """Add two numbers."""
        return x + y

    ft = FunctionTool(add)
    ctx = ExecutionContext()
    result = await ft.execute(ctx, x=3, y=4)
    assert result == 7


async def test_function_tool_async_execute():
    async def greet(name: str) -> str:
        """Greet someone."""
        return f"Hello, {name}"

    ft = FunctionTool(greet)
    ctx = ExecutionContext()
    result = await ft.execute(ctx, name="World")
    assert result == "Hello, World"


async def test_function_tool_context_injected():
    """Function with 'context' parameter receives the live ExecutionContext."""

    def step_info(context: ExecutionContext) -> str:
        """Return current step."""
        return f"step:{context.current_step}"

    ft = FunctionTool(step_info)
    ctx = ExecutionContext()
    ctx.increment_step()
    ctx.increment_step()
    result = await ft.execute(ctx)
    assert result == "step:2"


async def test_function_tool_no_context_not_injected():
    """Function without 'context' parameter works without context kwarg."""
    calls = []

    def record(**kwargs: object) -> str:
        """Record kwargs."""
        calls.append(set(kwargs.keys()))
        return "ok"

    ft = FunctionTool(record)
    ctx = ExecutionContext()
    await ft.execute(ctx)
    assert "context" not in calls[0]


def test_function_tool_name_from_function():
    def my_func(x: str) -> str:
        """Does something."""
        return x

    ft = FunctionTool(my_func)
    assert ft.name == "my_func"


def test_function_tool_name_override():
    def my_func(x: str) -> str:
        """Does something."""
        return x

    ft = FunctionTool(my_func, name="custom_name")
    assert ft.name == "custom_name"
    assert ft.tool_definition["function"]["name"] == "custom_name"


def test_function_tool_description_from_docstring():
    def my_func(x: str) -> str:
        """First paragraph.

        Second paragraph is ignored.
        """
        return x

    ft = FunctionTool(my_func)
    assert ft.description == "First paragraph."


def test_function_tool_context_excluded_from_schema():
    """'context' parameter must not appear in the tool schema."""

    def step_tool(context: ExecutionContext, query: str) -> str:
        """A tool with context."""
        return query

    ft = FunctionTool(step_tool)
    props = ft.tool_definition["function"]["parameters"]["properties"]
    assert "context" not in props
    assert "query" in props


# ── @tool decorator tests ─────────────────────────────────────────────────────


def test_tool_decorator_no_args():
    @tool
    def my_tool(x: str) -> str:
        """My tool description."""
        return x

    assert isinstance(my_tool, FunctionTool)
    assert my_tool.name == "my_tool"
    assert "My tool" in my_tool.description


def test_tool_decorator_with_name():
    @tool(name="renamed_tool")
    def original(x: str) -> str:
        """Original description."""
        return x

    assert isinstance(original, FunctionTool)
    assert original.name == "renamed_tool"
    assert original.tool_definition["function"]["name"] == "renamed_tool"


def test_tool_decorator_with_description():
    @tool(description="Custom description.")
    def my_tool(x: str) -> str:
        """Original docstring."""
        return x

    assert my_tool.description == "Custom description."


def test_tool_decorator_preserves_wraps():
    @tool
    def documented(x: str) -> str:
        """Important docstring."""
        return x

    assert documented.__doc__ == "Important docstring."


# ── LlmClient._build_messages tests ──────────────────────────────────────────


def test_build_messages_instructions_become_system():
    client = LlmClient("anthropic/claude-haiku-4-5-20251001")
    req = LlmRequest(
        instructions=["Be helpful.", "Be concise."],
        contents=[Message(role="user", content="hi")],
        tools=[],
    )
    msgs = client._build_messages(req)
    assert msgs[0] == {"role": "system", "content": "Be helpful."}
    assert msgs[1] == {"role": "system", "content": "Be concise."}
    assert msgs[2] == {"role": "user", "content": "hi"}


def test_build_messages_two_tool_calls_one_assistant():
    """Two ToolCall in contents → one assistant message."""
    client = LlmClient("anthropic/claude-haiku-4-5-20251001")
    req = LlmRequest(
        contents=[
            ToolCall(tool_call_id="tc-1", name="search", arguments={"q": "a"}),
            ToolCall(tool_call_id="tc-2", name="calc", arguments={"n": 1}),
        ],
        tools=[],
    )
    msgs = client._build_messages(req)
    assert len(msgs) == 1
    assert msgs[0]["role"] == "assistant"
    assert len(msgs[0]["tool_calls"]) == 2


def test_build_messages_tool_result():
    client = LlmClient("anthropic/claude-haiku-4-5-20251001")
    req = LlmRequest(
        contents=[
            ToolResult(
                tool_call_id="tc-1", name="search", status="success", content=["found it"]
            )
        ],
        tools=[],
    )
    msgs = client._build_messages(req)
    assert msgs[0]["role"] == "tool"
    assert msgs[0]["tool_call_id"] == "tc-1"
    assert "found it" in msgs[0]["content"]


# ── LlmClient._parse_response tests ──────────────────────────────────────────


def test_parse_response_text():
    client = LlmClient("anthropic/claude-haiku-4-5-20251001")
    raw = _make_raw_response(content="Hello!")
    resp = client._parse_response(raw)
    assert resp.error_message is None
    assert len(resp.content) == 1
    assert isinstance(resp.content[0], Message)
    assert resp.content[0].content == "Hello!"
    assert resp.usage_metadata["input_tokens"] == 10
    assert resp.usage_metadata["output_tokens"] == 5


def test_parse_response_tool_calls():
    client = LlmClient("anthropic/claude-haiku-4-5-20251001")
    tc_obj = _make_tool_call_obj("tc-1", "calculator", {"op": "add", "a": 1, "b": 2})
    raw = _make_raw_response(tool_calls=[tc_obj])
    resp = client._parse_response(raw)
    assert len(resp.content) == 1
    assert isinstance(resp.content[0], ToolCall)
    assert resp.content[0].name == "calculator"
    assert resp.content[0].arguments == {"op": "add", "a": 1, "b": 2}


def test_parse_response_text_and_tool_calls():
    client = LlmClient("anthropic/claude-haiku-4-5-20251001")
    tc_obj = _make_tool_call_obj("tc-2", "search", {"q": "test"})
    raw = _make_raw_response(content="Let me search.", tool_calls=[tc_obj])
    resp = client._parse_response(raw)
    assert len(resp.content) == 2
    assert isinstance(resp.content[0], Message)
    assert isinstance(resp.content[1], ToolCall)


# ── LlmClient.generate error handling tests ──────────────────────────────────


async def test_generate_network_error_returns_error_message():
    client = LlmClient("anthropic/claude-haiku-4-5-20251001")
    req = LlmRequest(contents=[Message(role="user", content="hi")], tools=[])

    with patch(
        "litellm.acompletion",
        AsyncMock(side_effect=litellm.APIConnectionError(
            message="connection failed", llm_provider="anthropic", model="claude-haiku"
        )),
    ):
        resp = await client.generate(req)

    assert resp.error_message is not None
    assert "connection" in resp.error_message.lower()
    assert resp.content == []


async def test_generate_auth_error_reraises():
    client = LlmClient("anthropic/claude-haiku-4-5-20251001")
    req = LlmRequest(contents=[Message(role="user", content="hi")], tools=[])

    with patch(
        "litellm.acompletion",
        AsyncMock(side_effect=litellm.AuthenticationError(
            message="Invalid API key", llm_provider="anthropic", model="claude-haiku"
        )),
    ), pytest.raises(litellm.AuthenticationError):
        await client.generate(req)


async def test_generate_success():
    client = LlmClient("anthropic/claude-haiku-4-5-20251001")
    req = LlmRequest(
        instructions=["Be helpful."],
        contents=[Message(role="user", content="hi")],
        tools=[],
    )
    raw = _make_raw_response(content="Hello back!")

    with patch("litellm.acompletion", AsyncMock(return_value=raw)):
        resp = await client.generate(req)

    assert resp.error_message is None
    assert isinstance(resp.content[0], Message)
    assert resp.content[0].content == "Hello back!"


# ── Live tests ────────────────────────────────────────────────────────────────


@pytest.mark.live
async def test_llm_client_generate_simple_message():
    from agentkit.config import FAST_MODEL

    client = LlmClient(FAST_MODEL)
    req = LlmRequest(
        instructions=["You are a helpful assistant."],
        contents=[Message(role="user", content="Say exactly: hello")],
        tools=[],
    )
    resp = await client.generate(req)
    assert resp.error_message is None
    assert any(isinstance(c, Message) for c in resp.content)


@pytest.mark.live
async def test_llm_client_generate_with_tool_returns_tool_call():
    from agentkit.config import FAST_MODEL
    from agentkit.tools_manual import calculator

    calc_tool = FunctionTool(calculator)
    client = LlmClient(FAST_MODEL)
    req = LlmRequest(
        instructions=["Use the calculator tool to answer math questions."],
        contents=[Message(role="user", content="What is 6 multiplied by 7?")],
        tools=[calc_tool],
    )
    resp = await client.generate(req)
    assert resp.error_message is None
    assert any(isinstance(c, ToolCall) and c.name == "calculator" for c in resp.content)
