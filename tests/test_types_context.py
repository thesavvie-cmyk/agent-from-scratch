"""Block 6 tests: types, ExecutionContext, transcript."""
from __future__ import annotations

import json

from agentkit.context import ExecutionContext
from agentkit.transcript import render, to_messages
from agentkit.types import Event, Message, ToolCall, ToolResult

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_call(call_id: str = "tc-1", name: str = "search") -> ToolCall:
    return ToolCall(tool_call_id=call_id, name=name, arguments={"q": "test"})


def _make_result(call_id: str = "tc-1", name: str = "search") -> ToolResult:
    return ToolResult(
        tool_call_id=call_id, name=name, status="success", content=["result text"]
    )


def _make_ctx() -> ExecutionContext:
    """Realistic 4-event trace: user → tool_calls → tool_results → assistant."""
    ctx = ExecutionContext()
    ctx.add_message("user", "What is 2+2?", author="user")
    ctx.add_tool_calls(
        [_make_call("tc-1", "calc"), _make_call("tc-2", "search")], author="agent"
    )
    ctx.add_tool_results(
        [_make_result("tc-1", "calc"), _make_result("tc-2", "search")],
        author="tool_runner",
    )
    ctx.add_message("assistant", "The answer is 4.", author="agent")
    return ctx


# ── Type round-trip (discriminator) tests ─────────────────────────────────────


def test_event_roundtrip_message():
    event = Event(
        execution_id="exec-1",
        author="test",
        content=[Message(role="user", content="hello")],
    )
    restored = Event.model_validate_json(event.model_dump_json())
    assert type(restored.content[0]) is Message
    assert restored.content[0].content == "hello"


def test_event_roundtrip_tool_call():
    event = Event(
        execution_id="exec-1",
        author="test",
        content=[_make_call()],
    )
    restored = Event.model_validate_json(event.model_dump_json())
    assert type(restored.content[0]) is ToolCall
    assert restored.content[0].name == "search"


def test_event_roundtrip_tool_result():
    event = Event(
        execution_id="exec-1",
        author="test",
        content=[_make_result()],
    )
    restored = Event.model_validate_json(event.model_dump_json())
    assert type(restored.content[0]) is ToolResult
    assert restored.content[0].status == "success"


def test_event_roundtrip_all_types_discriminator():
    """Discriminator must restore correct concrete types for all three variants."""
    event = Event(
        execution_id="exec-1",
        author="test",
        content=[
            Message(role="assistant", content="thinking"),
            _make_call("tc-x"),
            _make_result("tc-x"),
        ],
    )
    restored = Event.model_validate_json(event.model_dump_json())
    assert type(restored.content[0]) is Message
    assert type(restored.content[1]) is ToolCall
    assert type(restored.content[2]) is ToolResult


def test_event_roundtrip_preserves_values():
    call = ToolCall(
        tool_call_id="abc-123",
        name="my_tool",
        arguments={"key": "value", "num": 42},
    )
    event = Event(execution_id="exec-42", author="bot", content=[call])
    restored = Event.model_validate_json(event.model_dump_json())
    tc = restored.content[0]
    assert isinstance(tc, ToolCall)
    assert tc.tool_call_id == "abc-123"
    assert tc.arguments["num"] == 42


# ── ExecutionContext tests ─────────────────────────────────────────────────────


def test_context_execution_id_stamped_on_events():
    ctx = ExecutionContext()
    e1 = ctx.add_message("user", "hi", author="user")
    e2 = ctx.add_tool_calls([_make_call()], author="agent")
    assert e1.execution_id == ctx.execution_id
    assert e2.execution_id == ctx.execution_id


def test_context_event_order_preserved():
    ctx = ExecutionContext()
    ctx.add_message("user", "first", author="user")
    ctx.add_message("assistant", "second", author="agent")
    assert len(ctx.events) == 2
    first_msg = ctx.events[0].content[0]
    assert isinstance(first_msg, Message)
    assert first_msg.content == "first"


def test_context_increment_step():
    ctx = ExecutionContext()
    assert ctx.current_step == 0
    ctx.increment_step()
    ctx.increment_step()
    assert ctx.current_step == 2


def test_context_last_event_empty():
    ctx = ExecutionContext()
    assert ctx.last_event() is None


def test_context_last_event():
    ctx = ExecutionContext()
    ctx.add_message("user", "hello", author="user")
    e = ctx.add_message("assistant", "hi back", author="agent")
    assert ctx.last_event() is e


def test_context_iter_content_order():
    ctx = ExecutionContext()
    ctx.add_message("user", "q", author="user")
    ctx.add_tool_calls([_make_call("tc-1"), _make_call("tc-2")], author="agent")
    ctx.add_tool_results([_make_result("tc-1")], author="tools")

    items = list(ctx.iter_content())
    assert len(items) == 4
    assert isinstance(items[0], Message)
    assert isinstance(items[1], ToolCall)
    assert isinstance(items[2], ToolCall)
    assert isinstance(items[3], ToolResult)


def test_context_find_tool_call_found():
    ctx = ExecutionContext()
    ctx.add_tool_calls([_make_call("tc-abc")], author="agent")
    found = ctx.find_tool_call("tc-abc")
    assert found is not None
    assert found.tool_call_id == "tc-abc"


def test_context_find_tool_call_not_found():
    ctx = ExecutionContext()
    ctx.add_tool_calls([_make_call("tc-abc")], author="agent")
    assert ctx.find_tool_call("tc-xyz") is None


# ── to_messages tests ──────────────────────────────────────────────────────────


def test_to_messages_two_tool_calls_one_assistant_message():
    """Two ToolCall items in one event → ONE assistant message with two tool_calls."""
    ctx = ExecutionContext()
    ctx.add_tool_calls(
        [_make_call("tc-1", "search"), _make_call("tc-2", "calc")], author="agent"
    )
    messages = to_messages(ctx)

    assert len(messages) == 1
    assert messages[0]["role"] == "assistant"
    assert len(messages[0]["tool_calls"]) == 2
    ids = [tc["id"] for tc in messages[0]["tool_calls"]]
    assert "tc-1" in ids
    assert "tc-2" in ids


def test_to_messages_tool_results_separate():
    """Each ToolResult becomes its own tool message with correct tool_call_id."""
    ctx = ExecutionContext()
    ctx.add_tool_results(
        [_make_result("tc-1", "search"), _make_result("tc-2", "calc")],
        author="tools",
    )
    messages = to_messages(ctx)

    assert len(messages) == 2
    for m in messages:
        assert m["role"] == "tool"
    call_ids = {m["tool_call_id"] for m in messages}
    assert call_ids == {"tc-1", "tc-2"}


def test_to_messages_tool_call_arguments_json():
    """ToolCall arguments are serialised as a JSON string in the messages dict."""
    ctx = ExecutionContext()
    ctx.add_tool_calls(
        [ToolCall(tool_call_id="tc-1", name="calc", arguments={"op": "add", "a": 1, "b": 2})],
        author="agent",
    )
    messages = to_messages(ctx)
    args_str = messages[0]["tool_calls"][0]["function"]["arguments"]
    args = json.loads(args_str)
    assert args == {"op": "add", "a": 1, "b": 2}


def test_to_messages_realistic_role_sequence():
    """user → assistant(tool_calls) → tool → assistant gives correct role order."""
    ctx = _make_ctx()
    messages = to_messages(ctx)
    roles = [m["role"] for m in messages]
    assert roles == ["user", "assistant", "tool", "tool", "assistant"]


def test_to_messages_tool_call_id_matches_result():
    """tool_call_id in assistant message matches the corresponding tool message."""
    ctx = ExecutionContext()
    ctx.add_tool_calls([_make_call("unique-id")], author="agent")
    ctx.add_tool_results([_make_result("unique-id")], author="tools")

    messages = to_messages(ctx)
    assistant_id = messages[0]["tool_calls"][0]["id"]
    tool_id = messages[1]["tool_call_id"]
    assert assistant_id == tool_id == "unique-id"


def test_to_messages_empty_context():
    ctx = ExecutionContext()
    assert to_messages(ctx) == []


# ── render tests ──────────────────────────────────────────────────────────────


def test_render_empty_context():
    ctx = ExecutionContext()
    result = render(ctx)
    assert "ExecutionContext" in result
    assert "step=0" in result


def test_render_nonempty():
    ctx = _make_ctx()
    result = render(ctx)
    assert "user" in result
    assert "agent" in result
    assert "Message" in result
    assert "ToolCall" in result
    assert "ToolResult" in result
