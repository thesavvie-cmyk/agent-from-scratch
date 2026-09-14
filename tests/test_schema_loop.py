"""Block 4 tests: schema generation and simple_agent_loop."""
from __future__ import annotations

import logging
from typing import Literal
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from agentkit.loop import simple_agent_loop
from agentkit.schema import function_to_input_schema, function_to_tool_definition
from agentkit.tools.web import format_results

# ── Helpers ───────────────────────────────────────────────────────────────────


def func_required(query: str, count: int) -> str:
    """Sample function with only required params."""
    return query * count


def func_mixed(query: str, max_results: int = 5, flag: bool = False) -> list:
    """Sample function with required and optional params."""
    return []


def func_optional_str(query: str, extra: str | None = None) -> str:
    """Function using Optional[str]."""
    return query


def func_union_str(query: str, extra: str | None = None) -> str:
    """Function using str | None."""
    return query


def func_no_annotation(query, count) -> str:
    """Function without annotations."""
    return ""


# ── Schema unit tests ─────────────────────────────────────────────────────────


def test_function_to_input_schema_required_only():
    schema = function_to_input_schema(func_required)
    assert schema["type"] == "object"
    assert set(schema["required"]) == {"query", "count"}
    assert schema["properties"]["query"]["type"] == "string"
    assert schema["properties"]["count"]["type"] == "integer"


def test_function_to_input_schema_mixed_params():
    schema = function_to_input_schema(func_mixed)
    assert schema["required"] == ["query"]
    assert "max_results" not in schema["required"]
    assert "flag" not in schema["required"]
    assert schema["properties"]["max_results"]["type"] == "integer"
    assert schema["properties"]["flag"]["type"] == "boolean"


def test_function_to_input_schema_optional_typing():
    schema = function_to_input_schema(func_optional_str)
    assert schema["properties"]["extra"]["type"] == "string"
    assert "extra" not in schema["required"]


def test_function_to_input_schema_union_pipe_syntax():
    schema = function_to_input_schema(func_union_str)
    assert schema["properties"]["extra"]["type"] == "string"
    assert "extra" not in schema["required"]


def test_unknown_annotation_logs_warning(caplog):
    def func_custom(x: list[int]) -> str:
        return ""

    with caplog.at_level(logging.WARNING, logger="agentkit.schema"):
        schema = function_to_input_schema(func_custom)

    assert schema["properties"]["x"]["type"] == "string"
    assert any("unsupported annotation" in r.message for r in caplog.records)


def test_no_annotation_logs_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="agentkit.schema"):
        schema = function_to_input_schema(func_no_annotation)

    assert schema["properties"]["query"]["type"] == "string"
    assert any("no type annotation" in r.message for r in caplog.records)


def test_function_to_tool_definition_shape():
    defn = function_to_tool_definition(func_required)
    assert defn["type"] == "function"
    fn = defn["function"]
    assert fn["name"] == "func_required"
    assert "Sample function" in fn["description"]
    assert "parameters" in fn


class _InnerModel(BaseModel):
    x: int
    y: str = ""


def test_function_to_input_schema_pydantic_model_param():
    def func(data: _InnerModel, n: int) -> str:
        """A function with a Pydantic model param."""
        return ""

    schema = function_to_input_schema(func)
    data_schema = schema["properties"]["data"]
    assert "properties" in data_schema
    assert "x" in data_schema["properties"]
    assert schema["properties"]["n"]["type"] == "integer"


def test_function_to_input_schema_literal_becomes_enum():
    def func(op: Literal["add", "subtract"], x: int) -> int:
        """Do op."""
        return x

    schema = function_to_input_schema(func)
    assert schema["properties"]["op"]["type"] == "string"
    assert schema["properties"]["op"]["enum"] == ["add", "subtract"]
    assert schema["properties"]["x"]["type"] == "integer"
    assert "enum" not in schema["properties"]["x"]


def test_function_to_tool_definition_no_docstring():
    def no_doc(x: int) -> int:
        return x

    defn = function_to_tool_definition(no_doc)
    assert defn["function"]["description"] == ""


# ── format_results unit tests ─────────────────────────────────────────────────

TAVILY_FIXTURE = [
    {
        "title": "Kipchoge breaks record",
        "url": "https://example.com/kip",
        "content": "Eliud Kipchoge set a new marathon world record.",
        "score": 0.98,
        "raw_content": "<html>...</html>",
    },
    {
        "title": "Another result",
        "url": "https://example.com/other",
        "content": "Some other content here.",
        "score": 0.75,
        "raw_content": "<html>...</html>",
    },
]


def test_format_results_includes_title_url_content():
    formatted = format_results(TAVILY_FIXTURE)
    assert "Kipchoge breaks record" in formatted
    assert "https://example.com/kip" in formatted
    assert "Eliud Kipchoge set a new marathon world record." in formatted


def test_format_results_excludes_score_and_raw_content():
    formatted = format_results(TAVILY_FIXTURE)
    assert "0.98" not in formatted
    assert "raw_content" not in formatted
    assert "<html>" not in formatted


def test_format_results_empty():
    assert format_results([]) == "(no results)"


def test_format_results_multiple_results():
    formatted = format_results(TAVILY_FIXTURE)
    assert "[1]" in formatted
    assert "[2]" in formatted


# ── simple_agent_loop unit tests ──────────────────────────────────────────────


def _make_tool_call(call_id: str, name: str, args: str) -> MagicMock:
    tc = MagicMock()
    tc.id = call_id
    tc.function.name = name
    tc.function.arguments = args
    return tc


def _make_assistant_msg(content: str | None, tool_calls=None) -> MagicMock:
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls or []
    return msg


def _make_response(msg: MagicMock) -> MagicMock:
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message = msg
    return resp


def fake_tool(x: str) -> str:
    """A simple fake tool."""
    return f"result:{x}"


def test_simple_agent_loop_two_rounds_then_answer():
    """Round 1: model calls fake_tool; Round 2: model returns plain answer."""
    tc = _make_tool_call("call-1", "fake_tool", '{"x": "hello"}')
    round1 = _make_response(_make_assistant_msg(None, tool_calls=[tc]))
    round2 = _make_response(_make_assistant_msg("The answer is 42."))

    with patch("litellm.completion", side_effect=[round1, round2]):
        answer, history = simple_agent_loop(
            "You are helpful.", "What is the answer?", [fake_tool], max_rounds=5
        )

    assert answer == "The answer is 42."

    roles = [m["role"] for m in history]
    assert roles == ["system", "user", "assistant", "tool", "assistant"]

    tool_msg = history[3]
    assert tool_msg["tool_call_id"] == "call-1"
    assert tool_msg["content"] == "result:hello"


def test_simple_agent_loop_message_order():
    """Verify exact message sequence: system, user, assistant(tools), tool, assistant."""
    tc = _make_tool_call("call-99", "fake_tool", '{"x": "test"}')
    round1 = _make_response(_make_assistant_msg("thinking", tool_calls=[tc]))
    round2 = _make_response(_make_assistant_msg("Final."))

    with patch("litellm.completion", side_effect=[round1, round2]):
        _, history = simple_agent_loop("sys", "q", [fake_tool], max_rounds=3)

    assert history[0]["role"] == "system"
    assert history[1]["role"] == "user"
    assert history[2]["role"] == "assistant"
    assert history[2]["tool_calls"][0]["id"] == "call-99"
    assert history[3]["role"] == "tool"
    assert history[4]["role"] == "assistant"


def test_simple_agent_loop_tool_exception_becomes_error_string():
    """Tool raising an exception → 'Error: ...' in context, loop continues."""

    def broken_tool(x: str) -> str:
        """A tool that always fails."""
        raise ValueError("kaboom")

    tc = _make_tool_call("call-err", "broken_tool", '{"x": "oops"}')
    round1 = _make_response(_make_assistant_msg(None, tool_calls=[tc]))
    round2 = _make_response(_make_assistant_msg("I handled the error."))

    with patch("litellm.completion", side_effect=[round1, round2]):
        answer, history = simple_agent_loop(
            "sys", "q", [broken_tool], max_rounds=5
        )

    assert answer == "I handled the error."
    tool_msg = next(m for m in history if m["role"] == "tool")
    assert tool_msg["content"].startswith("Error:")
    assert "kaboom" in tool_msg["content"]


def test_simple_agent_loop_unknown_tool_becomes_error_string():
    """Model calls a tool not in toolbox → 'Error: unknown tool' in context."""
    tc = _make_tool_call("call-unk", "nonexistent_tool", "{}")
    round1 = _make_response(_make_assistant_msg(None, tool_calls=[tc]))
    round2 = _make_response(_make_assistant_msg("Got it."))

    with patch("litellm.completion", side_effect=[round1, round2]):
        _answer, history = simple_agent_loop("sys", "q", [fake_tool], max_rounds=5)

    tool_msg = next(m for m in history if m["role"] == "tool")
    assert "unknown tool" in tool_msg["content"]


def test_simple_agent_loop_max_rounds_raises():
    """Exceeding max_rounds raises RuntimeError."""
    tc = _make_tool_call("call-inf", "fake_tool", '{"x": "loop"}')
    always_calls = _make_response(_make_assistant_msg(None, tool_calls=[tc]))

    with patch("litellm.completion", return_value=always_calls), pytest.raises(RuntimeError, match="final answer"):
        simple_agent_loop("sys", "q", [fake_tool], max_rounds=3)


# ── Live tests ────────────────────────────────────────────────────────────────


@pytest.mark.live
def test_search_web_returns_nonempty():
    from agentkit.tools.web import search_web

    results = search_web("Python programming language", max_results=3)
    assert isinstance(results, list)
    assert len(results) > 0
    assert "title" in results[0]
    assert "url" in results[0]
    assert "content" in results[0]


@pytest.mark.live
def test_simple_agent_loop_2025_event_uses_search():
    """Agent answering about a 2025 event should call search_web at least once."""
    from agentkit.tools.web import search_web

    _, history = simple_agent_loop(
        "You are a helpful research assistant.",
        "Who won the Nobel Prize in Physics in 2025?",
        [search_web],
        max_rounds=6,
    )

    search_calls = [m for m in history if m["role"] == "tool"]
    assert len(search_calls) >= 1
