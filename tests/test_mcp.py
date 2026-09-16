"""Block 5 tests: McpToolset schema conversion, text extraction, loop collision."""
from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentkit.loop import async_simple_agent_loop
from agentkit.mcp_client import McpToolset
from agentkit.schema import build_tool_definition

# ── Fixtures / helpers ────────────────────────────────────────────────────────


class FakeMcpTool:
    """Minimal stand-in for an MCP Tool object."""

    def __init__(
        self,
        name: str = "my_tool",
        description: str = "Does something useful.",
        input_schema: dict | None = None,
    ) -> None:
        self.name = name
        self.description = description
        self.inputSchema = input_schema or {
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "required": ["x"],
        }


class FakeTextContent:
    def __init__(self, text: str) -> None:
        self.text = text


class FakeImageContent:
    """Content block without .text — should be ignored."""

    data = b"\x89PNG"


def _toolset_with_tools(*tools: FakeMcpTool) -> McpToolset:
    """Create a McpToolset instance with pre-populated _tools (bypasses __init__)."""
    ts = McpToolset.__new__(McpToolset)
    ts._tools = list(tools)
    return ts


def _make_tool_call(call_id: str, name: str, args: str) -> MagicMock:
    tc = MagicMock()
    tc.id = call_id
    tc.function.name = name
    tc.function.arguments = args
    return tc


def _make_response(content: str | None, tool_calls=None) -> MagicMock:
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls or []
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message = msg
    return resp


# ── build_tool_definition unit tests ─────────────────────────────────────────


def test_build_tool_definition_shape():
    params = {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}
    defn = build_tool_definition("search", "Search the web.", params)
    assert defn["type"] == "function"
    assert defn["function"]["name"] == "search"
    assert defn["function"]["description"] == "Search the web."
    assert defn["function"]["parameters"] is params


# ── MCP schema conversion unit tests ─────────────────────────────────────────


def test_mcp_tool_definitions_openai_format():
    ts = _toolset_with_tools(FakeMcpTool())
    defns = ts.tool_definitions()

    assert len(defns) == 1
    fn = defns[0]["function"]
    assert fn["name"] == "my_tool"
    assert fn["description"] == "Does something useful."
    assert fn["parameters"]["properties"]["x"]["type"] == "string"


def test_mcp_tool_definitions_multiple_tools():
    ts = _toolset_with_tools(
        FakeMcpTool("tool_a", "Tool A"),
        FakeMcpTool("tool_b", "Tool B"),
    )
    names = [d["function"]["name"] for d in ts.tool_definitions()]
    assert names == ["tool_a", "tool_b"]


def test_mcp_tool_definitions_empty_description():
    ts = _toolset_with_tools(FakeMcpTool(description=""))
    assert ts.tool_definitions()[0]["function"]["description"] == ""


# ── Text extraction unit tests ────────────────────────────────────────────────


def test_extract_text_single_block():
    result = McpToolset._extract_text([FakeTextContent("hello world")])
    assert result == "hello world"


def test_extract_text_multiple_blocks():
    result = McpToolset._extract_text([
        FakeTextContent("line 1"),
        FakeTextContent("line 2"),
    ])
    assert result == "line 1\nline 2"


def test_extract_text_ignores_non_text_blocks():
    result = McpToolset._extract_text([FakeImageContent(), FakeTextContent("text only")])
    assert result == "text only"


def test_extract_text_empty():
    assert McpToolset._extract_text([]) == "(empty response)"


def test_extract_text_all_non_text():
    assert McpToolset._extract_text([FakeImageContent()]) == "(empty response)"


# ── McpToolset.call() unit test ───────────────────────────────────────────────


async def test_mcp_call_returns_text():
    ts = McpToolset.__new__(McpToolset)
    fake_result = MagicMock()
    fake_result.content = [FakeTextContent("42.0")]
    mock_session = AsyncMock()
    mock_session.call_tool.return_value = fake_result
    ts._session = mock_session

    result = await ts.call("calculator", {"operator": "add", "first_number": 40, "second_number": 2})
    assert result == "42.0"
    mock_session.call_tool.assert_called_once_with(
        "calculator", {"operator": "add", "first_number": 40, "second_number": 2}
    )


async def test_mcp_call_exception_becomes_error_string():
    ts = McpToolset.__new__(McpToolset)
    mock_session = AsyncMock()
    mock_session.call_tool.side_effect = RuntimeError("server crashed")
    ts._session = mock_session

    result = await ts.call("some_tool", {})
    assert result.startswith("Error:")
    assert "server crashed" in result


# ── Loop collision unit tests ─────────────────────────────────────────────────


async def test_loop_mcp_collision_local_wins(caplog):
    """Local tool with same name as MCP tool — local must win, warning logged."""

    def fake_tool(x: str) -> str:
        """Local fake tool."""
        return f"local:{x}"

    mcp_ts = _toolset_with_tools(FakeMcpTool(name="fake_tool", description="MCP version"))

    tc = _make_tool_call("c1", "fake_tool", '{"x": "hi"}')
    r1 = _make_response(None, tool_calls=[tc])
    r2 = _make_response("Done.")

    with patch("litellm.completion", side_effect=[r1, r2]), caplog.at_level(logging.WARNING, logger="agentkit.loop"):
        answer, history = await async_simple_agent_loop(
            "sys", "q", [fake_tool], mcp_toolsets=[mcp_ts]
        )

    assert answer == "Done."
    tool_msg = next(m for m in history if m["role"] == "tool")
    # Local tool won: result contains "local:" prefix, not an MCP response
    assert tool_msg["content"] == "local:hi"
    assert any("collision" in r.message.lower() for r in caplog.records)


async def test_loop_mcp_tool_called_when_no_local():
    """MCP tool is called when name is not in local toolbox."""
    mcp_ts = _toolset_with_tools(FakeMcpTool(name="mcp_tool"))
    mcp_ts._session = AsyncMock()
    fake_result = MagicMock()
    fake_result.content = [FakeTextContent("mcp_result")]
    mcp_ts._session.call_tool.return_value = fake_result

    tc = _make_tool_call("c1", "mcp_tool", '{"x": "test"}')
    r1 = _make_response(None, tool_calls=[tc])
    r2 = _make_response("Got mcp_result.")

    with patch("litellm.completion", side_effect=[r1, r2]):
        _answer, history = await async_simple_agent_loop(
            "sys", "q", [], mcp_toolsets=[mcp_ts]
        )

    tool_msg = next(m for m in history if m["role"] == "tool")
    assert tool_msg["content"] == "mcp_result"


# ── Live tests ────────────────────────────────────────────────────────────────


@pytest.mark.live
async def test_mcp_custom_server_list_tools():
    """Custom server exposes search_web and calculator."""
    async with McpToolset("uv", ["run", "python", "-m", "agentkit.servers.tavily_server"]) as ts:
        names = [d["function"]["name"] for d in ts.tool_definitions()]
    assert "search_web" in names
    assert "calculator" in names


@pytest.mark.live
async def test_mcp_custom_server_calculator():
    """Calculator via MCP returns correct result."""
    async with McpToolset("uv", ["run", "python", "-m", "agentkit.servers.tavily_server"]) as ts:
        result = await ts.call("calculator", {"operator": "multiply", "first_number": 6.0, "second_number": 7.0})
    assert "42" in result
