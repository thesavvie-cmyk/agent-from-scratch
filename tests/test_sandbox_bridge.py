"""Unit and live tests for block 18: sandbox bridge and workspace tools."""
from __future__ import annotations

import asyncio
import json
import threading
import urllib.error
import urllib.request
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from agentkit.context import ExecutionContext
from agentkit.tools.sandbox_bridge import SandboxBridge

# ── Bridge helpers ─────────────────────────────────────────────────────────────


def _make_echo_tool(name: str = "echo_tool") -> Any:
    """FunctionTool that returns its kwargs as JSON."""
    from agentkit.tools.base import tool as tool_decorator

    # Build a uniquely-named function per call so @tool gets the right name
    async def _fn(context: Any, **kwargs: Any) -> str:  # type: ignore[misc]
        return json.dumps(kwargs)

    _fn.__name__ = name
    _fn.__qualname__ = name
    _fn.__doc__ = "Echo the input."
    return tool_decorator(_fn)


def _make_bridge(tools: list | None = None) -> SandboxBridge:
    ctx = ExecutionContext()
    loop = asyncio.new_event_loop()

    # Run the event loop in a background thread so run_coroutine_threadsafe works
    t = threading.Thread(target=loop.run_forever, daemon=True)
    t.start()

    bridge = SandboxBridge(tools or [_make_echo_tool()], ctx, loop)
    return bridge


# ── SandboxBridge.start / stop ─────────────────────────────────────────────────


def test_bridge_start_returns_host_port() -> None:
    bridge = _make_bridge()
    host, port = bridge.start()
    try:
        assert host == "127.0.0.1"
        assert 1024 <= port <= 65535
    finally:
        bridge.stop()


def test_bridge_stop_is_idempotent() -> None:
    bridge = _make_bridge()
    bridge.start()
    bridge.stop()
    bridge.stop()  # second stop must not raise


def test_bridge_serves_after_start() -> None:
    bridge = _make_bridge()
    host, port = bridge.start()
    try:
        url = f"http://{host}:{port}/tool/echo_tool"
        data = json.dumps({"text": "hello"}).encode()
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            payload = json.loads(resp.read())
        assert payload == {"result": json.dumps({"text": "hello"})}
    finally:
        bridge.stop()


def test_bridge_unknown_tool_returns_404() -> None:
    bridge = _make_bridge()
    host, port = bridge.start()
    try:
        url = f"http://{host}:{port}/tool/no_such_tool"
        data = b"{}"
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req, timeout=5)
        assert exc_info.value.code == 404
    finally:
        bridge.stop()


def test_bridge_bad_json_returns_400() -> None:
    bridge = _make_bridge()
    host, port = bridge.start()
    try:
        url = f"http://{host}:{port}/tool/echo_tool"
        req = urllib.request.Request(url, data=b"not json", headers={"Content-Type": "application/json"})
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req, timeout=5)
        assert exc_info.value.code == 400
    finally:
        bridge.stop()


def test_bridge_bad_route_returns_404() -> None:
    bridge = _make_bridge()
    host, port = bridge.start()
    try:
        url = f"http://{host}:{port}/wrong/path/here"
        req = urllib.request.Request(url, data=b"{}", headers={"Content-Type": "application/json"})
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req, timeout=5)
        assert exc_info.value.code == 404
    finally:
        bridge.stop()


# ── SandboxBridge.stub_code ────────────────────────────────────────────────────


def test_stub_code_defines_function() -> None:
    bridge = _make_bridge([_make_echo_tool("my_tool")])
    stubs = bridge.stub_code("127.0.0.1", 9999)
    assert "def my_tool(" in stubs
    assert "_call_bridge" in stubs
    assert "9999" in stubs


def test_stub_code_is_executable_python() -> None:
    bridge = _make_bridge([_make_echo_tool("my_tool")])
    stubs = bridge.stub_code("127.0.0.1", 9999)
    ns: dict = {}
    exec(stubs, ns)  # noqa: S102
    assert callable(ns["my_tool"])
    assert callable(ns["_call_bridge"])


def test_stub_code_calls_bridge_on_execution() -> None:
    """Execute stubs and verify the HTTP call reaches the live bridge."""
    bridge = _make_bridge([_make_echo_tool("echo_tool")])
    host, port = bridge.start()
    try:
        stubs = bridge.stub_code(host, port)
        ns: dict = {}
        exec(stubs, ns)  # noqa: S102
        result = ns["echo_tool"](value="test")
        assert json.loads(result) == {"value": "test"}
    finally:
        bridge.stop()


def test_instructions_hint_lists_tools() -> None:
    bridge = _make_bridge([_make_echo_tool("search_web"), _make_echo_tool("lookup")])
    hint = bridge.instructions_hint()
    assert "search_web" in hint
    assert "lookup" in hint


# ── workspace tools ────────────────────────────────────────────────────────────


def _make_sandbox_ctx() -> tuple[ExecutionContext, MagicMock]:
    from e2b import CommandResult, EntryInfo, FileType

    ctx = ExecutionContext()
    sandbox = MagicMock()

    # files mock
    sandbox.files = MagicMock()
    sandbox.files.write = AsyncMock(return_value=None)
    sandbox.files.read = AsyncMock(return_value="file content")
    mock_entry = MagicMock(spec=EntryInfo)
    mock_entry.name = "data.csv"
    mock_entry.type = FileType.FILE
    mock_entry.path = "/home/user/data.csv"
    sandbox.files.list = AsyncMock(return_value=[mock_entry])

    # commands mock
    sandbox.commands = MagicMock()
    sandbox.commands.run = AsyncMock(
        return_value=CommandResult(stdout="hello\n", stderr="", exit_code=0, error=None)
    )

    ctx.code_env = sandbox
    return ctx, sandbox


@pytest.mark.asyncio
async def test_run_command_returns_json() -> None:
    from agentkit.tools.workspace_sandbox import run_command

    ctx, _ = _make_sandbox_ctx()
    raw = await run_command(context=ctx, cmd="echo hello")
    data = json.loads(raw)
    assert data["stdout"] == "hello\n"
    assert data["exit_code"] == 0


@pytest.mark.asyncio
async def test_run_command_passes_cwd() -> None:
    from agentkit.tools.workspace_sandbox import run_command

    ctx, sb = _make_sandbox_ctx()
    await run_command(context=ctx, cmd="ls", cwd="/tmp")
    sb.commands.run.assert_awaited_once()
    _, kwargs = sb.commands.run.call_args
    assert kwargs.get("cwd") == "/tmp"


@pytest.mark.asyncio
async def test_run_command_no_sandbox_raises() -> None:
    from agentkit.tools.workspace_sandbox import run_command

    ctx = ExecutionContext()
    with pytest.raises(RuntimeError, match="no sandbox"):
        await run_command(context=ctx, cmd="echo hi")


@pytest.mark.asyncio
async def test_write_sandbox_file_calls_files_write() -> None:
    from agentkit.tools.workspace_sandbox import write_sandbox_file

    ctx, sb = _make_sandbox_ctx()
    result = await write_sandbox_file(context=ctx, path="/home/user/f.txt", content="hello")
    sb.files.write.assert_awaited_once_with("/home/user/f.txt", "hello")
    assert "5 bytes" in result


@pytest.mark.asyncio
async def test_read_sandbox_file_returns_content() -> None:
    from agentkit.tools.workspace_sandbox import read_sandbox_file

    ctx, _ = _make_sandbox_ctx()
    result = await read_sandbox_file(context=ctx, path="/home/user/f.txt")
    assert result == "file content"


@pytest.mark.asyncio
async def test_list_sandbox_files_returns_json() -> None:
    from agentkit.tools.workspace_sandbox import list_sandbox_files

    ctx, _ = _make_sandbox_ctx()
    raw = await list_sandbox_files(context=ctx, path="/home/user")
    entries = json.loads(raw)
    assert len(entries) == 1
    assert entries[0]["name"] == "data.csv"
    assert entries[0]["type"] == "file"


# ── Agent: workspace=True adds tools ──────────────────────────────────────────


def test_agent_workspace_adds_tools() -> None:
    from agentkit.agent import Agent
    from agentkit.llm import LlmClient

    agent = Agent(model=MagicMock(spec=LlmClient), code_execution="e2b", workspace=True)
    assert "run_command" in agent._toolbox
    assert "write_sandbox_file" in agent._toolbox
    assert "read_sandbox_file" in agent._toolbox
    assert "list_sandbox_files" in agent._toolbox


def test_agent_no_workspace_no_tools() -> None:
    from agentkit.agent import Agent
    from agentkit.llm import LlmClient

    agent = Agent(model=MagicMock(spec=LlmClient), code_execution="e2b", workspace=False)
    assert "run_command" not in agent._toolbox


def test_agent_workspace_without_code_execution_no_tools() -> None:
    """workspace=True without code_execution should NOT add workspace tools."""
    from agentkit.agent import Agent
    from agentkit.llm import LlmClient

    agent = Agent(model=MagicMock(spec=LlmClient), workspace=True)
    assert "run_command" not in agent._toolbox


# ── Agent: bridge started and stopped ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_agent_kill_sandbox_stops_bridge() -> None:
    """_kill_sandbox must call bridge.stop() and clear ctx.state['_bridge']."""
    from agentkit.agent import Agent
    from agentkit.context import ExecutionContext

    agent = Agent(model=MagicMock(), code_execution="e2b")
    ctx = ExecutionContext()
    ctx.code_env = MagicMock()
    ctx.code_env.kill = AsyncMock()

    mock_bridge = MagicMock()
    ctx.state["_bridge"] = mock_bridge

    await agent._kill_sandbox(ctx)

    mock_bridge.stop.assert_called_once()
    assert "_bridge" not in ctx.state
    assert ctx.code_env is None


# ── Live tests ─────────────────────────────────────────────────────────────────


@pytest.mark.live
@pytest.mark.asyncio
async def test_live_workspace_write_read_round_trip() -> None:
    """Real e2b: write a file and read it back."""
    import e2b_code_interpreter as e2b

    from agentkit.config import E2B_TIMEOUT
    from agentkit.tools.workspace_sandbox import read_sandbox_file, write_sandbox_file

    sandbox = await e2b.AsyncSandbox.create(timeout=E2B_TIMEOUT)
    try:
        ctx = ExecutionContext()
        ctx.code_env = sandbox
        msg = await write_sandbox_file(context=ctx, path="/home/user/test.txt", content="hello block18")
        assert "bytes" in msg
        content = await read_sandbox_file(context=ctx, path="/home/user/test.txt")
        assert content == "hello block18"
    finally:
        await sandbox.kill()


@pytest.mark.live
@pytest.mark.asyncio
async def test_live_run_command_echo() -> None:
    """Real e2b: run echo and verify stdout."""
    import e2b_code_interpreter as e2b

    from agentkit.config import E2B_TIMEOUT
    from agentkit.tools.workspace_sandbox import run_command

    sandbox = await e2b.AsyncSandbox.create(timeout=E2B_TIMEOUT)
    try:
        ctx = ExecutionContext()
        ctx.code_env = sandbox
        raw = await run_command(context=ctx, cmd="echo sandbox_ok")
        data = json.loads(raw)
        assert "sandbox_ok" in data["stdout"]
        assert data["exit_code"] == 0
    finally:
        await sandbox.kill()


@pytest.mark.live
@pytest.mark.asyncio
async def test_live_bridge_search_web_from_execute_python() -> None:
    """Real e2b: search_web callable from Python code via bridge (local loop only).

    NOTE: This test works because the test process runs on the same machine as
    the sandbox stub code, allowing 127.0.0.1 loopback.  In production with
    remote e2b sandboxes, replace 127.0.0.1 with a public tunnel address.
    """
    import e2b_code_interpreter as e2b

    from agentkit.config import E2B_TIMEOUT

    # Use write_sandbox_file as the bridged tool (no Tavily key needed)
    ctx = ExecutionContext()
    loop = asyncio.get_running_loop()
    sandbox = await e2b.AsyncSandbox.create(timeout=E2B_TIMEOUT)
    ctx.code_env = sandbox

    try:
        echo = _make_echo_tool("test_echo")
        bridge = SandboxBridge([echo], ctx, loop)
        host, port = bridge.start()
        stubs = bridge.stub_code(host, port)
        await sandbox.run_code(stubs)

        # Call the bridged tool from inside execute_python
        await sandbox.run_code("result = test_echo(value='from_sandbox')")
        # The result is stored in sandbox variable; read it back
        exec_result2 = await sandbox.run_code("result")
        assert exec_result2.results
        result_text = exec_result2.results[0].text
        assert "from_sandbox" in result_text
    finally:
        bridge.stop()
        await sandbox.kill()
