"""Unit and live tests for block 17: code execution via e2b sandbox."""
from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentkit.context import ExecutionContext
from agentkit.tools.code_execution import execute_python
from agentkit.types import Message, ToolCall

# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_execution(
    stdout: list[str] | None = None,
    result_text: str | None = None,
    error_name: str | None = None,
    error_value: str | None = None,
    error_tb: str | None = None,
) -> MagicMock:
    """Build a mock e2b Execution object."""
    logs = MagicMock()
    logs.stdout = stdout or []

    results = []
    if result_text is not None:
        r = MagicMock()
        r.is_main_result = True
        r.text = result_text
        results.append(r)

    error = None
    if error_name is not None:
        error = MagicMock()
        error.name = error_name
        error.value = error_value or ""
        error.traceback = error_tb or ""

    execution = MagicMock()
    execution.logs = logs
    execution.results = results
    execution.error = error
    return execution


def _make_sandbox(execution: MagicMock) -> MagicMock:
    sandbox = MagicMock()
    sandbox.run_code = AsyncMock(return_value=execution)
    sandbox.kill = AsyncMock()
    sandbox.sandbox_id = "mock-sandbox-id"
    return sandbox


def _mock_llm(*responses: Any) -> MagicMock:
    from agentkit.llm import LlmClient
    m = MagicMock(spec=LlmClient)
    m.generate = AsyncMock(side_effect=list(responses))
    return m


def _tool_resp(call_id: str, name: str, args: dict) -> Any:
    from agentkit.llm import LlmResponse
    return LlmResponse(content=[ToolCall(tool_call_id=call_id, name=name, arguments=args)])


def _text_resp(text: str) -> Any:
    from agentkit.llm import LlmResponse
    return LlmResponse(content=[Message(role="assistant", content=text)])


# ── execute_python: no sandbox raises configuration error ─────────────────────


@pytest.mark.asyncio
async def test_execute_python_no_sandbox_raises() -> None:
    ctx = ExecutionContext()
    assert ctx.code_env is None
    with pytest.raises(RuntimeError, match="no sandbox available"):
        await execute_python(context=ctx, code="1+1")


# ── execute_python: success path returns JSON with stdout and result ───────────


@pytest.mark.asyncio
async def test_execute_python_success_stdout_and_result() -> None:
    ctx = ExecutionContext()
    ctx.code_env = _make_sandbox(_make_execution(
        stdout=["hello", "world"],
        result_text="42",
    ))
    raw = await execute_python(context=ctx, code="print('hello'); print('world'); 42")
    data = json.loads(raw)
    assert data["stdout"] == ["hello", "world"]
    assert data["result"] == "42"
    assert data["error"] is None


@pytest.mark.asyncio
async def test_execute_python_no_output_fields_null() -> None:
    ctx = ExecutionContext()
    ctx.code_env = _make_sandbox(_make_execution())
    raw = await execute_python(context=ctx, code="x = 1")
    data = json.loads(raw)
    assert data["stdout"] == []
    assert data["result"] is None
    assert data["error"] is None


# ── execute_python: execution error returned as JSON, not raised ──────────────


@pytest.mark.asyncio
async def test_execute_python_execution_error_in_json() -> None:
    ctx = ExecutionContext()
    ctx.code_env = _make_sandbox(_make_execution(
        error_name="ZeroDivisionError",
        error_value="division by zero",
        error_tb="Traceback ...\nZeroDivisionError: division by zero",
    ))
    raw = await execute_python(context=ctx, code="1/0")
    data = json.loads(raw)
    assert data["error"] is not None
    assert data["error"]["type"] == "ZeroDivisionError"
    assert "division by zero" in data["error"]["message"]
    assert data["error"]["traceback"] != ""


@pytest.mark.asyncio
async def test_execute_python_execution_error_does_not_raise() -> None:
    """Execution errors must not bubble up as exceptions — model fixes the code."""
    ctx = ExecutionContext()
    ctx.code_env = _make_sandbox(_make_execution(
        error_name="NameError", error_value="name 'x' is not defined", error_tb="..."
    ))
    # Should not raise
    result = await execute_python(context=ctx, code="print(x)")
    assert "NameError" in result


# ── Agent: code_execution="e2b" adds execute_python to toolbox ────────────────


def test_agent_code_execution_adds_tool() -> None:
    from agentkit.agent import Agent
    from agentkit.llm import LlmClient

    agent = Agent(model=MagicMock(spec=LlmClient), code_execution="e2b")
    assert "execute_python" in agent._toolbox


def test_agent_no_code_execution_no_tool() -> None:
    from agentkit.agent import Agent
    from agentkit.llm import LlmClient

    agent = Agent(model=MagicMock(spec=LlmClient))
    assert "execute_python" not in agent._toolbox


# ── Agent: sandbox lifecycle — kill() called in all exit paths ────────────────


@pytest.mark.asyncio
async def test_agent_kill_called_on_normal_completion() -> None:
    from agentkit.agent import Agent

    mock_sandbox = MagicMock()
    mock_sandbox.kill = AsyncMock()
    mock_sandbox.sandbox_id = "s1"
    mock_sandbox.run_code = AsyncMock(return_value=_make_execution(result_text="4"))

    model = _mock_llm(
        _tool_resp("c1", "execute_python", {"code": "2+2"}),
        _text_resp("The answer is 4."),
    )
    agent = Agent(model=model, code_execution="e2b")

    with patch.object(agent, "_create_sandbox", AsyncMock(side_effect=lambda ctx: setattr(ctx, "code_env", mock_sandbox))):
        await agent.run("What is 2+2?")

    mock_sandbox.kill.assert_awaited_once()


@pytest.mark.asyncio
async def test_agent_kill_called_on_max_steps() -> None:
    from agentkit.agent import Agent
    from agentkit.llm import LlmResponse

    mock_sandbox = MagicMock()
    mock_sandbox.kill = AsyncMock()
    mock_sandbox.sandbox_id = "s2"

    # Always return a tool call — never terminates, hits max_steps
    always_call = LlmResponse(
        content=[ToolCall(tool_call_id="c1", name="execute_python", arguments={"code": "1"})],
        usage_metadata={},
    )
    mock_sandbox.run_code = AsyncMock(return_value=_make_execution(result_text="1"))

    model = MagicMock()
    model.generate = AsyncMock(return_value=always_call)
    agent = Agent(model=model, code_execution="e2b", max_steps=2)

    with patch.object(agent, "_create_sandbox", AsyncMock(side_effect=lambda ctx: setattr(ctx, "code_env", mock_sandbox))):
        result = await agent.run("Run forever.")

    assert "max_steps" in str(result.output)
    mock_sandbox.kill.assert_awaited_once()


@pytest.mark.asyncio
async def test_agent_kill_called_on_exception() -> None:
    from agentkit.agent import Agent

    mock_sandbox = MagicMock()
    mock_sandbox.kill = AsyncMock()
    mock_sandbox.sandbox_id = "s3"

    model = MagicMock()
    model.generate = AsyncMock(side_effect=RuntimeError("Unexpected LLM crash"))
    agent = Agent(model=model, code_execution="e2b", max_steps=3)

    with (
        patch.object(agent, "_create_sandbox", AsyncMock(side_effect=lambda ctx: setattr(ctx, "code_env", mock_sandbox))),
        pytest.raises(RuntimeError, match="Unexpected LLM crash"),
    ):
        await agent.run("Crash.")

    mock_sandbox.kill.assert_awaited_once()


# ── Agent: no sandbox created when code_execution is None ─────────────────────


@pytest.mark.asyncio
async def test_agent_no_sandbox_when_code_execution_none() -> None:
    from agentkit.agent import Agent

    model = _mock_llm(_text_resp("Done."))
    agent = Agent(model=model)
    result = await agent.run("Simple question.")
    assert result.context.code_env is None


# ── Live tests ─────────────────────────────────────────────────────────────────


@pytest.mark.live
@pytest.mark.asyncio
async def test_live_sandbox_basic_arithmetic() -> None:
    """Real e2b sandbox: 2+2 returns 4."""
    import e2b_code_interpreter as e2b

    from agentkit.config import E2B_TIMEOUT

    sandbox = await e2b.AsyncSandbox.create(timeout=E2B_TIMEOUT)
    try:
        ctx = ExecutionContext()
        ctx.code_env = sandbox
        raw = await execute_python(context=ctx, code="2 + 2")
        data = json.loads(raw)
        assert data["result"] == "4"
        assert data["error"] is None
    finally:
        await sandbox.kill()


@pytest.mark.live
@pytest.mark.asyncio
async def test_live_sandbox_state_persists_between_calls() -> None:
    """Variables assigned in one call survive into the next."""
    import e2b_code_interpreter as e2b

    from agentkit.config import E2B_TIMEOUT

    sandbox = await e2b.AsyncSandbox.create(timeout=E2B_TIMEOUT)
    try:
        ctx = ExecutionContext()
        ctx.code_env = sandbox
        await execute_python(context=ctx, code="x = 42")
        raw2 = await execute_python(context=ctx, code="x * 2")
        data = json.loads(raw2)
        assert data["result"] == "84"
    finally:
        await sandbox.kill()


@pytest.mark.live
@pytest.mark.asyncio
async def test_live_sandbox_error_does_not_kill_sandbox() -> None:
    """Division by zero returns error JSON; sandbox stays alive for next call."""
    import e2b_code_interpreter as e2b

    from agentkit.config import E2B_TIMEOUT

    sandbox = await e2b.AsyncSandbox.create(timeout=E2B_TIMEOUT)
    try:
        ctx = ExecutionContext()
        ctx.code_env = sandbox
        raw1 = await execute_python(context=ctx, code="1/0")
        data1 = json.loads(raw1)
        assert data1["error"] is not None
        assert data1["error"]["type"] == "ZeroDivisionError"

        # Sandbox must still be usable
        raw2 = await execute_python(context=ctx, code="1 + 1")
        data2 = json.loads(raw2)
        assert data2["result"] == "2"
        assert data2["error"] is None
    finally:
        await sandbox.kill()
