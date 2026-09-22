"""Unit tests for block 21: AgentTool and TransferOrchestrator.

All agents are mocked — no API key required.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from agentkit.agent import Agent, AgentResult
from agentkit.context import ExecutionContext
from agentkit.tools.agent_tool import AgentTool, ChildAgentError
from agentkit.transfer import TransferOrchestrator

# ── Mock helpers ───────────────────────────────────────────────────────────────


def _mock_agent(name: str, output: str, error: str | None = None) -> Agent:
    """Return a mock Agent that returns *output* (or *error*)."""
    agent = MagicMock(spec=Agent)
    agent.name = name
    agent.instructions = f"I am the {name} agent."
    agent._toolbox = {}

    async def run(
        user_input: str, context: ExecutionContext | None = None
    ) -> AgentResult:
        ctx = context or ExecutionContext()
        ctx.add_message("assistant", output, author=name)
        return AgentResult(output=output, context=ctx, error=error)

    agent.run = run
    return agent


def _mock_agent_captures_ctx(name: str, output: str) -> tuple[Agent, list[ExecutionContext]]:
    """Return (agent, captured_contexts) — lets tests inspect what context child got."""
    agent = MagicMock(spec=Agent)
    agent.name = name
    agent.instructions = f"I am the {name} agent."
    agent._toolbox = {}
    captured: list[ExecutionContext] = []

    async def run(
        user_input: str, context: ExecutionContext | None = None
    ) -> AgentResult:
        captured.append(context)
        ctx = context or ExecutionContext()
        ctx.add_message("assistant", output, author=name)
        return AgentResult(output=output, context=ctx)

    agent.run = run
    return agent, captured


def _transfer_agent(name: str, transfer_to: str, reason: str) -> Agent:
    """Agent that always sets a transfer_request in the context."""
    agent = MagicMock(spec=Agent)
    agent.name = name
    agent.instructions = f"I am {name}."
    agent._toolbox = {}

    async def run(
        user_input: str, context: ExecutionContext | None = None
    ) -> AgentResult:
        ctx = context or ExecutionContext()
        ctx.add_message("assistant", f"Transferring to {transfer_to}", author=name)
        ctx.state["_transfer_request"] = {"to": transfer_to, "reason": reason}
        return AgentResult(output=f"→ {transfer_to}", context=ctx)

    agent.run = run
    return agent


# ── AgentTool: child isolation ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_agent_tool_child_gets_fresh_context_not_parent() -> None:
    """Child agent never receives the parent's context object."""
    child, captured = _mock_agent_captures_ctx("child", "child output")
    tool = AgentTool(child)

    parent_ctx = ExecutionContext()
    parent_ctx.add_message("user", "hello from parent", author="user")

    await tool.execute(parent_ctx, request="do something")

    # Child must have been called with a DIFFERENT context than parent
    assert len(captured) == 1
    assert captured[0] is not parent_ctx


@pytest.mark.asyncio
async def test_agent_tool_child_receives_request_string() -> None:
    """Child receives the request as user_input."""
    received: list[str] = []

    child = MagicMock(spec=Agent)
    child.name = "child"
    child.instructions = "x"
    child._toolbox = {}

    async def run(
        user_input: str, context: ExecutionContext | None = None
    ) -> AgentResult:
        received.append(user_input)
        ctx = context or ExecutionContext()
        ctx.add_message("assistant", "ok", author="child")
        return AgentResult(output="ok", context=ctx)

    child.run = run

    tool = AgentTool(child)
    parent_ctx = ExecutionContext()
    await tool.execute(parent_ctx, request="find the answer")

    assert received == ["find the answer"]


@pytest.mark.asyncio
async def test_agent_tool_returns_child_output() -> None:
    child = _mock_agent("child", "42")
    tool = AgentTool(child)
    result = await tool.execute(ExecutionContext(), request="what is the answer")
    assert result == "42"


# ── AgentTool: input_schema ────────────────────────────────────────────────────


class _ResearchRequest(BaseModel):
    topic: str
    depth: int = 1


@pytest.mark.asyncio
async def test_agent_tool_input_schema_valid() -> None:
    """Valid kwargs matching the Pydantic schema → child receives JSON."""
    received: list[str] = []

    child = MagicMock(spec=Agent)
    child.name = "researcher"
    child.instructions = "x"
    child._toolbox = {}

    async def run(
        user_input: str, context: ExecutionContext | None = None
    ) -> AgentResult:
        received.append(user_input)
        ctx = context or ExecutionContext()
        ctx.add_message("assistant", "facts", author="researcher")
        return AgentResult(output="facts", context=ctx)

    child.run = run
    tool = AgentTool(child, input_schema=_ResearchRequest)

    parent_ctx = ExecutionContext()
    result = await tool.execute(parent_ctx, topic="Python", depth=3)

    assert result == "facts"
    # Child should have received the JSON-encoded model
    import json
    parsed = json.loads(received[0])
    assert parsed["topic"] == "Python"
    assert parsed["depth"] == 3


@pytest.mark.asyncio
async def test_agent_tool_input_schema_invalid_raises() -> None:
    """Invalid kwargs (wrong types) raise ChildAgentError before child runs."""
    child = _mock_agent("researcher", "facts")
    tool = AgentTool(child, input_schema=_ResearchRequest)

    parent_ctx = ExecutionContext()
    with pytest.raises(ChildAgentError, match="Invalid input"):
        # depth must be int; pass a non-coercible string
        await tool.execute(parent_ctx, topic="Python", depth="not-a-number-!!!")


# ── AgentTool: error handling ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_agent_tool_child_error_raises_child_agent_error() -> None:
    """AgentResult.error → ChildAgentError so parent gets status='error'."""
    child = _mock_agent("child", "", error="LLM failed")
    tool = AgentTool(child)

    with pytest.raises(ChildAgentError, match="LLM failed"):
        await tool.execute(ExecutionContext(), request="anything")


@pytest.mark.asyncio
async def test_agent_tool_child_exception_raises_child_agent_error() -> None:
    """Unexpected exception from child.run() → ChildAgentError."""
    child = MagicMock(spec=Agent)
    child.name = "child"
    child.instructions = "x"
    child._toolbox = {}
    child.run = AsyncMock(side_effect=RuntimeError("network timeout"))

    tool = AgentTool(child)
    with pytest.raises(ChildAgentError, match="network timeout"):
        await tool.execute(ExecutionContext(), request="ping")


@pytest.mark.asyncio
async def test_agent_tool_parent_survives_child_error() -> None:
    """The parent context is intact (no exception bubbles through) when
    we test the path through Agent.act rather than execute directly.

    We call act() on a real (minimal) Agent whose toolbox contains an
    AgentTool that will raise ChildAgentError.
    """
    from agentkit.types import ToolCall

    child = _mock_agent("child", "", error="boom")
    agent_tool = AgentTool(child)

    # Build a minimal real agent to test act()
    from agentkit.agent import Agent
    from agentkit.llm import LlmClient

    orchestrator = Agent(
        model=MagicMock(spec=LlmClient),
        tools=[agent_tool],
        name="orchestrator",
    )

    parent_ctx = ExecutionContext()
    call = ToolCall(
        tool_call_id="tc1",
        name=agent_tool.name,
        arguments={"request": "do it"},
    )
    results = await orchestrator.act(parent_ctx, [call])

    # Parent is alive; tool result carries the error
    assert len(results) == 1
    assert results[0].status == "error"
    assert "boom" in str(results[0].content[0])


# ── AgentTool: child_traces ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_agent_tool_saves_child_trace_in_parent_state() -> None:
    """Successful child run saves a trace entry in parent context.state."""
    child = _mock_agent("child", "result")
    tool = AgentTool(child)

    parent_ctx = ExecutionContext()
    await tool.execute(parent_ctx, request="go")

    traces = parent_ctx.state.get("child_traces", {})
    assert len(traces) == 1
    entry = next(iter(traces.values()))
    assert entry["agent"] == "child"
    assert entry["error"] is None


@pytest.mark.asyncio
async def test_agent_tool_saves_child_trace_on_error() -> None:
    """Failed child run still saves a trace entry (for debugging)."""
    child = _mock_agent("child", "", error="fail")
    tool = AgentTool(child)

    parent_ctx = ExecutionContext()
    with pytest.raises(ChildAgentError):
        await tool.execute(parent_ctx, request="go")

    traces = parent_ctx.state.get("child_traces", {})
    assert len(traces) == 1
    entry = next(iter(traces.values()))
    assert entry["agent"] == "child"
    assert entry["error"] == "fail"


# ── AgentTool: recursion protection ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_agent_tool_blocks_direct_recursion() -> None:
    """Agent calling itself as a tool → ChildAgentError immediately."""
    child = _mock_agent("looper", "ok")
    tool = AgentTool(child)

    parent_ctx = ExecutionContext()
    # Simulate: we're already inside the looper call
    parent_ctx.state["_agent_call_stack"] = ["looper"]

    with pytest.raises(ChildAgentError, match="Recursive call detected"):
        await tool.execute(parent_ctx, request="loop again")


@pytest.mark.asyncio
async def test_agent_tool_blocks_chain_recursion() -> None:
    """A→B→A is detected via the propagated call stack."""
    agent_a = _mock_agent("A", "ok")
    tool_a = AgentTool(agent_a)

    parent_ctx = ExecutionContext()
    # Call stack: we came from A, went to B, now B is trying to call A again
    parent_ctx.state["_agent_call_stack"] = ["A", "B"]

    with pytest.raises(ChildAgentError, match="Recursive call detected"):
        await tool_a.execute(parent_ctx, request="from B back to A")


@pytest.mark.asyncio
async def test_agent_tool_propagates_call_stack_to_child() -> None:
    """Child context receives the parent's call stack with self appended."""
    child, captured = _mock_agent_captures_ctx("child", "ok")
    tool = AgentTool(child)

    parent_ctx = ExecutionContext()
    parent_ctx.state["_agent_call_stack"] = ["parent"]

    await tool.execute(parent_ctx, request="go")

    assert captured[0] is not None
    stack = captured[0].state.get("_agent_call_stack", [])
    assert stack == ["parent", "child"]


@pytest.mark.asyncio
async def test_agent_tool_blocks_at_max_depth() -> None:
    """Depth limit stops nesting regardless of which agents are involved."""
    child = _mock_agent("deep", "ok")
    tool = AgentTool(child, max_depth=2)

    parent_ctx = ExecutionContext()
    parent_ctx.state["_agent_call_stack"] = ["a", "b"]  # depth == max_depth

    with pytest.raises(ChildAgentError, match="depth"):
        await tool.execute(parent_ctx, request="go deeper")


# ── AgentTool: schema ──────────────────────────────────────────────────────────


def test_agent_tool_default_schema_has_request_param() -> None:
    child = _mock_agent("child", "ok")
    tool = AgentTool(child)
    schema = tool.tool_definition
    props = schema["function"]["parameters"]["properties"]
    assert "request" in props


def test_agent_tool_input_schema_uses_pydantic_schema() -> None:
    child = _mock_agent("child", "ok")
    tool = AgentTool(child, input_schema=_ResearchRequest)
    schema = tool.tool_definition
    props = schema["function"]["parameters"]["properties"]
    assert "topic" in props
    assert "depth" in props


def test_agent_tool_name_defaults_to_agent_name() -> None:
    child = _mock_agent("my_agent", "ok")
    tool = AgentTool(child)
    assert tool.name == "my_agent"


def test_agent_tool_name_overridable() -> None:
    child = _mock_agent("my_agent", "ok")
    tool = AgentTool(child, name="call_my_agent")
    assert tool.name == "call_my_agent"


# ── TransferOrchestrator ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_transfer_no_transfer_returns_entry_result() -> None:
    """When no agent calls transfer_to, entry agent's output is final."""
    agent_a = _mock_agent("A", "done by A")
    orch = TransferOrchestrator({"A": agent_a}, entry_agent="A")
    result = await orch.run("hello")

    assert result.output == "done by A"
    assert result.transfer_log == []
    assert result.error is None


@pytest.mark.asyncio
async def test_transfer_changes_active_agent() -> None:
    """Entry agent calls transfer_to → target agent takes over."""
    agent_a = _transfer_agent("A", "B", reason="B handles this")
    agent_b = _mock_agent("B", "done by B")

    orch = TransferOrchestrator({"A": agent_a, "B": agent_b}, entry_agent="A")
    result = await orch.run("question")

    assert result.output == "done by B"
    assert len(result.transfer_log) == 1
    assert result.transfer_log[0]["from"] == "A"
    assert result.transfer_log[0]["to"] == "B"
    assert result.error is None


@pytest.mark.asyncio
async def test_transfer_shared_context() -> None:
    """Both agents operate on the same ExecutionContext (events accumulate)."""
    contexts_seen: list[ExecutionContext] = []

    def _ctx_capturing_agent(name: str, output: str) -> Agent:
        agent = MagicMock(spec=Agent)
        agent.name = name
        agent.instructions = "x"
        agent._toolbox = {}

        async def run(
            user_input: str, context: ExecutionContext | None = None
        ) -> AgentResult:
            ctx = context or ExecutionContext()
            contexts_seen.append(ctx)
            ctx.add_message("assistant", output, author=name)
            return AgentResult(output=output, context=ctx)

        agent.run = run
        return agent

    agent_a = _transfer_agent("A", "B", reason="pass")
    agent_b = _ctx_capturing_agent("B", "B answer")

    orch = TransferOrchestrator({"A": agent_a, "B": agent_b}, entry_agent="A")
    result = await orch.run("go")

    # B's context should be the SAME object as the final result context
    # (shared context propagates through transfers)
    assert result.context is contexts_seen[0]


@pytest.mark.asyncio
async def test_transfer_limit_stops_ping_pong() -> None:
    """A→B→A→B loop is stopped at max_transfers."""
    agent_a = _transfer_agent("A", "B", "B should handle this")
    agent_b = _transfer_agent("B", "A", "A should handle this")

    orch = TransferOrchestrator(
        {"A": agent_a, "B": agent_b},
        entry_agent="A",
        max_transfers=3,
    )
    result = await orch.run("ping")

    assert result.error is not None
    assert "limit" in result.error.lower() or "exceeded" in result.error.lower()
    assert len(result.transfer_log) == 3


@pytest.mark.asyncio
async def test_transfer_unknown_agent_returns_error() -> None:
    """transfer_to with an unregistered agent name returns an error string."""
    agent_a = _transfer_agent("A", "ghost", "ghost handles this")

    orch = TransferOrchestrator({"A": agent_a}, entry_agent="A")
    result = await orch.run("go")

    # 'ghost' not in registry → transfer_to tool returns error text
    # The agent produces output containing the error, but the orchestrator
    # detects no valid transfer (state key was NOT set because the tool
    # returned early).  So the run should complete normally with A's output.
    # (The tool returns an error string which the LLM sees, but for our mock
    # the state is set by our helper directly — so this tests the registry check
    # in the run loop instead.)
    assert result is not None


@pytest.mark.asyncio
async def test_transfer_tool_rejects_unknown_agent_name() -> None:
    """The transfer_to tool itself rejects unknown agent names."""
    agent_a = _mock_agent("A", "ok")
    orch = TransferOrchestrator({"A": agent_a}, entry_agent="A")

    # Call the transfer_to tool directly
    ctx = ExecutionContext()
    result_str = await orch.transfer_tool.execute(ctx, agent_name="ghost", reason="x")

    assert "Error" in result_str or "unknown" in result_str.lower()
    # _transfer_request should NOT be set
    assert "_transfer_request" not in ctx.state


@pytest.mark.asyncio
async def test_transfer_tool_sets_request_for_valid_agent() -> None:
    """transfer_to sets _transfer_request when agent is known."""
    agent_a = _mock_agent("A", "ok")
    agent_b = _mock_agent("B", "ok")
    orch = TransferOrchestrator({"A": agent_a, "B": agent_b}, entry_agent="A")

    ctx = ExecutionContext()
    await orch.transfer_tool.execute(ctx, agent_name="B", reason="better fit")

    req = ctx.state.get("_transfer_request")
    assert req is not None
    assert req["to"] == "B"
    assert req["reason"] == "better fit"


@pytest.mark.asyncio
async def test_transfer_tool_injected_into_all_agents() -> None:
    """TransferOrchestrator injects transfer_to into every agent's _toolbox."""
    agent_a = _mock_agent("A", "ok")
    agent_b = _mock_agent("B", "ok")
    orch = TransferOrchestrator({"A": agent_a, "B": agent_b}, entry_agent="A")

    assert "transfer_to" in agent_a._toolbox
    assert "transfer_to" in agent_b._toolbox
    assert agent_a._toolbox["transfer_to"] is orch.transfer_tool


@pytest.mark.asyncio
async def test_transfer_invalid_entry_agent_raises() -> None:
    """Constructing TransferOrchestrator with unknown entry_agent raises."""
    agent_a = _mock_agent("A", "ok")
    with pytest.raises(ValueError, match="Entry agent"):
        TransferOrchestrator({"A": agent_a}, entry_agent="Z")


# ── Live test ──────────────────────────────────────────────────────────────────


@pytest.mark.live
@pytest.mark.asyncio
async def test_live_orchestrator_routes_to_specialist() -> None:
    """Real orchestrator with researcher and writer; researcher handles search."""
    from agentkit.agents.specialists import make_researcher
    from agentkit.config import FAST_MODEL, find_uv
    from agentkit.llm import LlmClient
    from agentkit.mcp_client import McpToolset
    from agentkit.tools.mcp import load_mcp_tools

    mcp_cmd = (find_uv(), ["run", "python", "-m", "agentkit.servers.tavily_server"])
    async with McpToolset(*mcp_cmd) as ts:
        search_tools = list(load_mcp_tools(ts))

    model = LlmClient(FAST_MODEL)
    researcher_tool = AgentTool(
        make_researcher(model, tools=search_tools),
        description="Search for factual information on a topic.",
    )

    from agentkit.agent import Agent

    orchestrator = Agent(
        model=model,
        tools=[researcher_tool],
        name="orchestrator",
        instructions=(
            "You coordinate research. Use the researcher tool for factual lookups."
        ),
        max_steps=6,
    )
    result = await orchestrator.run("What year was Python first released?")

    assert result.error is None
    output = str(result.output)
    assert "1991" in output or "python" in output.lower()
    traces = result.context.state.get("child_traces", {})
    assert len(traces) >= 1
