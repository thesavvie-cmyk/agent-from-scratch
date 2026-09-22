"""Unit tests for block 20: multi-agent workflow primitives.

All agents are mocked — no API key required.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from agentkit.agent import Agent, AgentResult
from agentkit.context import ExecutionContext
from agentkit.workflow import (
    LoopWorkflow,
    ParallelWorkflow,
    SequentialWorkflow,
    WorkflowResult,
    WorkflowStep,
)

# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_agent(name: str, output: str, error: str | None = None) -> Agent:
    """Return a mock Agent that always returns *output*."""
    agent = MagicMock(spec=Agent)
    agent.name = name
    ctx = ExecutionContext()
    ctx.add_message("assistant", output, author=name)
    ctx.state["token_usage"] = {"input_tokens": 10, "output_tokens": 5}
    result = AgentResult(output=output, context=ctx, error=error)
    agent.run = AsyncMock(return_value=result)
    return agent


def _make_raising_agent(name: str, exc: Exception) -> Agent:
    """Return a mock Agent whose run() raises *exc*."""
    agent = MagicMock(spec=Agent)
    agent.name = name
    agent.run = AsyncMock(side_effect=exc)
    return agent


# ── WorkflowResult ─────────────────────────────────────────────────────────────


def test_workflow_result_all_events_deduplicates() -> None:
    """Events shared across step contexts appear only once in all_events."""
    ctx = ExecutionContext()
    ctx.add_message("assistant", "hello", author="a")
    r1 = AgentResult(output="hello", context=ctx)

    # Simulate share_context: r2 holds the same context object
    r2 = AgentResult(output="world", context=ctx)

    wr = WorkflowResult(output="world", step_results=[r1, r2])
    events = wr.all_events
    # The single event must not be duplicated
    assert len(events) == 1


def test_workflow_result_total_tokens() -> None:
    def _ctx(inp: int, out: int) -> ExecutionContext:
        c = ExecutionContext()
        c.state["token_usage"] = {"input_tokens": inp, "output_tokens": out}
        return c

    r1 = AgentResult(output="a", context=_ctx(100, 50))
    r2 = AgentResult(output="b", context=_ctx(200, 80))
    wr = WorkflowResult(output="b", step_results=[r1, r2])
    assert wr.total_tokens() == {"input_tokens": 300, "output_tokens": 130}


def test_workflow_result_tokens_per_step() -> None:
    def _ctx(inp: int, out: int) -> ExecutionContext:
        c = ExecutionContext()
        c.state["token_usage"] = {"input_tokens": inp, "output_tokens": out}
        return c

    r1 = AgentResult(output="a", context=_ctx(10, 5))
    r2 = AgentResult(output="b", context=_ctx(20, 8))
    wr = WorkflowResult(output="b", step_results=[r1, r2])
    tps = wr.tokens_per_step()
    assert tps[0] == {"input_tokens": 10, "output_tokens": 5}
    assert tps[1] == {"input_tokens": 20, "output_tokens": 8}


# ── SequentialWorkflow ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_sequential_runs_in_order() -> None:
    """Steps run in order; each step receives the previous step's output."""
    calls: list[str] = []

    ctx_a = ExecutionContext()
    ctx_a.add_message("assistant", "result-a", author="a")
    agent_a = MagicMock(spec=Agent)
    agent_a.name = "a"

    ctx_b = ExecutionContext()
    ctx_b.add_message("assistant", "result-b", author="b")
    agent_b = MagicMock(spec=Agent)
    agent_b.name = "b"

    async def run_a(user_input: str, context: ExecutionContext | None = None) -> AgentResult:
        calls.append(f"a:{user_input}")
        return AgentResult(output="result-a", context=ctx_a)

    async def run_b(user_input: str, context: ExecutionContext | None = None) -> AgentResult:
        calls.append(f"b:{user_input}")
        return AgentResult(output="result-b", context=ctx_b)

    agent_a.run = run_a
    agent_b.run = run_b

    wf = SequentialWorkflow([WorkflowStep(agent_a), WorkflowStep(agent_b)])
    result = await wf.run("start")

    assert calls[0] == "a:start"
    assert calls[1] == "b:result-a"
    assert result.output == "result-b"
    assert len(result.step_results) == 2


@pytest.mark.asyncio
async def test_sequential_output_propagation() -> None:
    """The output of step N becomes the input of step N+1."""
    received_inputs: list[str] = []

    agents = []
    outputs = ["step1-out", "step2-out", "step3-out"]
    for i, out in enumerate(outputs):
        a = MagicMock(spec=Agent)
        a.name = f"agent{i}"
        ctx = ExecutionContext()
        ctx.add_message("assistant", out, author=a.name)

        async def _run(
            user_input: str,
            context: ExecutionContext | None = None,
            _out: str = out,
            _ctx: ExecutionContext = ctx,
        ) -> AgentResult:
            received_inputs.append(user_input)
            return AgentResult(output=_out, context=_ctx)

        a.run = _run
        agents.append(a)

    wf = SequentialWorkflow([WorkflowStep(a) for a in agents])
    await wf.run("original")

    assert received_inputs[0] == "original"
    assert received_inputs[1] == "step1-out"
    assert received_inputs[2] == "step2-out"


@pytest.mark.asyncio
async def test_sequential_stops_on_error() -> None:
    """When a step has an error, subsequent steps are not executed."""
    agent_ok = _make_agent("ok", "fine")
    agent_fail = _make_agent("fail", "", error="boom")
    agent_never = _make_agent("never", "should not run")

    wf = SequentialWorkflow([
        WorkflowStep(agent_ok),
        WorkflowStep(agent_fail),
        WorkflowStep(agent_never),
    ])
    result = await wf.run("go")

    assert result.error == "boom"
    assert len(result.step_results) == 2
    agent_never.run.assert_not_called()


@pytest.mark.asyncio
async def test_sequential_all_events_from_all_steps() -> None:
    """all_events aggregates events from every step's context."""
    agent_a = _make_agent("a", "out-a")
    agent_b = _make_agent("b", "out-b")
    wf = SequentialWorkflow([WorkflowStep(agent_a), WorkflowStep(agent_b)])
    result = await wf.run("hi")

    events = result.all_events
    authors = {e.author for e in events}
    # Both agents contributed events
    assert "a" in authors
    assert "b" in authors


@pytest.mark.asyncio
async def test_sequential_requires_nonempty_steps() -> None:
    with pytest.raises(ValueError, match="at least one"):
        SequentialWorkflow([])


# ── share_context ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_share_context_false_passes_none_context() -> None:
    """share_context=False → agent.run receives context=None."""
    received_ctx: list[ExecutionContext | None] = []

    agent = MagicMock(spec=Agent)
    agent.name = "a"
    ctx = ExecutionContext()
    ctx.add_message("assistant", "out", author="a")

    async def run(
        user_input: str, context: ExecutionContext | None = None
    ) -> AgentResult:
        received_ctx.append(context)
        return AgentResult(output="out", context=ctx)

    agent.run = run
    step = WorkflowStep(agent, share_context=False)
    wf = SequentialWorkflow([step])
    await wf.run("hi")

    assert received_ctx[0] is None


@pytest.mark.asyncio
async def test_share_context_true_passes_previous_context() -> None:
    """share_context=True → second agent receives first agent's context."""
    ctx_a = ExecutionContext()
    ctx_a.add_message("assistant", "out-a", author="a")

    agent_a = MagicMock(spec=Agent)
    agent_a.name = "a"
    agent_a.run = AsyncMock(return_value=AgentResult(output="out-a", context=ctx_a))

    received_ctx: list[ExecutionContext | None] = []

    ctx_b = ExecutionContext()
    ctx_b.add_message("assistant", "out-b", author="b")
    agent_b = MagicMock(spec=Agent)
    agent_b.name = "b"

    async def run_b(
        user_input: str, context: ExecutionContext | None = None
    ) -> AgentResult:
        received_ctx.append(context)
        return AgentResult(output="out-b", context=ctx_b)

    agent_b.run = run_b

    wf = SequentialWorkflow([
        WorkflowStep(agent_a, share_context=True),
        WorkflowStep(agent_b, share_context=True),
    ])
    await wf.run("start")

    # Second step should receive the first step's context
    assert received_ctx[0] is ctx_a


# ── author in events ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_event_author_matches_agent_name() -> None:
    """Events produced by each agent carry that agent's name as author."""
    agent_a = _make_agent("researcher", "facts about X")
    agent_b = _make_agent("writer", "article about X")

    wf = SequentialWorkflow([WorkflowStep(agent_a), WorkflowStep(agent_b)])
    result = await wf.run("topic")

    authors = [e.author for e in result.all_events]
    assert "researcher" in authors
    assert "writer" in authors


# ── ParallelWorkflow ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_parallel_all_steps_receive_same_input() -> None:
    """All parallel steps receive the original user_input."""
    received: list[str] = []

    def _agent(name: str) -> Agent:
        a = MagicMock(spec=Agent)
        a.name = name
        ctx = ExecutionContext()
        ctx.add_message("assistant", f"out-{name}", author=name)

        async def run(
            user_input: str, context: ExecutionContext | None = None
        ) -> AgentResult:
            received.append(user_input)
            return AgentResult(output=f"out-{name}", context=ctx)

        a.run = run
        return a

    wf = ParallelWorkflow([WorkflowStep(_agent("x")), WorkflowStep(_agent("y"))])
    await wf.run("query")

    assert received == ["query", "query"]


@pytest.mark.asyncio
async def test_parallel_exception_does_not_abort_others() -> None:
    """An exception in one parallel step does not prevent others from running."""
    agent_good = _make_agent("good", "success")
    agent_bad = _make_raising_agent("bad", RuntimeError("exploded"))

    wf = ParallelWorkflow([WorkflowStep(agent_good), WorkflowStep(agent_bad)])
    result = await wf.run("hi")

    assert result.error is not None
    assert "exploded" in result.error
    # The good agent's result is still present
    assert len(result.step_results) == 2
    assert result.step_results[0].output == "success"


@pytest.mark.asyncio
async def test_parallel_merge_function_called_with_all_outputs() -> None:
    """The merge function receives one string per step."""
    merge_inputs: list[list[str]] = []

    def merge(outputs: list[str]) -> str:
        merge_inputs.append(list(outputs))
        return "merged"

    agent_a = _make_agent("a", "alpha")
    agent_b = _make_agent("b", "beta")

    wf = ParallelWorkflow([WorkflowStep(agent_a), WorkflowStep(agent_b)], merge=merge)
    result = await wf.run("go")

    assert merge_inputs == [["alpha", "beta"]]
    assert result.output == "merged"


@pytest.mark.asyncio
async def test_parallel_requires_nonempty_steps() -> None:
    with pytest.raises(ValueError, match="at least one"):
        ParallelWorkflow([])


# ── LoopWorkflow ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_loop_stops_when_condition_false() -> None:
    """Loop stops immediately when condition returns False after first run."""
    agent = _make_agent("a", "done")
    wf = LoopWorkflow(
        WorkflowStep(agent),
        condition=lambda _r: False,  # stop after first iteration
        max_iterations=5,
    )
    result = await wf.run("go")

    assert agent.run.call_count == 1
    assert result.output == "done"
    assert len(result.step_results) == 1


@pytest.mark.asyncio
async def test_loop_stops_at_max_iterations() -> None:
    """Loop stops at max_iterations even when condition always returns True."""
    agent = _make_agent("a", "keep going")
    wf = LoopWorkflow(
        WorkflowStep(agent),
        condition=lambda _r: True,  # always want to continue
        max_iterations=3,
    )
    result = await wf.run("go")

    assert agent.run.call_count == 3
    assert len(result.step_results) == 3


@pytest.mark.asyncio
async def test_loop_passes_previous_output_as_next_input() -> None:
    """Each iteration receives the previous iteration's output as input."""
    received: list[str] = []
    iteration = 0

    agent = MagicMock(spec=Agent)
    agent.name = "loop_agent"

    async def run(
        user_input: str, context: ExecutionContext | None = None
    ) -> AgentResult:
        nonlocal iteration
        received.append(user_input)
        out = f"iter-{iteration}"
        iteration += 1
        ctx = ExecutionContext()
        ctx.add_message("assistant", out, author="loop_agent")
        return AgentResult(output=out, context=ctx)

    agent.run = run

    # condition: stop after 3 iterations
    count = 0

    def cond(_r: AgentResult) -> bool:
        nonlocal count
        count += 1
        return count < 3

    wf = LoopWorkflow(WorkflowStep(agent), condition=cond, max_iterations=5)
    await wf.run("start")

    assert received[0] == "start"
    assert received[1] == "iter-0"
    assert received[2] == "iter-1"


@pytest.mark.asyncio
async def test_loop_stops_on_error() -> None:
    """Loop stops immediately when a step returns an error."""
    agent = _make_agent("a", "", error="oops")
    wf = LoopWorkflow(
        WorkflowStep(agent),
        condition=lambda _r: True,
        max_iterations=5,
    )
    result = await wf.run("go")

    assert agent.run.call_count == 1
    assert result.error == "oops"


@pytest.mark.asyncio
async def test_loop_requires_positive_max_iterations() -> None:
    agent = _make_agent("a", "x")
    with pytest.raises(ValueError, match="max_iterations"):
        LoopWorkflow(WorkflowStep(agent), condition=lambda _r: False, max_iterations=0)


# ── Specialists ────────────────────────────────────────────────────────────────


def test_specialists_have_correct_names() -> None:
    """Each specialist factory sets the expected agent name."""
    from agentkit.agents.specialists import (
        make_coder,
        make_researcher,
        make_reviewer,
        make_writer,
    )
    from agentkit.llm import LlmClient

    model = MagicMock(spec=LlmClient)
    assert make_researcher(model).name == "researcher"
    assert make_coder(model).name == "coder"
    assert make_writer(model).name == "writer"
    assert make_reviewer(model).name == "reviewer"


def test_specialists_instructions_not_empty() -> None:
    from agentkit.agents.specialists import (
        make_coder,
        make_researcher,
        make_reviewer,
        make_writer,
    )
    from agentkit.llm import LlmClient

    model = MagicMock(spec=LlmClient)
    for factory in (make_researcher, make_coder, make_writer, make_reviewer):
        agent = factory(model)
        assert len(agent.instructions) > 20


# ── Live test ──────────────────────────────────────────────────────────────────


@pytest.mark.live
@pytest.mark.asyncio
async def test_live_blog_pipeline() -> None:
    """Researcher → writer pipeline produces coherent text."""
    from agentkit.agents.specialists import make_researcher, make_writer
    from agentkit.config import FAST_MODEL, find_uv
    from agentkit.llm import LlmClient
    from agentkit.mcp_client import McpToolset
    from agentkit.tools.mcp import load_mcp_tools

    mcp_cmd = (find_uv(), ["run", "python", "-m", "agentkit.servers.tavily_server"])
    async with McpToolset(*mcp_cmd) as ts:
        search_tools = list(load_mcp_tools(ts))

    model = LlmClient(FAST_MODEL)
    wf = SequentialWorkflow([
        WorkflowStep(make_researcher(model, tools=search_tools)),
        WorkflowStep(make_writer(model)),
    ])
    result = await wf.run("The impact of large language models on software development")

    assert result.error is None
    output = str(result.output)
    assert len(output) > 200
    assert len(result.step_results) == 2
    # Check events carry correct authors
    authors = {e.author for e in result.all_events}
    assert "researcher" in authors
    assert "writer" in authors
