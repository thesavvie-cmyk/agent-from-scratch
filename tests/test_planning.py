"""Unit tests for block 15: planning."""
from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from agentkit.context import ExecutionContext
from agentkit.planning import (
    Plan,
    Task,
    TaskStatus,
    create_plan,
    get_plan,
    update_task,
)
from agentkit.types import Message, ToolCall

# ── Plan data model ────────────────────────────────────────────────────────────


def test_plan_next_pending_returns_first_pending() -> None:
    plan = Plan(tasks=[
        Task(id="t1", description="step 1", status=TaskStatus.done),
        Task(id="t2", description="step 2"),
        Task(id="t3", description="step 3"),
    ])
    t = plan.next_pending()
    assert t is not None
    assert t.id == "t2"


def test_plan_next_pending_none_when_all_done() -> None:
    plan = Plan(tasks=[
        Task(id="t1", description="step 1", status=TaskStatus.done),
        Task(id="t2", description="step 2", status=TaskStatus.failed),
    ])
    assert plan.next_pending() is None


def test_plan_mark_updates_status_and_result() -> None:
    plan = Plan(tasks=[Task(id="t1", description="step 1")])
    plan.mark("t1", TaskStatus.done, result="answer found")
    assert plan.tasks[0].status == TaskStatus.done
    assert plan.tasks[0].result == "answer found"


def test_plan_mark_unknown_task_raises_value_error() -> None:
    plan = Plan(tasks=[Task(id="t1", description="step 1")])
    with pytest.raises(ValueError, match="t99"):
        plan.mark("t99", TaskStatus.done)


def test_plan_mark_error_message_lists_available_ids() -> None:
    plan = Plan(tasks=[
        Task(id="t1", description="a"),
        Task(id="t2", description="b"),
    ])
    with pytest.raises(ValueError) as exc_info:
        plan.mark("t99", TaskStatus.done)
    assert "t1" in str(exc_info.value)
    assert "t2" in str(exc_info.value)


def test_plan_is_complete_false_when_pending() -> None:
    plan = Plan(tasks=[
        Task(id="t1", description="a", status=TaskStatus.done),
        Task(id="t2", description="b"),
    ])
    assert not plan.is_complete()


def test_plan_is_complete_true_when_all_done_or_failed() -> None:
    plan = Plan(tasks=[
        Task(id="t1", description="a", status=TaskStatus.done),
        Task(id="t2", description="b", status=TaskStatus.failed),
    ])
    assert plan.is_complete()


def test_plan_is_complete_false_when_in_progress() -> None:
    plan = Plan(tasks=[
        Task(id="t1", description="a", status=TaskStatus.in_progress),
    ])
    assert not plan.is_complete()


def test_plan_is_complete_false_empty() -> None:
    assert not Plan(tasks=[]).is_complete()


def test_plan_progress_symbols() -> None:
    plan = Plan(tasks=[
        Task(id="t1", description="find data", status=TaskStatus.done),
        Task(id="t2", description="compute", status=TaskStatus.in_progress),
        Task(id="t3", description="report"),
        Task(id="t4", description="cleanup", status=TaskStatus.failed),
    ])
    p = plan.progress()
    assert "[x] find data" in p
    assert "[>] compute" in p
    assert "[ ] report" in p
    assert "[!] cleanup" in p


def test_plan_progress_separator() -> None:
    plan = Plan(tasks=[
        Task(id="t1", description="a"),
        Task(id="t2", description="b"),
    ])
    parts = plan.progress().split(" | ")
    assert len(parts) == 2


# ── Planning tools ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_create_plan_stores_in_context() -> None:
    ctx = ExecutionContext()
    result = await create_plan(context=ctx, tasks=["step one", "step two"])
    assert "plan" in ctx.state
    plan = Plan.model_validate(ctx.state["plan"])
    assert len(plan.tasks) == 2
    assert plan.tasks[0].id == "t1"
    assert plan.tasks[1].id == "t2"
    assert "step one" in result


@pytest.mark.asyncio
async def test_create_plan_replaces_existing_and_notes_it() -> None:
    ctx = ExecutionContext()
    await create_plan(context=ctx, tasks=["old step"])
    result = await create_plan(context=ctx, tasks=["new step"])
    assert "replaced" in result.lower()
    plan = Plan.model_validate(ctx.state["plan"])
    assert plan.tasks[0].description == "new step"


@pytest.mark.asyncio
async def test_update_task_changes_status() -> None:
    ctx = ExecutionContext()
    await create_plan(context=ctx, tasks=["step one"])
    result = await update_task(context=ctx, task_id="t1", status="done", result="finished")
    assert "done" in result
    plan = Plan.model_validate(ctx.state["plan"])
    assert plan.tasks[0].status == TaskStatus.done
    assert plan.tasks[0].result == "finished"


@pytest.mark.asyncio
async def test_update_task_unknown_id_returns_error_string() -> None:
    ctx = ExecutionContext()
    await create_plan(context=ctx, tasks=["step one"])
    result = await update_task(context=ctx, task_id="t99", status="done")
    assert "t99" in result
    assert "not found" in result.lower()


@pytest.mark.asyncio
async def test_update_task_no_plan_returns_error_string() -> None:
    ctx = ExecutionContext()
    result = await update_task(context=ctx, task_id="t1", status="done")
    assert "no plan" in result.lower()


@pytest.mark.asyncio
async def test_get_plan_returns_all_tasks() -> None:
    ctx = ExecutionContext()
    await create_plan(context=ctx, tasks=["search", "compute", "report"])
    result = await get_plan(context=ctx)
    assert "search" in result
    assert "compute" in result
    assert "report" in result


@pytest.mark.asyncio
async def test_get_plan_no_plan_returns_helpful_message() -> None:
    ctx = ExecutionContext()
    result = await get_plan(context=ctx)
    assert "no plan" in result.lower()


@pytest.mark.asyncio
async def test_get_plan_shows_complete_when_done() -> None:
    ctx = ExecutionContext()
    await create_plan(context=ctx, tasks=["only step"])
    await update_task(context=ctx, task_id="t1", status="done")
    result = await get_plan(context=ctx)
    assert "Complete" in result


# ── create_plan tool schema has list[str] ──────────────────────────────────────


def test_create_plan_schema_has_array_type() -> None:
    schema = create_plan.tool_definition
    tasks_schema = schema["function"]["parameters"]["properties"]["tasks"]
    assert tasks_schema["type"] == "array"
    assert tasks_schema["items"]["type"] == "string"


# ── Agent integration ──────────────────────────────────────────────────────────


def test_agent_planning_adds_tools_to_toolbox() -> None:
    from agentkit.agent import Agent
    from agentkit.llm import LlmClient

    agent = Agent(model=MagicMock(spec=LlmClient), planning=True)
    assert "create_plan" in agent._toolbox
    assert "update_task" in agent._toolbox
    assert "get_plan" in agent._toolbox


def test_agent_no_planning_no_plan_tools() -> None:
    from agentkit.agent import Agent
    from agentkit.llm import LlmClient

    agent = Agent(model=MagicMock(spec=LlmClient), planning=False)
    assert "create_plan" not in agent._toolbox
    assert "update_task" not in agent._toolbox
    assert "get_plan" not in agent._toolbox


def test_agent_planning_instructions_in_effective() -> None:
    from agentkit.agent import Agent
    from agentkit.llm import LlmClient
    from agentkit.planning import PLANNING_INSTRUCTIONS

    agent = Agent(
        model=MagicMock(spec=LlmClient),
        instructions="Be concise.",
        planning=True,
    )
    assert "Be concise." in agent._effective_instructions
    assert PLANNING_INSTRUCTIONS in agent._effective_instructions


@pytest.mark.asyncio
async def test_agent_plan_injected_in_instructions_on_step2() -> None:
    """After create_plan is called, subsequent steps include plan progress."""
    from agentkit.agent import Agent
    from agentkit.llm import LlmClient, LlmResponse

    captured: list[Any] = []
    call_count = 0

    async def _mock_generate(req: Any) -> LlmResponse:
        nonlocal call_count
        captured.append(req)
        call_count += 1
        if call_count == 1:
            # Agent creates a plan
            return LlmResponse(
                content=[ToolCall(
                    tool_call_id="c1",
                    name="create_plan",
                    arguments={"tasks": ["search for data", "compute result"]},
                )],
                usage_metadata={},
            )
        # Subsequent steps: final text answer
        return LlmResponse(
            content=[Message(role="assistant", content="Done!")],
            usage_metadata={},
        )

    mock_client = MagicMock(spec=LlmClient)
    mock_client.generate = AsyncMock(side_effect=_mock_generate)

    agent = Agent(model=mock_client, planning=True, max_steps=5)
    await agent.run("Do something multi-step")

    assert len(captured) >= 2, "Expected at least 2 LLM calls"
    second_instructions = " ".join(captured[1].instructions)
    # Plan progress should appear (brackets from status symbols)
    assert "[ ]" in second_instructions or "[x]" in second_instructions or "[>]" in second_instructions


@pytest.mark.asyncio
async def test_agent_think_first_produces_message_before_tool_calls() -> None:
    """With think_first=True, there is a Message-only event before any ToolCall."""
    from agentkit.agent import Agent
    from agentkit.llm import LlmClient, LlmResponse

    call_count = 0

    async def _mock_generate(req: Any) -> LlmResponse:
        nonlocal call_count
        call_count += 1
        # Think call has no tools in request → return plain text
        # Action call (tools available) → return final answer
        return LlmResponse(
            content=[Message(role="assistant", content=f"Response {call_count}")],
            usage_metadata={},
        )

    mock_client = MagicMock(spec=LlmClient)
    mock_client.generate = AsyncMock(side_effect=_mock_generate)

    agent = Agent(model=mock_client, think_first=True, max_steps=4)
    result = await agent.run("Analyse this task")

    ctx = result.context
    # Find Message-only events after the user message (index 0)
    think_events = [
        e for e in ctx.events[1:]
        if (
            any(isinstance(i, Message) for i in e.content)
            and not any(isinstance(i, ToolCall) for i in e.content)
        )
    ]
    assert len(think_events) >= 1, "Expected at least one think (Message-only) event"


# ── Live tests ─────────────────────────────────────────────────────────────────


@pytest.mark.live
@pytest.mark.asyncio
async def test_live_agent_planning_solves_multistep() -> None:
    """Agent with planning solves a multi-step calculation task."""
    from agentkit.agent import Agent
    from agentkit.config import FAST_MODEL
    from agentkit.llm import LlmClient
    from agentkit.tools.base import FunctionTool
    from agentkit.tools_manual import calculator

    agent = Agent(
        model=LlmClient(FAST_MODEL),
        tools=[FunctionTool(calculator)],
        instructions="Use the calculator for all arithmetic.",
        max_steps=12,
        planning=True,
    )
    result = await agent.run(
        "First calculate 123 * 456. Then add 789 to that result. "
        "Finally multiply by 2. Show each step."
    )
    assert result.error is None
    plan_data = result.context.state.get("plan")
    assert plan_data is not None, "Agent should have created a plan"
    from agentkit.planning import Plan
    plan = Plan.model_validate(plan_data)
    assert plan.is_complete(), f"Plan not complete: {plan.progress()}"
