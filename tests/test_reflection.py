"""Unit and live tests for block 16: reflection tool and Agent integration."""
from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from agentkit.context import ExecutionContext
from agentkit.reflection import _REPLAN_INSTRUCTION, REFLECTION_INSTRUCTIONS, reflection
from agentkit.types import Message, ToolCall

# ── Helpers ────────────────────────────────────────────────────────────────────


def _mock_model(*responses: Any) -> MagicMock:
    from agentkit.llm import LlmClient
    m = MagicMock(spec=LlmClient)
    m.generate = AsyncMock(side_effect=list(responses))
    return m


def _text_resp(text: str) -> Any:
    from agentkit.llm import LlmResponse
    return LlmResponse(content=[Message(role="assistant", content=text)])


def _tool_call_resp(call_id: str, name: str, args: dict) -> Any:
    from agentkit.llm import LlmResponse
    return LlmResponse(content=[ToolCall(tool_call_id=call_id, name=name, arguments=args)])


# ── reflection tool: state writes ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_reflection_writes_need_replan_true() -> None:
    ctx = ExecutionContext()
    result = await reflection(context=ctx, analysis="Things are not working.", need_replan=True)
    assert ctx.state["need_replan"] is True
    assert "need_replan=True" in result


@pytest.mark.asyncio
async def test_reflection_writes_need_replan_false() -> None:
    ctx = ExecutionContext()
    result = await reflection(context=ctx, analysis="Progress is fine.", need_replan=False)
    assert ctx.state["need_replan"] is False
    assert "need_replan=True" not in result


@pytest.mark.asyncio
async def test_reflection_includes_analysis_in_return() -> None:
    ctx = ExecutionContext()
    analysis = "I have found X and still need Y."
    result = await reflection(context=ctx, analysis=analysis)
    assert analysis in result


# ── Agent: reflection=True adds tool to toolbox ────────────────────────────────


def test_agent_reflection_true_adds_tool_to_toolbox() -> None:
    from agentkit.agent import Agent
    from agentkit.llm import LlmClient

    agent = Agent(model=MagicMock(spec=LlmClient), reflection=True)
    assert "reflection" in agent._toolbox


def test_agent_reflection_false_no_tool_in_toolbox() -> None:
    from agentkit.agent import Agent
    from agentkit.llm import LlmClient

    agent = Agent(model=MagicMock(spec=LlmClient), reflection=False)
    assert "reflection" not in agent._toolbox


def test_agent_reflection_true_adds_instructions() -> None:
    from agentkit.agent import Agent
    from agentkit.llm import LlmClient

    agent = Agent(
        model=MagicMock(spec=LlmClient),
        instructions="Be concise.",
        reflection=True,
    )
    assert "Be concise." in agent._effective_instructions
    assert REFLECTION_INSTRUCTIONS in agent._effective_instructions


def test_agent_reflection_false_no_reflection_instructions() -> None:
    from agentkit.agent import Agent
    from agentkit.llm import LlmClient

    agent = Agent(
        model=MagicMock(spec=LlmClient),
        instructions="Be concise.",
        reflection=False,
    )
    assert REFLECTION_INSTRUCTIONS not in agent._effective_instructions


# ── Agent: need_replan flag consumed and injected on next step ─────────────────


@pytest.mark.asyncio
async def test_agent_injects_replan_instruction_when_flag_set() -> None:
    """After reflection sets need_replan=True, the next LLM step gets the
    replan instruction; the flag is then consumed (not injected again)."""
    from agentkit.agent import Agent
    from agentkit.llm import LlmClient, LlmResponse

    captured_instructions: list[list[str]] = []
    call_count = 0

    async def _mock_generate(req: Any) -> LlmResponse:
        nonlocal call_count
        call_count += 1
        captured_instructions.append(list(req.instructions))
        if call_count == 1:
            # Step 1: call reflection with need_replan=True
            return LlmResponse(
                content=[ToolCall(
                    tool_call_id="r1",
                    name="reflection",
                    arguments={"analysis": "Stuck.", "need_replan": True},
                )],
                usage_metadata={},
            )
        # Step 2 and beyond: return final answer
        return LlmResponse(
            content=[Message(role="assistant", content="Final answer.")],
            usage_metadata={},
        )

    mock_client = MagicMock(spec=LlmClient)
    mock_client.generate = AsyncMock(side_effect=_mock_generate)

    agent = Agent(model=mock_client, reflection=True, max_steps=5)
    await agent.run("Do something.")

    assert len(captured_instructions) >= 2
    # Step 2 instructions must contain the replan instruction
    step2_text = " ".join(captured_instructions[1])
    assert _REPLAN_INSTRUCTION in step2_text

    # Step 3 (if any) must NOT contain replan instruction again (flag consumed)
    if len(captured_instructions) >= 3:
        step3_text = " ".join(captured_instructions[2])
        assert _REPLAN_INSTRUCTION not in step3_text


@pytest.mark.asyncio
async def test_agent_no_replan_injection_when_flag_false() -> None:
    """When reflection sets need_replan=False, replan instruction is NOT injected."""
    from agentkit.agent import Agent
    from agentkit.llm import LlmClient, LlmResponse

    captured_instructions: list[list[str]] = []
    call_count = 0

    async def _mock_generate(req: Any) -> LlmResponse:
        nonlocal call_count
        call_count += 1
        captured_instructions.append(list(req.instructions))
        if call_count == 1:
            return LlmResponse(
                content=[ToolCall(
                    tool_call_id="r1",
                    name="reflection",
                    arguments={"analysis": "All good.", "need_replan": False},
                )],
                usage_metadata={},
            )
        return LlmResponse(
            content=[Message(role="assistant", content="Done.")],
            usage_metadata={},
        )

    mock_client = MagicMock(spec=LlmClient)
    mock_client.generate = AsyncMock(side_effect=_mock_generate)

    agent = Agent(model=mock_client, reflection=True, max_steps=5)
    await agent.run("Simple task.")

    assert len(captured_instructions) >= 2
    step2_text = " ".join(captured_instructions[1])
    assert _REPLAN_INSTRUCTION not in step2_text


# ── Agent: event ordering with reflection then final_answer ───────────────────


@pytest.mark.asyncio
async def test_reflection_event_order_in_trace() -> None:
    """Events in trace: user → think(reflection call) → tool_result(reflection)
    → think(final answer). Reflection comes before the final answer."""
    from agentkit.agent import Agent
    from agentkit.llm import LlmClient, LlmResponse

    call_count = 0

    async def _mock_generate(req: Any) -> LlmResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return LlmResponse(
                content=[ToolCall(
                    tool_call_id="r1",
                    name="reflection",
                    arguments={"analysis": "Checking progress.", "need_replan": False},
                )],
                usage_metadata={},
            )
        return LlmResponse(
            content=[Message(role="assistant", content="The answer is 42.")],
            usage_metadata={},
        )

    mock_client = MagicMock(spec=LlmClient)
    mock_client.generate = AsyncMock(side_effect=_mock_generate)

    agent = Agent(model=mock_client, reflection=True, max_steps=5)
    result = await agent.run("Answer the question.")

    events = result.context.events
    # Find reflection call and check it precedes the final answer
    reflection_call_idx = None
    final_answer_idx = None
    for i, evt in enumerate(events):
        for item in evt.content:
            if isinstance(item, ToolCall) and item.name == "reflection":
                reflection_call_idx = i
            if isinstance(item, Message) and item.role == "assistant" and "42" in item.content:
                final_answer_idx = i

    assert reflection_call_idx is not None, "Expected a reflection ToolCall in trace"
    assert final_answer_idx is not None, "Expected a final answer Message in trace"
    assert reflection_call_idx < final_answer_idx, "Reflection must precede final answer"


# ── Agent: reflection=False state is not touched ──────────────────────────────


@pytest.mark.asyncio
async def test_reflection_disabled_state_not_modified() -> None:
    """With reflection=False, context.state should not get a need_replan key
    from the agent machinery (no tool, no injection)."""
    from agentkit.agent import Agent
    from agentkit.llm import LlmClient, LlmResponse

    mock_client = MagicMock(spec=LlmClient)
    mock_client.generate = AsyncMock(return_value=LlmResponse(
        content=[Message(role="assistant", content="Done.")],
        usage_metadata={},
    ))

    agent = Agent(model=mock_client, reflection=False, max_steps=3)
    result = await agent.run("Simple question.")

    assert "reflection" not in agent._toolbox
    assert "need_replan" not in result.context.state


# ── planning + reflection together ────────────────────────────────────────────


def test_agent_planning_and_reflection_both_active() -> None:
    from agentkit.agent import Agent
    from agentkit.llm import LlmClient

    agent = Agent(
        model=MagicMock(spec=LlmClient),
        planning=True,
        reflection=True,
    )
    assert "create_plan" in agent._toolbox
    assert "reflection" in agent._toolbox
    assert REFLECTION_INSTRUCTIONS in agent._effective_instructions


# ── Live tests ─────────────────────────────────────────────────────────────────


@pytest.mark.live
@pytest.mark.asyncio
async def test_live_broken_tool_reflection_switches_to_search() -> None:
    """Agent with a broken get_wikipedia_page and reflection=True should
    call reflection (ERROR ANALYSIS) and then succeed using search_web."""
    from agentkit.agent import Agent
    from agentkit.config import FAST_MODEL, find_uv
    from agentkit.llm import LlmClient
    from agentkit.mcp_client import McpToolset
    from agentkit.tools.base import tool
    from agentkit.tools.mcp import load_mcp_tools

    @tool
    def get_wikipedia_page(title: str) -> str:
        """Fetch a Wikipedia article by title."""
        raise RuntimeError(f"Wikipedia unavailable (title={title!r})")

    mcp_cmd = (find_uv(), ["run", "python", "-m", "agentkit.servers.tavily_server"])
    async with McpToolset(*mcp_cmd) as ts:
        search_tools = list(load_mcp_tools(ts))
        tools = [get_wikipedia_page, *search_tools]

        agent = Agent(
            model=LlmClient(FAST_MODEL),
            tools=tools,
            instructions=(
                "Always try get_wikipedia_page first. "
                "Only use other tools if Wikipedia fails."
            ),
            max_steps=8,
            reflection=True,
        )
        result = await agent.run(
            "What is the speed of light in a vacuum in metres per second?"
        )

    assert result.error is None
    # Agent should have reached an answer (not max_steps message)
    assert "299" in str(result.output), f"Expected speed of light in output: {result.output}"

    # Verify: at least one web search tool was called (switched away from Wikipedia)
    web_calls = [
        item
        for e in result.context.events
        for item in e.content
        if isinstance(item, ToolCall) and "search" in item.name.lower()
    ]
    assert web_calls, "Agent should have switched to search after Wikipedia failed"
