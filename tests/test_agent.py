"""Block 8 tests: Agent, AgentResult, structured output."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from agentkit.agent import Agent, AgentResult
from agentkit.context import ExecutionContext
from agentkit.llm import LlmClient, LlmResponse
from agentkit.tools.base import FunctionTool, tool
from agentkit.types import Event, Message, ToolCall, ToolResult

# ── Helpers ────────────────────────────────────────────────────────────────────


def _mock_model(*responses: LlmResponse) -> MagicMock:
    m = MagicMock(spec=LlmClient)
    m.generate = AsyncMock(side_effect=list(responses))
    return m


def _text_response(text: str) -> LlmResponse:
    return LlmResponse(content=[Message(role="assistant", content=text)])


def _tool_call_response(call_id: str, name: str, args: dict) -> LlmResponse:
    return LlmResponse(
        content=[ToolCall(tool_call_id=call_id, name=name, arguments=args)]
    )


def _error_response(msg: str) -> LlmResponse:
    return LlmResponse(content=[], error_message=msg)


@tool
def fake_tool(x: str) -> str:
    """A simple fake tool."""
    return f"result:{x}"


# ── run: plain text answer ─────────────────────────────────────────────────────


async def test_run_plain_text_one_step():
    agent = Agent(model=_mock_model(_text_response("The answer is 42.")), max_steps=5)
    result = await agent.run("What is the answer?")

    assert isinstance(result, AgentResult)
    assert result.output == "The answer is 42."
    # events: user message + think event
    assert len(result.context.events) == 2
    assert result.context.current_step == 1


# ── run: tool_call → tool_result → text ───────────────────────────────────────


async def test_run_tool_call_then_text_two_steps():
    model = _mock_model(
        _tool_call_response("tc-1", "fake_tool", {"x": "hello"}),
        _text_response("Final answer."),
    )
    agent = Agent(model=model, tools=[fake_tool], max_steps=5)
    result = await agent.run("Do something.")

    assert result.output == "Final answer."
    # events: user, think(tool_call), tool_results, think(text)
    assert len(result.context.events) == 4
    assert result.context.current_step == 2

    # verify event order and types
    events = result.context.events
    assert any(isinstance(i, Message) and i.role == "user" for i in events[0].content)
    assert any(isinstance(i, ToolCall) for i in events[1].content)
    assert any(isinstance(i, ToolResult) for i in events[2].content)
    assert any(isinstance(i, Message) and i.role == "assistant" for i in events[3].content)


# ── act: unknown tool → error ToolResult ──────────────────────────────────────


async def test_act_unknown_tool_returns_error_result():
    agent = Agent(model=MagicMock(), tools=[fake_tool], max_steps=5)
    ctx = ExecutionContext()
    call = ToolCall(tool_call_id="tc-x", name="nonexistent", arguments={})

    results = await agent.act(ctx, [call])

    assert len(results) == 1
    assert results[0].status == "error"
    assert "Unknown tool" in results[0].content[0]


# ── act: tool raises → error ToolResult, loop continues ──────────────────────


async def test_act_tool_exception_returns_error_result():
    @tool
    def broken(x: str) -> str:
        """Always raises."""
        raise ValueError("kaboom")

    agent = Agent(model=MagicMock(), tools=[broken], max_steps=5)
    ctx = ExecutionContext()
    call = ToolCall(tool_call_id="tc-y", name="broken", arguments={"x": "oops"})

    results = await agent.act(ctx, [call])

    assert results[0].status == "error"
    assert "kaboom" in results[0].content[0]


async def test_run_tool_exception_loop_continues():
    """Tool raises → error ToolResult → agent continues to final text answer."""

    @tool
    def exploding(x: str) -> str:
        """Explodes."""
        raise RuntimeError("boom")

    model = _mock_model(
        _tool_call_response("tc-1", "exploding", {"x": "oops"}),
        _text_response("Recovered."),
    )
    agent = Agent(model=model, tools=[exploding], max_steps=5)
    result = await agent.run("Do it.")

    assert result.output == "Recovered."
    tool_result_event = result.context.events[2]
    tr = tool_result_event.content[0]
    assert isinstance(tr, ToolResult)
    assert tr.status == "error"


# ── max_steps ─────────────────────────────────────────────────────────────────


async def test_run_max_steps_terminates():
    """Agent always getting tool_call must stop at max_steps, not loop forever."""
    always_tool = _tool_call_response("tc-inf", "fake_tool", {"x": "loop"})
    model = MagicMock(spec=LlmClient)
    model.generate = AsyncMock(return_value=always_tool)

    agent = Agent(model=model, tools=[fake_tool], max_steps=3)
    result = await agent.run("Loop forever.")

    assert "[max_steps=3" in result.output
    assert result.context.current_step == 3


# ── error_message ──────────────────────────────────────────────────────────────


async def test_run_llm_error_sets_error_field():
    """LlmResponse with error_message → AgentResult.error is set, output is empty."""
    agent = Agent(model=_mock_model(_error_response("connection failed")), max_steps=5)
    result = await agent.run("hi")

    assert result.error is not None
    assert "connection failed" in result.error
    assert result.output == ""


async def test_run_config_error_propagates_as_exception():
    """Configuration errors (bad key, unknown model) must not be swallowed."""
    import litellm

    mock_model = MagicMock(spec=LlmClient)
    mock_model.generate = AsyncMock(
        side_effect=litellm.AuthenticationError(
            message="Invalid API key",
            llm_provider="anthropic",
            model="claude-haiku",
        )
    )
    agent = Agent(model=mock_model, max_steps=5)
    with pytest.raises(litellm.AuthenticationError):
        await agent.run("hi")


# ── structured output ─────────────────────────────────────────────────────────


class _Out(BaseModel):
    answer: str
    confidence: float = 1.0


def test_setup_tools_adds_final_answer():
    agent = Agent(model=MagicMock(), output_type=_Out, max_steps=5)
    assert "final_answer" in agent._toolbox
    assert agent.output_tool_name == "final_answer"
    schema = agent._toolbox["final_answer"].tool_definition
    assert schema["function"]["name"] == "final_answer"
    assert "properties" in schema["function"]["parameters"]


def test_prepare_request_tool_choice_required_with_output_type():
    agent = Agent(model=MagicMock(), output_type=_Out, max_steps=5)
    ctx = ExecutionContext()
    req = agent._prepare_llm_request(ctx)
    assert req.tool_choice == "required"


def test_prepare_request_tool_choice_auto_with_tools():
    agent = Agent(model=MagicMock(), tools=[fake_tool], max_steps=5)
    ctx = ExecutionContext()
    req = agent._prepare_llm_request(ctx)
    assert req.tool_choice == "auto"


def test_prepare_request_tool_choice_none_no_tools():
    agent = Agent(model=MagicMock(), max_steps=5)
    ctx = ExecutionContext()
    req = agent._prepare_llm_request(ctx)
    assert req.tool_choice is None


def test_is_final_response_false_with_tool_call():
    agent = Agent(model=MagicMock(), max_steps=5)
    event = Event(
        execution_id="test",
        author="agent",
        content=[
            ToolCall(tool_call_id="tc-1", name="some_tool", arguments={}),
            Message(role="assistant", content="Let me search..."),
        ],
    )
    assert not agent._is_final_response(event)


def test_is_final_response_true_plain_message():
    agent = Agent(model=MagicMock(), max_steps=5)
    event = Event(
        execution_id="test",
        author="agent",
        content=[Message(role="assistant", content="42")],
    )
    assert agent._is_final_response(event)


def test_is_final_response_structured_true_for_output_tool():
    agent = Agent(model=MagicMock(), output_type=_Out, max_steps=5)
    event = Event(
        execution_id="test",
        author="agent",
        content=[
            ToolCall(
                tool_call_id="tc-1",
                name="final_answer",
                arguments={"answer": "Paris", "confidence": 0.9},
            )
        ],
    )
    assert agent._is_final_response(event)


def test_is_final_response_structured_false_for_other_tool():
    agent = Agent(model=MagicMock(), output_type=_Out, max_steps=5)
    event = Event(
        execution_id="test",
        author="agent",
        content=[ToolCall(tool_call_id="tc-1", name="search", arguments={"q": "x"})],
    )
    assert not agent._is_final_response(event)


async def test_run_structured_output_returns_model_instance():
    final_call = LlmResponse(
        content=[
            ToolCall(
                tool_call_id="tc-1",
                name="final_answer",
                arguments={"answer": "Paris", "confidence": 0.95},
            )
        ]
    )
    agent = Agent(model=_mock_model(final_call), output_type=_Out, max_steps=5)
    result = await agent.run("Capital of France?")

    assert isinstance(result.output, _Out)
    assert result.output.answer == "Paris"


# ── token tracking ─────────────────────────────────────────────────────────────


async def test_token_usage_accumulated_in_context():
    resp1 = LlmResponse(
        content=[ToolCall(tool_call_id="tc-1", name="fake_tool", arguments={"x": "a"})],
        usage_metadata={"input_tokens": 100, "output_tokens": 20},
    )
    resp2 = LlmResponse(
        content=[Message(role="assistant", content="done")],
        usage_metadata={"input_tokens": 200, "output_tokens": 30},
    )
    agent = Agent(model=_mock_model(resp1, resp2), tools=[fake_tool], max_steps=5)
    result = await agent.run("go")

    usage = result.context.state["token_usage"]
    assert usage["input_tokens"] == 300
    assert usage["output_tokens"] == 50


# ── live tests ─────────────────────────────────────────────────────────────────


@pytest.mark.live
async def test_agent_live_calculator():
    from agentkit.config import FAST_MODEL
    from agentkit.llm import LlmClient
    from agentkit.tools.base import FunctionTool
    from agentkit.tools_manual import calculator

    agent = Agent(
        model=LlmClient(FAST_MODEL),
        tools=[FunctionTool(calculator)],
        instructions="Use the calculator tool for arithmetic.",
        max_steps=5,
    )
    result = await agent.run("What is 1234 multiplied by 5678?")
    assert "7006652" in str(result.output)


@pytest.mark.live
async def test_agent_live_structured_output():
    from typing import Literal

    from agentkit.config import FAST_MODEL
    from agentkit.llm import LlmClient

    class Sentiment(BaseModel):
        sentiment: Literal["positive", "negative", "neutral"]
        confidence: float

    agent = Agent(
        model=LlmClient(FAST_MODEL),
        output_type=Sentiment,
        instructions="Analyze the sentiment of the text.",
        max_steps=3,
    )
    result = await agent.run("I love this product!")
    assert isinstance(result.output, Sentiment)
    assert result.output.sentiment in ("positive", "negative", "neutral")


@pytest.mark.live
async def test_agent_live_search_web():
    from agentkit.config import FAST_MODEL
    from agentkit.llm import LlmClient
    from agentkit.tools.web import search_web

    agent = Agent(
        model=LlmClient(FAST_MODEL),
        tools=[FunctionTool(search_web)],
        instructions="Use search_web to answer questions about recent events.",
        max_steps=5,
    )
    result = await agent.run("Who won the Nobel Prize in Physics in 2025?")
    assert result.output
    assert not str(result.output).startswith("[llm_error]")
