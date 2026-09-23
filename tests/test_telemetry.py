"""Unit tests for OpenTelemetry instrumentation (block 23).

All tests are mocked — no real LLM calls, no API keys required.
Tests are skipped automatically when the opentelemetry SDK is not installed.

Run:
    uv run --group dev --group tracing pytest tests/test_telemetry.py -v
"""
from __future__ import annotations

import pytest

# Skip the whole module if OTel SDK is not installed
otel_sdk = pytest.importorskip(
    "opentelemetry.sdk.trace",
    reason="opentelemetry-sdk not installed; run with --group tracing",
)

from unittest.mock import AsyncMock, patch

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from agentkit.agent import Agent
from agentkit.context import ExecutionContext
from agentkit.llm import LlmClient, LlmResponse
from agentkit.telemetry import _NoOpTracer, get_tracer, reset_tracing, setup_tracing
from agentkit.tools.agent_tool import AgentTool
from agentkit.tools.base import FunctionTool
from agentkit.types import Message, ToolCall
from agentkit.workflow import ParallelWorkflow, SequentialWorkflow, WorkflowStep

# ── Helpers ────────────────────────────────────────────────────────────────────


def make_text_response(text: str = "done") -> LlmResponse:
    return LlmResponse(
        content=[Message(role="assistant", content=text)],
        usage_metadata={"input_tokens": 50, "output_tokens": 5},
    )


def make_tool_call_response(tool_name: str, args: dict) -> LlmResponse:
    return LlmResponse(
        content=[ToolCall(tool_call_id="tc1", name=tool_name, arguments=args)],
        usage_metadata={"input_tokens": 60, "output_tokens": 8},
    )


def _span_names(exporter: InMemorySpanExporter) -> list[str]:
    return [s.name for s in exporter.get_finished_spans()]


def _find(exporter: InMemorySpanExporter, name: str):
    """Return the first span with the given name."""
    for s in exporter.get_finished_spans():
        if s.name == name:
            return s
    return None


def _trace_ids(exporter: InMemorySpanExporter) -> set[int]:
    return {s.context.trace_id for s in exporter.get_finished_spans()}


# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset():
    """Ensure telemetry is reset to no-op after every test."""
    yield
    reset_tracing()


@pytest.fixture
def exporter() -> InMemorySpanExporter:
    """Create a fresh TracerProvider + InMemorySpanExporter and set it up."""
    exp = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exp))
    setup_tracing(_provider=provider)
    return exp


# ── No-op tests ────────────────────────────────────────────────────────────────


def test_default_is_noop():
    """Without setup_tracing, get_tracer() returns _NoOpTracer."""
    assert isinstance(get_tracer(), _NoOpTracer)


def test_noop_exporter_none():
    """setup_tracing(exporter_type='none') keeps no-op tracer."""
    setup_tracing(exporter_type="none")
    assert isinstance(get_tracer(), _NoOpTracer)


def test_noop_span_methods_dont_raise():
    """_NoOpSpan methods must not raise."""
    tracer = _NoOpTracer()
    with tracer.start_as_current_span("test") as span:
        span.set_attribute("k", "v")
        span.add_event("evt", {"a": 1})
        span.record_exception(ValueError("oops"))


async def test_noop_no_spans_recorded():
    """No spans should be produced when tracing is not configured."""
    # No setup_tracing call — global tracer is _NoOpTracer
    model = LlmClient("claude-haiku-4-5")
    agent = Agent(model=model, name="noop_agent", max_steps=1)
    with patch.object(LlmClient, "generate", AsyncMock(return_value=make_text_response())):
        result = await agent.run("hello")
    assert result.output == "done"
    # Verify get_tracer() is still _NoOpTracer (no real exporter was set up)
    assert isinstance(get_tracer(), _NoOpTracer)


# ── gen_ai.chat span ────────────────────────────────────────────────────────────


async def test_llm_span_created(exporter: InMemorySpanExporter):
    """LlmClient.generate produces a gen_ai.chat span with token attributes."""
    model = LlmClient("claude-haiku-4-5")

    import litellm

    fake_usage = type("U", (), {"prompt_tokens": 42, "completion_tokens": 7})()
    fake_choice = type("C", (), {"message": type("M", (), {
        "content": "answer",
        "tool_calls": None,
    })()})()
    fake_raw = type("R", (), {"choices": [fake_choice], "usage": fake_usage})()

    with patch.object(litellm, "acompletion", AsyncMock(return_value=fake_raw)):
        from agentkit.llm import LlmRequest
        resp = await model.generate(LlmRequest(instructions=["sys"], contents=[]))

    assert resp.usage_metadata["input_tokens"] == 42
    spans = exporter.get_finished_spans()
    assert any(s.name == "gen_ai.chat" for s in spans)

    chat_span = _find(exporter, "gen_ai.chat")
    assert chat_span is not None
    assert chat_span.attributes.get("gen_ai.system") == "anthropic"
    assert chat_span.attributes.get("gen_ai.request.model") == "claude-haiku-4-5"
    assert chat_span.attributes.get("gen_ai.usage.input_tokens") == 42
    assert chat_span.attributes.get("gen_ai.usage.output_tokens") == 7


# ── agent.run / agent.step hierarchy ───────────────────────────────────────────


async def test_agent_span_hierarchy(exporter: InMemorySpanExporter):
    """agent.run contains agent.step contains gen_ai.chat — same trace_id."""
    model = LlmClient("claude-haiku-4-5")
    agent = Agent(model=model, name="test_agent", max_steps=2)

    import litellm

    fake_usage = type("U", (), {"prompt_tokens": 10, "completion_tokens": 5})()
    fake_choice = type("C", (), {"message": type("M", (), {
        "content": "answer",
        "tool_calls": None,
    })()})()
    fake_raw = type("R", (), {"choices": [fake_choice], "usage": fake_usage})()

    with patch.object(litellm, "acompletion", AsyncMock(return_value=fake_raw)):
        await agent.run("hello")

    names = _span_names(exporter)
    assert "agent.run" in names
    assert "agent.step" in names
    assert "gen_ai.chat" in names

    # All spans must share the same trace_id
    assert len(_trace_ids(exporter)) == 1, "All spans must belong to one trace"

    # agent.step must be nested under agent.run
    run_span = _find(exporter, "agent.run")
    step_span = _find(exporter, "agent.step")
    chat_span = _find(exporter, "gen_ai.chat")
    assert run_span is not None
    assert step_span is not None
    assert chat_span is not None

    assert step_span.parent.span_id == run_span.context.span_id
    assert chat_span.parent.span_id == step_span.context.span_id


async def test_agent_run_span_attributes(exporter: InMemorySpanExporter):
    """agent.run span carries agent.name and agent.max_steps."""
    model = LlmClient("claude-haiku-4-5")
    agent = Agent(model=model, name="my_agent", max_steps=3)

    with patch.object(LlmClient, "generate", AsyncMock(return_value=make_text_response())):
        await agent.run("q")

    span = _find(exporter, "agent.run")
    assert span is not None
    assert span.attributes.get("agent.name") == "my_agent"
    assert span.attributes.get("agent.max_steps") == 3


async def test_tool_execute_span(exporter: InMemorySpanExporter):
    """tool.execute span is created and nested inside agent.step."""
    model = LlmClient("claude-haiku-4-5")

    call_num = 0
    responses = [
        make_tool_call_response("echo", {"msg": "hi"}),
        make_text_response("all done"),
    ]

    async def side_effect(request):
        nonlocal call_num
        r = responses[call_num]
        call_num += 1
        return r

    async def echo_fn(context: ExecutionContext, msg: str) -> str:
        return f"echoed: {msg}"

    echo_tool = FunctionTool(echo_fn, name="echo", description="Echo")
    agent = Agent(model=model, tools=[echo_tool], name="tool_agent", max_steps=3)

    with patch.object(LlmClient, "generate", side_effect=side_effect):
        await agent.run("test")

    tool_span = _find(exporter, "tool.execute")
    assert tool_span is not None
    assert tool_span.attributes.get("tool.name") == "echo"
    assert tool_span.attributes.get("tool.status") == "success"

    step_span = _find(exporter, "agent.step")
    assert tool_span.parent.span_id == step_span.context.span_id


# ── AgentTool context propagation ─────────────────────────────────────────────


async def test_agent_tool_same_trace_id(exporter: InMemorySpanExporter):
    """CRITICAL: child agent.run must share trace_id with orchestrator.run.

    OTel context propagates automatically through await chains via contextvars.
    AgentTool.execute is awaited inside Agent.act inside the tool.execute span,
    so child agent.run auto-parents to tool.execute — same trace throughout.
    """
    model = LlmClient("claude-haiku-4-5")

    # Responses in call order:
    # 1. orchestrator step 1  → calls child agent tool
    # 2. child agent step 1   → returns text
    # 3. orchestrator step 2  → returns text (after seeing tool result)
    call_num = 0
    responses = [
        make_tool_call_response("child_agent", {"request": "sub-task"}),
        make_text_response("child result"),
        make_text_response("final answer"),
    ]

    async def side_effect(request):
        nonlocal call_num
        r = responses[call_num]
        call_num += 1
        return r

    child = Agent(model=model, name="child_agent", max_steps=2)
    child_tool = AgentTool(child, description="Run child agent.")

    orchestrator = Agent(
        model=model,
        tools=[child_tool],
        name="orchestrator",
        max_steps=4,
    )

    with patch.object(LlmClient, "generate", side_effect=side_effect):
        result = await orchestrator.run("do something")

    assert result.output == "final answer"

    spans = exporter.get_finished_spans()
    trace_ids = _trace_ids(exporter)
    assert len(trace_ids) == 1, (
        f"Expected 1 trace_id, got {len(trace_ids)}: "
        f"{[format(t, '032x') for t in trace_ids]}"
    )

    # Verify hierarchy: orchestrator.agent.run → agent.step → tool.execute[child_agent]
    #                   → child agent.run
    names = [s.name for s in spans]
    assert names.count("agent.run") == 2  # orchestrator + child
    assert "tool.execute" in names

    tool_span = _find(exporter, "tool.execute")
    assert tool_span is not None
    assert tool_span.attributes.get("tool.name") == "child_agent"

    # Find child agent.run (the one nested under tool.execute)
    child_run_spans = [
        s for s in spans
        if s.name == "agent.run" and s.parent
        and s.parent.span_id == tool_span.context.span_id
    ]
    assert len(child_run_spans) == 1, "Child agent.run must be nested under tool.execute"


# ── ParallelWorkflow context propagation ──────────────────────────────────────


async def test_parallel_workflow_same_trace_id(exporter: InMemorySpanExporter):
    """Parallel agent spans share trace_id — asyncio.gather copies contextvars."""
    model = LlmClient("claude-haiku-4-5")

    agent_a = Agent(model=model, name="agent_a", max_steps=1)
    agent_b = Agent(model=model, name="agent_b", max_steps=1)
    wf = ParallelWorkflow([WorkflowStep(agent_a), WorkflowStep(agent_b)])

    with patch.object(LlmClient, "generate", AsyncMock(return_value=make_text_response())):
        await wf.run("hello")

    trace_ids = _trace_ids(exporter)
    assert len(trace_ids) == 1, (
        f"Expected 1 trace_id for parallel workflow, got {len(trace_ids)}"
    )
    assert "workflow.parallel.run" in _span_names(exporter)


# ── SequentialWorkflow spans ───────────────────────────────────────────────────


async def test_sequential_workflow_span(exporter: InMemorySpanExporter):
    """SequentialWorkflow emits a workflow.sequential.run span."""
    model = LlmClient("claude-haiku-4-5")
    agent_a = Agent(model=model, name="step_a", max_steps=1)
    agent_b = Agent(model=model, name="step_b", max_steps=1)
    wf = SequentialWorkflow([WorkflowStep(agent_a), WorkflowStep(agent_b)])

    with patch.object(LlmClient, "generate", AsyncMock(return_value=make_text_response())):
        await wf.run("input")

    assert "workflow.sequential.run" in _span_names(exporter)
    wf_span = _find(exporter, "workflow.sequential.run")
    assert wf_span.attributes.get("workflow.num_steps") == 2

    # Both agent.run spans share the workflow's trace_id
    assert len(_trace_ids(exporter)) == 1


# ── CAPTURE_CONTENT ────────────────────────────────────────────────────────────


async def test_capture_content_false_no_text(exporter: InMemorySpanExporter):
    """CRITICAL security test: when CAPTURE_CONTENT is False (default),
    LLM prompt and completion text must NOT appear in span attributes."""
    import agentkit.telemetry as tel

    original = tel.CAPTURE_CONTENT
    tel.CAPTURE_CONTENT = False
    try:
        import litellm

        fake_usage = type("U", (), {"prompt_tokens": 10, "completion_tokens": 3})()
        fake_choice = type("C", (), {"message": type("M", (), {
            "content": "secret answer",
            "tool_calls": None,
        })()})()
        fake_raw = type("R", (), {"choices": [fake_choice], "usage": fake_usage})()

        with patch.object(litellm, "acompletion", AsyncMock(return_value=fake_raw)):
            from agentkit.llm import LlmRequest

            resp = await LlmClient("claude-haiku-4-5").generate(
                LlmRequest(instructions=["sensitive system prompt"], contents=[])
            )

        assert resp.usage_metadata["input_tokens"] == 10
        chat_span = _find(exporter, "gen_ai.chat")
        assert chat_span is not None
        # Prompt and completion text must NOT be stored
        assert "gen_ai.request.messages" not in chat_span.attributes, (
            "CAPTURE_CONTENT=False must not store prompt text in spans"
        )
        assert "gen_ai.completion" not in chat_span.attributes, (
            "CAPTURE_CONTENT=False must not store completion text in spans"
        )
        # But model and token counts ARE stored
        assert "gen_ai.request.model" in chat_span.attributes
        assert "gen_ai.usage.input_tokens" in chat_span.attributes
    finally:
        tel.CAPTURE_CONTENT = original


async def test_capture_content_true_stores_text(exporter: InMemorySpanExporter):
    """When CAPTURE_CONTENT is True, LLM prompt text IS stored in spans."""
    import agentkit.telemetry as tel

    original = tel.CAPTURE_CONTENT
    tel.CAPTURE_CONTENT = True
    try:
        import litellm

        fake_usage = type("U", (), {"prompt_tokens": 10, "completion_tokens": 3})()
        fake_choice = type("C", (), {"message": type("M", (), {
            "content": "verbose answer",
            "tool_calls": None,
        })()})()
        fake_raw = type("R", (), {"choices": [fake_choice], "usage": fake_usage})()

        with patch.object(litellm, "acompletion", AsyncMock(return_value=fake_raw)):
            from agentkit.llm import LlmRequest

            await LlmClient("claude-haiku-4-5").generate(
                LlmRequest(instructions=["system"], contents=[])
            )

        chat_span = _find(exporter, "gen_ai.chat")
        assert chat_span is not None
        assert "gen_ai.request.messages" in chat_span.attributes
        assert "gen_ai.completion" in chat_span.attributes
    finally:
        tel.CAPTURE_CONTENT = original
