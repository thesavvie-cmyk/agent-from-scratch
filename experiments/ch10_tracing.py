"""Chapter 10 OpenTelemetry observability experiments (block 23).

Sections
--------
a) No-op baseline — verify zero spans when tracing not configured.
b) Span hierarchy — run a mock agent, print the span tree.
c) AgentTool propagation — verify child agent shares trace_id with parent.
d) ParallelWorkflow — verify sibling spans share one trace_id.
e) FileSpanExporter — write a trace to JSONL, read it back, print tree.

Sections a–e all use a mock LLM — no API key needed.

Usage
-----
    uv run --group tracing python experiments/ch10_tracing.py --section all
    uv run --group tracing python experiments/ch10_tracing.py --section c

Live experiments (need API key + Tavily):
    uv run --group tracing python experiments/ch10_tracing.py --section live

Requires: opentelemetry-sdk (install with --group tracing)
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, patch

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

try:
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )
except ImportError:
    print("opentelemetry-sdk not installed.  Run: uv sync --group tracing", file=sys.stderr)
    sys.exit(1)

from agentkit.agent import Agent
from agentkit.llm import LlmClient, LlmResponse
from agentkit.telemetry import (
    FileSpanExporter,
    _NoOpTracer,
    _span_to_dict,
    get_tracer,
    reset_tracing,
    setup_tracing,
)
from agentkit.tools.agent_tool import AgentTool
from agentkit.types import Message, ToolCall
from agentkit.workflow import ParallelWorkflow, WorkflowStep

# ── Mock LLM helpers ───────────────────────────────────────────────────────────


def _text(text: str = "done") -> LlmResponse:
    return LlmResponse(
        content=[Message(role="assistant", content=text)],
        usage_metadata={"input_tokens": 50, "output_tokens": 5},
    )


def _tool_call(tool_name: str, args: dict) -> LlmResponse:
    return LlmResponse(
        content=[ToolCall(tool_call_id="tc1", name=tool_name, arguments=args)],
        usage_metadata={"input_tokens": 60, "output_tokens": 8},
    )


def _hr(title: str) -> None:
    print(f"\n{'=' * 68}\n  {title}\n{'=' * 68}")


def _setup_memory_exporter() -> InMemorySpanExporter:
    """Create a fresh in-memory exporter and wire it up."""
    exp = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exp))
    setup_tracing(_provider=provider)
    return exp


def _print_spans(exp: InMemorySpanExporter) -> None:
    """Simple tree print from in-memory spans."""
    from scripts.trace_tree import print_traces

    spans = [_span_to_dict(s) for s in exp.get_finished_spans()]
    print_traces(spans)


# ── Section A — No-op baseline ─────────────────────────────────────────────────


async def section_a() -> None:
    _hr("Section A — No-op baseline (no setup_tracing)")
    reset_tracing()  # ensure no exporter
    assert isinstance(get_tracer(), _NoOpTracer), "Expected _NoOpTracer"

    model = LlmClient("claude-haiku-4-5")
    agent = Agent(model=model, name="noop_agent", max_steps=1)

    with patch.object(LlmClient, "generate", AsyncMock(return_value=_text("hi"))):
        t0 = time.perf_counter()
        result = await agent.run("hello")
        elapsed = time.perf_counter() - t0

    print(f"\n  Result: {result.output}")
    print(f"  Elapsed: {elapsed * 1000:.1f}ms  (no exporter overhead)")
    print("  Tracer type:", type(get_tracer()).__name__)
    print("\n  PASS: no spans collected, execution proceeds normally.")


# ── Section B — Span hierarchy ─────────────────────────────────────────────────


async def section_b() -> None:
    _hr("Section B — Span hierarchy: agent.run → agent.step → gen_ai.chat")

    exp = _setup_memory_exporter()
    model = LlmClient("claude-haiku-4-5")
    agent = Agent(model=model, name="demo_agent", max_steps=2)

    import litellm

    fake_usage = type("U", (), {"prompt_tokens": 30, "completion_tokens": 8})()
    fake_choice = type("C", (), {"message": type("M", (), {
        "content": "42",
        "tool_calls": None,
    })()})()
    fake_raw = type("R", (), {"choices": [fake_choice], "usage": fake_usage})()

    with patch.object(litellm, "acompletion", AsyncMock(return_value=fake_raw)):
        result = await agent.run("What is 6 * 7?")

    print(f"\n  Answer: {result.output}")
    print(f"  Spans collected: {len(exp.get_finished_spans())}")
    _print_spans(exp)

    trace_ids = {s.context.trace_id for s in exp.get_finished_spans()}
    assert len(trace_ids) == 1, f"Expected 1 trace_id, got {len(trace_ids)}"
    print("\n  PASS: all spans share one trace_id.")
    reset_tracing()


# ── Section C — AgentTool propagation ─────────────────────────────────────────


async def section_c() -> None:
    _hr("Section C — AgentTool: child span inherits parent trace_id")

    exp = _setup_memory_exporter()
    model = LlmClient("claude-haiku-4-5")

    call_num = 0
    responses = [
        _tool_call("researcher", {"request": "find facts"}),
        _text("42 is the answer"),   # child agent
        _text("final answer"),       # orchestrator after tool result
    ]

    async def side_effect(request):
        nonlocal call_num
        r = responses[call_num]
        call_num += 1
        return r

    child = Agent(model=model, name="researcher", max_steps=2)
    child_tool = AgentTool(child, description="Research facts.")
    orch = Agent(model=model, tools=[child_tool], name="orchestrator", max_steps=4)

    with patch.object(LlmClient, "generate", side_effect=side_effect):
        result = await orch.run("find the answer")

    spans = exp.get_finished_spans()
    trace_ids = {s.context.trace_id for s in spans}

    print(f"\n  Result: {result.output}")
    print(f"  Total spans: {len(spans)}")
    print(f"  Distinct trace_ids: {len(trace_ids)}")
    _print_spans(exp)

    assert len(trace_ids) == 1, (
        f"FAIL: expected 1 trace_id, got {len(trace_ids)}\n"
        "Child agent spans are not connected to the parent trace!"
    )
    print("\n  PASS: orchestrator + child agent share one trace_id.")
    reset_tracing()


# ── Section D — Parallel workflow ──────────────────────────────────────────────


async def section_d() -> None:
    _hr("Section D — ParallelWorkflow: sibling spans share trace_id")

    exp = _setup_memory_exporter()
    model = LlmClient("claude-haiku-4-5")

    agent_a = Agent(model=model, name="researcher_A", max_steps=1)
    agent_b = Agent(model=model, name="researcher_B", max_steps=1)
    wf = ParallelWorkflow([WorkflowStep(agent_a), WorkflowStep(agent_b)])

    with patch.object(LlmClient, "generate", AsyncMock(return_value=_text("result"))):
        await wf.run("research topic X")

    spans = exp.get_finished_spans()
    trace_ids = {s.context.trace_id for s in spans}
    names = [s.name for s in spans]

    print(f"\n  Total spans: {len(spans)}")
    print(f"  Span names: {names}")
    print(f"  Distinct trace_ids: {len(trace_ids)}")
    _print_spans(exp)

    assert len(trace_ids) == 1, f"FAIL: expected 1 trace_id, got {len(trace_ids)}"
    wf_span = next((s for s in spans if s.name == "workflow.parallel.run"), None)
    assert wf_span is not None
    # Both agent.run spans should be nested under the workflow span
    agent_runs = [s for s in spans if s.name == "agent.run"]
    assert len(agent_runs) == 2
    for ar in agent_runs:
        assert ar.parent.span_id == wf_span.context.span_id, (
            "agent.run should be nested under workflow.parallel.run"
        )
    print("\n  PASS: both parallel agents nested under workflow.parallel.run.")
    reset_tracing()


# ── Section E — FileSpanExporter ───────────────────────────────────────────────


async def section_e() -> None:
    _hr("Section E — FileSpanExporter: write JSONL, read back, print tree")

    with tempfile.NamedTemporaryFile(
        suffix=".jsonl", delete=False, mode="w", encoding="utf-8"
    ) as tmp:
        trace_path = Path(tmp.name)

    try:
        # Setup file exporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        resource = Resource.create({"service.name": "demo"})
        provider = TracerProvider(resource=resource)
        file_exp = FileSpanExporter(trace_path)
        provider.add_span_processor(BatchSpanProcessor(file_exp))
        setup_tracing(_provider=provider)

        model = LlmClient("claude-haiku-4-5")
        agent = Agent(model=model, name="file_agent", max_steps=1)

        import litellm

        fake_usage = type("U", (), {"prompt_tokens": 20, "completion_tokens": 4})()
        fake_choice = type("C", (), {"message": type("M", (), {
            "content": "Paris",
            "tool_calls": None,
        })()})()
        fake_raw = type("R", (), {"choices": [fake_choice], "usage": fake_usage})()

        with patch.object(litellm, "acompletion", AsyncMock(return_value=fake_raw)):
            result = await agent.run("Capital of France?")

        # Force flush
        provider.force_flush()

        print(f"\n  Answer: {result.output}")
        print(f"  Trace file: {trace_path}")

        lines = trace_path.read_text(encoding="utf-8").strip().splitlines()
        print(f"  Lines in JSONL: {len(lines)}")

        import json

        from scripts.trace_tree import print_traces

        spans = [json.loads(line) for line in lines if line.strip()]
        print_traces(spans)
        print("\n  PASS: spans written and read back successfully.")
    finally:
        trace_path.unlink(missing_ok=True)
        reset_tracing()


# ── Live section (needs API key) ───────────────────────────────────────────────


async def section_live() -> None:
    _hr("Section LIVE — real agent run with file exporter (needs API key + Tavily)")

    from agentkit.config import FAST_MODEL, find_uv
    from agentkit.mcp_client import McpToolset
    from agentkit.tools.mcp import load_mcp_tools

    trace_path = Path("results/traces/live_trace.jsonl")
    print(f"  Writing traces to {trace_path}")
    setup_tracing(service_name="agentkit-live", exporter_type="file", filepath=str(trace_path))

    MCP_CMD = (find_uv(), ["run", "python", "-m", "agentkit.servers.tavily_server"])
    async with McpToolset(*MCP_CMD) as ts:
        search_tools = list(load_mcp_tools(ts))

    model = LlmClient(FAST_MODEL)
    agent = Agent(model=model, tools=search_tools, name="live_agent", max_steps=4)
    result = await agent.run("What is the capital of France?")
    print(f"\n  Answer: {result.output}")
    print(f"  Steps: {result.context.current_step}")
    print(f"\n  View with: uv run python scripts/trace_tree.py {trace_path}")
    reset_tracing()


# ── Main ───────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Block 23 tracing experiments")
    parser.add_argument(
        "--section", default="b",
        choices=["a", "b", "c", "d", "e", "all", "live"],
    )
    return parser.parse_args()


async def _main(args: argparse.Namespace) -> None:
    if args.section in ("a", "all"):
        await section_a()
    if args.section in ("b", "all"):
        await section_b()
    if args.section in ("c", "all"):
        await section_c()
    if args.section in ("d", "all"):
        await section_d()
    if args.section in ("e", "all"):
        await section_e()
    if args.section == "live":
        await section_live()


def main() -> None:
    args = _parse_args()
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try:
        asyncio.run(_main(args))
    except KeyboardInterrupt:
        print("\n[interrupted]")
        sys.exit(1)
    print()


if __name__ == "__main__":
    main()
