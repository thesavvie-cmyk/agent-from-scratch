"""Chapter 9 Agent-as-Tool and Transfer experiments (block 21).

Sections
--------
a) Orchestrator with researcher + coder as AgentTools.
   Three requests: research-only / code-only / both.
   Shows which child agents are called and what their traces look like.

b) Isolation cost: compare tokens for orchestrator+AgentTools vs
   SequentialWorkflow(share_context=True) for the same pipeline.

c) Transfer: dispatcher + two domain specialists.
   Domain-specific requests confirm correct routing.

d) Ping-pong: two agents that keep transferring to each other.
   Verifies the transfer limit fires.

e) Debug trace: researcher gets a broken tool; show how the failure
   appears in the orchestrator trace and in child_traces.

Usage
-----
    uv run python experiments/ch09_orchestration.py --section a
    uv run python experiments/ch09_orchestration.py --section all

Requires ANTHROPIC_API_KEY + TAVILY_API_KEY.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

from agentkit.agent import Agent
from agentkit.agents.specialists import make_coder, make_researcher, make_writer
from agentkit.config import FAST_MODEL, find_uv
from agentkit.llm import LlmClient
from agentkit.mcp_client import McpToolset
from agentkit.tools.agent_tool import AgentTool
from agentkit.tools.base import FunctionTool
from agentkit.tools.mcp import load_mcp_tools
from agentkit.transfer import TransferOrchestrator
from agentkit.types import ToolCall
from agentkit.workflow import SequentialWorkflow, WorkflowStep

MCP_CMD = (find_uv(), ["run", "python", "-m", "agentkit.servers.tavily_server"])


def _hr(title: str) -> None:
    print(f"\n{'=' * 68}\n  {title}\n{'=' * 68}")


def _count_child_calls(context: object, tool_name: str) -> int:
    from agentkit.context import ExecutionContext
    if not isinstance(context, ExecutionContext):
        return 0
    return sum(
        1 for ev in context.events
        for item in ev.content
        if isinstance(item, ToolCall) and item.name == tool_name
    )


# ── Section A — Orchestrator routing ──────────────────────────────────────────


async def section_a(model: LlmClient, search_tools: list) -> None:
    _hr("Section A — Orchestrator with researcher + coder as AgentTools")

    researcher_tool = AgentTool(
        make_researcher(model, tools=search_tools),
        description="Search for factual information, statistics, and current events.",
    )
    coder_tool = AgentTool(
        make_coder(model),
        description="Write Python code to solve algorithmic or data-processing tasks.",
    )

    orchestrator = Agent(
        model=model,
        tools=[researcher_tool, coder_tool],
        name="orchestrator",
        instructions=(
            "You coordinate tasks. Use researcher for factual lookups, "
            "coder for writing code. Call only what is needed."
        ),
        max_steps=8,
    )

    queries = [
        "What is the population of Tokyo?",
        "Write a Python function that returns the nth Fibonacci number.",
        "Find the boiling point of water in Fahrenheit and write code to convert it to Celsius.",
    ]

    for q in queries:
        print(f"\n  Q: {q}")
        t0 = time.perf_counter()
        result = await orchestrator.run(q)
        elapsed = time.perf_counter() - t0

        traces = result.context.state.get("child_traces", {})
        called = [v["agent"] for v in traces.values()]
        u = result.context.state.get("token_usage", {})

        print(f"  Child agents called: {called}")
        print(f"  Steps={result.context.current_step}  in={u.get('input_tokens',0)}  elapsed={elapsed:.1f}s")
        print(f"  Answer: {str(result.output)[:150]}")


# ── Section B — Isolation cost ─────────────────────────────────────────────────


async def section_b(model: LlmClient, search_tools: list) -> None:
    _hr("Section B — AgentTool isolation vs SequentialWorkflow(share_context=True)")

    topic = "history of the Python programming language"

    # Approach 1: orchestrator + AgentTool (isolated contexts)
    researcher_tool = AgentTool(
        make_researcher(model, tools=search_tools),
        description="Research a topic in depth.",
    )
    writer_tool = AgentTool(
        make_writer(model),
        description="Write an article from research notes.",
    )
    orchestrator = Agent(
        model=model,
        tools=[researcher_tool, writer_tool],
        name="orchestrator",
        instructions=(
            "Research the topic, then write an article. "
            "Use researcher first, then writer."
        ),
        max_steps=6,
    )
    t0 = time.perf_counter()
    orch_result = await orchestrator.run(f"Write a short article about: {topic}")
    orch_elapsed = time.perf_counter() - t0
    orch_u = orch_result.context.state.get("token_usage", {})
    orch_child_traces = orch_result.context.state.get("child_traces", {})
    # Approach 2: SequentialWorkflow with share_context=True
    t0 = time.perf_counter()
    wf_result = await SequentialWorkflow([
        WorkflowStep(make_researcher(model, tools=search_tools), share_context=True),
        WorkflowStep(make_writer(model), share_context=True),
    ]).run(topic)
    wf_elapsed = time.perf_counter() - t0
    wf_tokens = wf_result.total_tokens()
    wf_last_in = wf_result.tokens_per_step()[-1]["input_tokens"]

    print(f"\n  {'Approach':<30} {'Total in':>10} {'Total out':>10} {'Time':>8}")
    print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*8}")
    print(
        f"  {'AgentTool (isolated)':<30} "
        f"{orch_u.get('input_tokens', 0):>10,} "
        f"{orch_u.get('output_tokens', 0):>10,} "
        f"{orch_elapsed:>7.1f}s"
    )
    print(
        f"  {'Workflow share_ctx=True':<30} "
        f"{wf_tokens['input_tokens']:>10,} "
        f"{wf_tokens['output_tokens']:>10,} "
        f"{wf_elapsed:>7.1f}s"
    )
    print(
        f"\n  Workflow: last agent (writer) saw {wf_last_in:,} input tokens "
        f"(includes full history)."
    )
    print(
        f"  AgentTool: child traces={len(orch_child_traces)} — "
        f"each child got only its own context."
    )
    print(
        "\n  Interpretation: AgentTool keeps each child's token cost low but "
        "loses history between steps. share_context accumulates cost."
    )


# ── Section C — Transfer routing ───────────────────────────────────────────────


async def section_c(model: LlmClient, search_tools: list) -> None:
    _hr("Section C — Transfer: dispatcher + two domain specialists")

    # Domain A: Python / programming
    coder = Agent(
        model=model,
        tools=[],
        name="python_expert",
        instructions=(
            "You are a Python programming expert.\n"
            "Answer Python and software development questions concisely."
        ),
        max_steps=4,
    )

    # Domain B: history / geography
    historian = Agent(
        model=model,
        tools=search_tools,
        name="historian",
        instructions=(
            "You are a history and geography expert.\n"
            "Answer historical and geographical questions concisely."
        ),
        max_steps=4,
    )

    # Dispatcher
    dispatcher = Agent(
        model=model,
        tools=[],
        name="dispatcher",
        instructions=(
            "You route questions to the right specialist.\n"
            "Use transfer_to('python_expert') for programming questions.\n"
            "Use transfer_to('historian') for history and geography questions.\n"
            "Always transfer — never answer directly."
        ),
        max_steps=3,
    )

    orch = TransferOrchestrator(
        {"dispatcher": dispatcher, "python_expert": coder, "historian": historian},
        entry_agent="dispatcher",
        max_transfers=2,
    )

    queries = [
        "What does the `yield` keyword do in Python?",
        "What year did World War II end?",
    ]

    for q in queries:
        print(f"\n  Q: {q}")
        result = await orch.run(q)
        log_str = " → ".join(f"{t['from']}→{t['to']}" for t in result.transfer_log)
        print(f"  Transfers: {log_str or '(none)'}")
        print(f"  Error: {result.error or '—'}")
        print(f"  Answer: {str(result.output)[:150]}")


# ── Section D — Ping-pong ──────────────────────────────────────────────────────


async def section_d(model: LlmClient) -> None:
    _hr("Section D — Ping-pong: transfer limit fires")

    # Two agents that always transfer to each other
    agent_a = Agent(
        model=model,
        tools=[],
        name="agent_a",
        instructions=(
            "Always transfer_to('agent_b') with reason 'agent_b handles this better'."
        ),
        max_steps=3,
    )
    agent_b = Agent(
        model=model,
        tools=[],
        name="agent_b",
        instructions=(
            "Always transfer_to('agent_a') with reason 'agent_a handles this better'."
        ),
        max_steps=3,
    )

    orch = TransferOrchestrator(
        {"agent_a": agent_a, "agent_b": agent_b},
        entry_agent="agent_a",
        max_transfers=4,
    )

    result = await orch.run("Who should answer this?")

    print(f"\n  Transfer log ({len(result.transfer_log)} entries):")
    for entry in result.transfer_log:
        print(f"    {entry['from']} → {entry['to']}: {entry['reason']}")
    print(f"\n  Error: {result.error}")
    print(f"  Limit of 4 transfers {'FIRED ✓' if result.error else 'did NOT fire ✗'}")


# ── Section E — Debug trace ────────────────────────────────────────────────────


async def section_e(model: LlmClient) -> None:
    _hr("Section E — Debug: broken tool in researcher, trace in orchestrator")

    # Researcher with a tool that always raises
    async def broken_search(context: object, query: str) -> str:
        raise RuntimeError("Search service unavailable")

    broken_tool = FunctionTool(broken_search, name="search_web", description="Search the web")

    broken_researcher = Agent(
        model=model,
        tools=[broken_tool],
        name="researcher",
        instructions="Search for information. Use search_web.",
        max_steps=3,
    )

    researcher_agent_tool = AgentTool(
        broken_researcher,
        description="Research a topic.",
    )

    orchestrator = Agent(
        model=model,
        tools=[researcher_agent_tool],
        name="orchestrator",
        instructions="Use the researcher tool to answer questions.",
        max_steps=4,
    )

    result = await orchestrator.run("What is the capital of France?")
    child_traces = result.context.state.get("child_traces", {})

    print(f"\n  Orchestrator error: {result.error or '—'}")
    print(f"  Orchestrator output: {str(result.output)[:150]}")
    print(f"\n  Child traces saved: {len(child_traces)}")
    for tid, trace in child_traces.items():
        print(f"    [{tid}] agent={trace['agent']}  steps={trace['steps']}  error={trace['error']!r}")

    print("\n  Orchestrator trace (tool calls):")
    for ev in result.context.events:
        for item in ev.content:
            if isinstance(item, ToolCall):
                print(f"    CALL  [{ev.author}] {item.name}({str(item.arguments)[:60]})")


# ── Main ───────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Block 21 orchestration experiments")
    parser.add_argument(
        "--section", default="a",
        choices=["a", "b", "c", "d", "e", "all"],
    )
    return parser.parse_args()


async def _main(args: argparse.Namespace) -> None:
    async with McpToolset(*MCP_CMD) as ts:
        search_tools = list(load_mcp_tools(ts))

    model = LlmClient(FAST_MODEL)

    if args.section in ("a", "all"):
        await section_a(model, search_tools)
    if args.section in ("b", "all"):
        await section_b(model, search_tools)
    if args.section in ("c", "all"):
        await section_c(model, search_tools)
    if args.section in ("d", "all"):
        await section_d(model)
    if args.section in ("e", "all"):
        await section_e(model)


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
