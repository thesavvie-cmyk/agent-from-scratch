"""Chapter 10 A2A (Agent-to-Agent) protocol experiments (block 22).

Sections
--------
a) AgentCard discovery — start a server, fetch its card, inspect capabilities.
b) A2ATool vs AgentTool — orchestrator calls the same researcher both ways;
   compare round-trip latency (in-process vs HTTP).
c) Two-agent pipeline over A2A — researcher and writer each exposed as
   separate HTTP servers; orchestrator wires them with A2ATools.
d) Error propagation — remote agent raises; A2ATool surfaces the error string;
   orchestrator continues gracefully.

Usage
-----
    uv run --group a2a python experiments/ch10_a2a.py --section a
    uv run --group a2a python experiments/ch10_a2a.py --section all

No API key required — all agents use a mocked LLM.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

from agentkit.a2a import A2AClient, A2AServer, A2ATool, AgentCard, TaskStatus
from agentkit.agent import Agent, AgentResult
from agentkit.context import ExecutionContext
from agentkit.tools.agent_tool import AgentTool

# ── mock helpers ──────────────────────────────────────────────────────────────


def _mock_llm(reply: str) -> Any:
    """Return a minimal LlmClient mock that always replies with *reply*."""
    from agentkit.llm import LlmResponse
    from agentkit.types import Message

    response = LlmResponse(
        content=[Message(role="assistant", content=reply)],
        usage_metadata={"input_tokens": 10, "output_tokens": len(reply.split())},
    )
    llm = MagicMock()
    llm.generate = AsyncMock(return_value=response)
    return llm


def _mock_agent(name: str, answer: str) -> Any:
    """Return a minimal mock Agent that always returns *answer*."""
    result = AgentResult(output=answer, context=ExecutionContext(), error=None)
    agent = MagicMock(spec=Agent)
    agent.name = name
    agent.run = AsyncMock(return_value=result)
    return agent


def _hr(title: str) -> None:
    print(f"\n{'=' * 68}\n  {title}\n{'=' * 68}")


# ── Section A — AgentCard discovery ──────────────────────────────────────────


async def section_a() -> None:
    _hr("Section A — AgentCard discovery")

    agent = _mock_agent("researcher", "Python was created by Guido van Rossum in 1991.")

    async with A2AServer(
        agent,
        name="researcher",
        description="Searches for factual information.",
        port=0,
    ) as server:
        url = f"http://127.0.0.1:{server.actual_port}"
        client = A2AClient(url)
        card: AgentCard = await client.get_card()

    print(f"\n  URL:         {url}")
    print(f"  Name:        {card.name}")
    print(f"  Description: {card.description}")
    print(f"  Version:     {card.version}")
    print(f"  Capabilities: {card.capabilities}")
    print(
        "\n  The AgentCard is the 'business card' of an A2A agent.\n"
        "  Any client can fetch it before deciding whether to delegate a task."
    )


# ── Section B — A2ATool vs AgentTool latency ─────────────────────────────────


async def section_b() -> None:
    _hr("Section B — A2ATool (HTTP) vs AgentTool (in-process) latency")

    answer = (
        "Python f-strings (PEP 498) are string literals prefixed with 'f'. "
        "Expressions inside {} are evaluated at runtime."
    )
    question = "Explain Python f-strings in one sentence."

    # ── AgentTool (in-process) ─────────────────────────────────────────────
    in_proc_agent = _mock_agent("researcher", answer)
    in_proc_tool = AgentTool(in_proc_agent, description="Research facts in-process.")

    t0 = time.perf_counter()
    for _ in range(5):
        from agentkit.context import ExecutionContext as _EC
        await in_proc_tool.execute(_EC(), task=question)
    in_proc_elapsed = (time.perf_counter() - t0) / 5

    # ── A2ATool (HTTP) ────────────────────────────────────────────────────
    http_agent = _mock_agent("researcher", answer)
    async with A2AServer(http_agent, name="researcher", port=0) as server:
        url = f"http://127.0.0.1:{server.actual_port}"
        a2a_tool = A2ATool(
            A2AClient(url),
            name="researcher",
            description="Research facts via A2A.",
        )

        t0 = time.perf_counter()
        for _ in range(5):
            await a2a_tool.execute(ExecutionContext(), task=question)
        http_elapsed = (time.perf_counter() - t0) / 5

    print(f"\n  {'Method':<25} {'Avg latency (5 calls)':>22}")
    print(f"  {'-'*25} {'-'*22}")
    print(f"  {'AgentTool (in-process)':<25} {in_proc_elapsed*1000:>19.1f} ms")
    print(f"  {'A2ATool (HTTP)':<25} {http_elapsed*1000:>19.1f} ms")
    print(
        "\n  Interpretation: HTTP overhead is the price of network isolation.\n"
        "  In-process AgentTool is faster; A2ATool enables heterogeneous\n"
        "  agents (different languages, machines, or framework versions)."
    )


# ── Section C — Two-agent pipeline over A2A ──────────────────────────────────


async def section_c() -> None:
    _hr("Section C — Two-agent pipeline over A2A: researcher → writer")

    research_answer = (
        "Python f-strings were introduced in Python 3.6 (PEP 498). "
        "They allow embedding expressions inside string literals using curly braces."
    )
    written_article = (
        "# Python f-strings\n\n"
        "Introduced in Python 3.6, f-strings let you embed expressions directly\n"
        "inside string literals. Prefix the string with 'f' and put any Python\n"
        "expression inside curly braces: f'Hello, {name}!'"
    )

    researcher = _mock_agent("researcher", research_answer)
    writer = _mock_agent("writer", written_article)

    async with (
        A2AServer(researcher, name="researcher", description="Finds facts.", port=0) as r_srv,
        A2AServer(writer, name="writer", description="Writes articles.", port=0) as w_srv,
    ):
        r_url = f"http://127.0.0.1:{r_srv.actual_port}"
        w_url = f"http://127.0.0.1:{w_srv.actual_port}"

        # Discover both agents via their cards
        r_card = await A2AClient(r_url).get_card()
        w_card = await A2AClient(w_url).get_card()
        print(f"\n  Discovered: {r_card.name} @ {r_card.url}")
        print(f"  Discovered: {w_card.name} @ {w_card.url}")

        # Pipeline: research → write
        research_task = await A2AClient(r_url).send_task("What are Python f-strings?")
        assert research_task.status == TaskStatus.completed
        print(f"\n  [researcher] status={research_task.status.value}")
        print(f"    {research_task.output_text[:80]}…")

        write_task = await A2AClient(w_url).send_task(
            f"Write a short article based on this research:\n{research_task.output_text}"
        )
        assert write_task.status == TaskStatus.completed
        print(f"\n  [writer] status={write_task.status.value}")

    print(f"\n  === Final article ===\n{write_task.output_text}")
    print(
        "\n  Each agent ran in its own process boundary (separate HTTP server).\n"
        "  In a real deployment these could be on different machines or\n"
        "  written in completely different languages."
    )


# ── Section D — Error propagation ────────────────────────────────────────────


async def section_d() -> None:
    _hr("Section D — Error propagation: remote agent raises")

    # Agent that always fails
    failing_result = AgentResult(
        output="",
        context=ExecutionContext(),
        error="search service unavailable",
    )
    failing_agent = MagicMock(spec=Agent)
    failing_agent.name = "broken_researcher"
    failing_agent.run = AsyncMock(return_value=failing_result)

    async with A2AServer(failing_agent, name="broken_researcher", port=0) as server:
        url = f"http://127.0.0.1:{server.actual_port}"
        client = A2AClient(url)

        # Direct client call
        task = await client.send_task("What is the capital of France?")
        print("\n  Direct client call:")
        print(f"    task.status  = {task.status.value}")
        print(f"    task.error   = {task.error!r}")

        # Via A2ATool
        tool = A2ATool(client, name="research", description="Research facts")
        tool_result = await tool.execute(ExecutionContext(), task="What is the capital of France?")
        print("\n  A2ATool.execute() returned:")
        print(f"    {tool_result!r}")

    print(
        "\n  A2ATool surfaces remote errors as plain strings — the same format\n"
        "  as local tool errors.  The orchestrator's error-handling logic\n"
        "  works identically regardless of whether the tool is local or remote."
    )


# ── main ──────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Block 22 A2A experiments")
    parser.add_argument(
        "--section", default="a",
        choices=["a", "b", "c", "d", "all"],
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
