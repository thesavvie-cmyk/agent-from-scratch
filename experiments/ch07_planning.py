"""Chapter 7 planning experiments (block 15).

Sections
--------
a) Kipchoge pace — planning vs bare agent (steps comparison)
b) Nobel Prize laureates — 3 configs: no plan / think_first / full planning
c) 46 presidential birthplaces — block-12 failure case with planning
d) "1234 * 5678" — overhead measurement for a trivial task

Usage
-----
    uv run python experiments/ch07_planning.py --section a
    uv run python experiments/ch07_planning.py --section all

Requires ANTHROPIC_API_KEY + TAVILY_API_KEY (sections b, c).
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path
from typing import Any

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

from agentkit.agent import Agent
from agentkit.config import FAST_MODEL, find_uv
from agentkit.llm import LlmClient
from agentkit.mcp_client import McpToolset
from agentkit.planning import Plan
from agentkit.tools.base import FunctionTool
from agentkit.tools.mcp import load_mcp_tools
from agentkit.tools_manual import calculator

RESULTS_DIR = Path(__file__).parent.parent / "results"
MCP_CMD = (find_uv(), ["run", "python", "-m", "agentkit.servers.tavily_server"])


def _hr(title: str) -> None:
    print(f"\n{'─' * 60}\n  {title}\n{'─' * 60}")


def _calc_tool() -> FunctionTool:
    return FunctionTool(calculator)


def _print_plan(ctx: Any) -> None:
    plan_data = ctx.state.get("plan")
    if plan_data:
        plan = Plan.model_validate(plan_data)
        print(f"  Plan: {plan.progress()}")


def _token_summary(ctx: Any) -> str:
    u = ctx.state.get("token_usage", {})
    return f"in={u.get('input_tokens', 0)}  out={u.get('output_tokens', 0)}"


# ── Section A — Kipchoge pace with planning ────────────────────────────────────


async def section_a() -> None:
    _hr("Section A — Kipchoge pace: planning vs no planning")

    question = (
        "Eliud Kipchoge ran the Berlin Marathon (42.195 km) in 2:01:09. "
        "Calculate his average speed in km/h and his pace in min/km. "
        "Show the calculations step by step."
    )

    for label, use_planning in [("no planning", False), ("with planning", True)]:
        print(f"\n  [{label}]")
        agent = Agent(
            model=LlmClient(FAST_MODEL),
            tools=[_calc_tool()],
            instructions="Use the calculator for arithmetic. Show your work.",
            max_steps=14,
            planning=use_planning,
        )
        t0 = time.perf_counter()
        result = await agent.run(question)
        elapsed = time.perf_counter() - t0

        steps = result.context.current_step
        tokens = _token_summary(result.context)
        print(f"  Steps: {steps}  {tokens}  ({elapsed:.1f}s)")
        _print_plan(result.context)
        print(f"  Answer: {str(result.output)[:200]}")

    print("\n  Block-5 reference (no planning, haiku): 2-4 steps")


# ── Section B — Nobel Prize laureates ─────────────────────────────────────────


_NOBEL_QUESTION = (
    "Find the Nobel Prize laureates in Physics for 2024, 2023, and 2022. "
    "For each year: winner name(s) and their birth country. "
    "Then calculate the average age at the time of the award for all individual "
    "laureates (use birth year only for the age estimate)."
)


async def _run_nobel(
    label: str,
    tools: list[Any],
    planning: bool,
    think_first: bool,
) -> dict[str, Any]:
    agent = Agent(
        model=LlmClient(FAST_MODEL),
        tools=tools,
        instructions="You are a research assistant. Use search tools to find facts.",
        max_steps=14,
        planning=planning,
        think_first=think_first,
    )
    t0 = time.perf_counter()
    result = await agent.run(_NOBEL_QUESTION)
    elapsed = time.perf_counter() - t0
    u = result.context.state.get("token_usage", {})
    return {
        "label": label,
        "steps": result.context.current_step,
        "input_tokens": u.get("input_tokens", 0),
        "output_tokens": u.get("output_tokens", 0),
        "elapsed": elapsed,
        "output": str(result.output),
        "context": result.context,
        "error": result.error,
    }


async def section_b() -> None:
    _hr("Section B — Nobel Physics laureates: 3 configurations")

    async with McpToolset(*MCP_CMD) as ts:
        search_tools = load_mcp_tools(ts)

        configs = [
            ("no planning",    False, False),
            ("think_first",    False, True),
            ("full planning",  True,  False),
        ]

        rows = []
        for label, planning, think_first in configs:
            print(f"\n  Running [{label}]...")
            row = await _run_nobel(
                label, list(search_tools), planning, think_first
            )
            rows.append(row)
            _print_plan(row["context"])
            if row["error"]:
                print(f"  ERROR: {row['error']}")

    # Table
    print(f"\n  {'Config':<16} {'Steps':>6} {'In-tok':>8} {'Out-tok':>8} {'Time':>6}")
    print(f"  {'─'*16} {'─'*6} {'─'*8} {'─'*8} {'─'*6}")
    for r in rows:
        print(
            f"  {r['label']:<16} {r['steps']:>6} "
            f"{r['input_tokens']:>8} {r['output_tokens']:>8} "
            f"{r['elapsed']:>5.1f}s"
        )

    # Check whether correct laureates appear in any output
    laureates_2024 = ["hinton", "hopfield"]
    laureates_2022_23 = ["aspect", "clauser", "zeilinger"]

    print("\n  Correctness check (laureate names in output):")
    for r in rows:
        out = r["output"].lower()
        found_2024 = any(n in out for n in laureates_2024)
        found_2023 = any(n in out for n in laureates_2022_23)
        avg_age = "age" in out or "average" in out
        print(
            f"  [{r['label']:<14}] "
            f"2024={'ok' if found_2024 else '--'}  "
            f"2022/23={'ok' if found_2023 else '--'}  "
            f"avg_age={'ok' if avg_age else '--'}"
        )

    print("\n  Full answers:")
    for r in rows:
        print(f"\n  [{r['label']}]")
        print(f"  {r['output'][:400]}")


# ── Section C — 46 presidential birthplaces ───────────────────────────────────


_PRESIDENTS_QUESTION = (
    "List the birthplace (city and state) for all 46 US presidents, "
    "from George Washington (#1) to Joe Biden (#46), in order."
)


async def section_c() -> None:
    _hr("Section C — 46 presidential birthplaces (block-12 failure case)")

    print("\n  Block-12 finding: haiku collected partial data across 8 web searches")
    print("  but couldn't synthesise all 46 from snippet-level results.")
    print("  Question: does a plan help structure the search systematically?\n")

    async with McpToolset(*MCP_CMD) as ts:
        search_tools = load_mcp_tools(ts)

        agent = Agent(
            model=LlmClient(FAST_MODEL),
            tools=list(search_tools),
            instructions=(
                "You are a research assistant. Search systematically. "
                "When listing items, number them explicitly."
            ),
            max_steps=16,
            planning=True,
        )
        t0 = time.perf_counter()
        result = await agent.run(_PRESIDENTS_QUESTION)
        elapsed = time.perf_counter() - t0

    u = result.context.state.get("token_usage", {})
    print(f"  Steps: {result.context.current_step}  "
          f"in={u.get('input_tokens', 0)}  out={u.get('output_tokens', 0)}  "
          f"({elapsed:.1f}s)")

    plan_data = result.context.state.get("plan")
    if plan_data:
        plan = Plan.model_validate(plan_data)
        print(f"\n  Plan final state: {plan.progress()}")
        print(f"  Complete: {plan.is_complete()}")

    # Count how many numbered entries appear
    output = str(result.output)
    numbered = [line for line in output.split("\n") if line.strip()[:2].rstrip(".").isdigit()]
    print(f"\n  Numbered entries in output: {len(numbered)}")
    print("  (target: 46)")

    print(f"\n  Output (first 800 chars):\n  {output[:800]}")

    # Print trace summary
    print(f"\n  Trace ({len(result.context.events)} events):")
    for i, evt in enumerate(result.context.events):
        for item in evt.content:
            from agentkit.types import Message, ToolCall, ToolResult
            if isinstance(item, ToolCall):
                args_preview = str(item.arguments)[:60]
                print(f"    [{i}] tool_call: {item.name}({args_preview})")
            elif isinstance(item, ToolResult):
                print(f"    [{i}] tool_result: {item.name} -> {str(item.content[0])[:60]}")
            elif isinstance(item, Message) and item.role == "assistant":
                print(f"    [{i}] assistant: {item.content[:60]}")


# ── Section D — trivial task overhead ─────────────────────────────────────────


async def section_d() -> None:
    _hr("Section D — overhead: '1234 * 5678' with and without planning")

    question = "What is 1234 multiplied by 5678?"

    for label, use_planning, use_think in [
        ("bare",           False, False),
        ("think_first",    False, True),
        ("full planning",  True,  False),
    ]:
        agent = Agent(
            model=LlmClient(FAST_MODEL),
            tools=[_calc_tool()],
            instructions="Use the calculator.",
            max_steps=8,
            planning=use_planning,
            think_first=use_think,
        )
        result = await agent.run(question)
        u = result.context.state.get("token_usage", {})
        in_tok = u.get("input_tokens", 0)
        out_tok = u.get("output_tokens", 0)
        steps = result.context.current_step
        plan_data = result.context.state.get("plan")
        plan_str = ""
        if plan_data:
            plan = Plan.model_validate(plan_data)
            plan_str = f"  [{plan.progress()}]"
        print(
            f"  [{label:<14}]  steps={steps}  "
            f"in={in_tok}  out={out_tok}{plan_str}"
        )

    print("\n  Overhead = extra steps/tokens from plan creation vs bare agent.")


# ── main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Block 15 planning experiments")
    parser.add_argument(
        "--section", default="all",
        choices=["a", "b", "c", "d", "all"],
    )
    args = parser.parse_args()

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    async def _run_all() -> None:
        if args.section in ("a", "all"):
            await section_a()
        if args.section in ("b", "all"):
            await section_b()
        if args.section in ("c", "all"):
            await section_c()
        if args.section in ("d", "all"):
            await section_d()

    try:
        asyncio.run(_run_all())
    except KeyboardInterrupt:
        print("\n[interrupted]")
        sys.exit(1)

    print()


if __name__ == "__main__":
    main()
