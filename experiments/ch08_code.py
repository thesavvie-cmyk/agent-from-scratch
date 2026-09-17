"""Chapter 8 code execution experiments (block 17).

Sections
--------
a) 100th Fibonacci number -- control example. Without code: model guesses
   a 21-digit number and almost always gets it wrong. With code: exact.
   Correct answer: 354224848179261915075
b) State persistence: two sequential tasks in one run() where the second
   uses a variable from the first.
c) Error recovery: model writes code with a bug, sees traceback, fixes it.
d) 46 presidential birthplaces -- the task that defeated compaction (block 12),
   planning (block 15), and reflection (block 16). Three configs:
   search-only, search+code, code-only.
e) Overhead: "1234 * 5678" with code -- measure sandbox creation time and
   extra tokens vs bare agent.

Usage
-----
    uv run python experiments/ch08_code.py --section a
    uv run python experiments/ch08_code.py --section all

Requires ANTHROPIC_API_KEY + E2B_API_KEY (all sections).
TAVILY_API_KEY required for sections d (search configs).
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from typing import Any

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

from agentkit.agent import Agent
from agentkit.config import FAST_MODEL, find_uv
from agentkit.llm import LlmClient
from agentkit.mcp_client import McpToolset
from agentkit.tools.mcp import load_mcp_tools

MCP_CMD = (find_uv(), ["run", "python", "-m", "agentkit.servers.tavily_server"])
FIBONACCI_100 = 354224848179261915075


def _hr(title: str) -> None:
    print(f"\n{'=' * 68}\n  {title}\n{'=' * 68}")


def _token_summary(ctx: Any) -> str:
    u = ctx.state.get("token_usage", {})
    return f"in={u.get('input_tokens', 0)}  out={u.get('output_tokens', 0)}"


def _print_trace(ctx: Any, max_items: int = 60) -> None:
    from agentkit.types import Message, ToolCall, ToolResult
    printed = 0
    for i, event in enumerate(ctx.events):
        for item in event.content:
            if printed >= max_items:
                print("    ... (truncated)")
                return
            if isinstance(item, ToolCall):
                args = str(item.arguments)[:100]
                print(f"    [{i}] CALL   {item.name}({args})")
                printed += 1
            elif isinstance(item, ToolResult):
                txt = str(item.content[0])[:100] if item.content else ""
                print(f"    [{i}] RESULT {item.name} [{item.status}] {txt}")
                printed += 1
            elif isinstance(item, Message) and item.role == "assistant":
                print(f"    [{i}] THINK  {item.content[:140]}")
                printed += 1


# ── Section A — Fibonacci 100 ──────────────────────────────────────────────────


async def section_a() -> None:
    _hr("Section A -- 100th Fibonacci number (correct: 354224848179261915075)")

    question = (
        "What is the 100th Fibonacci number? "
        "(F(1)=1, F(2)=1, F(3)=2, ...) Give only the number."
    )

    for label, use_code in [("without code", False), ("with code", True)]:
        agent = Agent(
            model=LlmClient(FAST_MODEL),
            instructions="Answer arithmetic questions precisely.",
            max_steps=6,
            code_execution="e2b" if use_code else None,
        )
        t0 = time.perf_counter()
        result = await agent.run(question)
        elapsed = time.perf_counter() - t0

        answer = str(result.output).strip()
        correct = str(FIBONACCI_100) in answer.replace(",", "").replace(" ", "")
        print(
            f"\n  [{label:<14}]  steps={result.context.current_step}"
            f"  {_token_summary(result.context)}  ({elapsed:.1f}s)"
        )
        print(f"    Answer: {answer[:120]}")
        print(f"    Correct (contains {FIBONACCI_100}): {'YES' if correct else 'NO'}")


# ── Section B — state persistence ─────────────────────────────────────────────


async def section_b() -> None:
    _hr("Section B -- State persistence: two tasks, second uses first's variable")

    question = (
        "Step 1: compute the sum of squares of integers from 1 to 50 and "
        "store it in a variable. "
        "Step 2: compute the square root of that sum. "
        "Show both results."
    )
    # Expected: sum of squares = 42925, sqrt ≈ 207.19

    agent = Agent(
        model=LlmClient(FAST_MODEL),
        instructions=(
            "Use execute_python for all calculations. "
            "Variables assigned in one call persist in the next."
        ),
        max_steps=8,
        code_execution="e2b",
    )
    result = await agent.run(question)
    print(f"\n  Steps: {result.context.current_step}  {_token_summary(result.context)}")
    print(f"  Answer: {str(result.output)[:300]}")
    print("\n  Trace:")
    _print_trace(result.context)


# ── Section C — error recovery ────────────────────────────────────────────────


async def section_c() -> None:
    _hr("Section C -- Error recovery: model writes buggy code, sees traceback, fixes")

    question = (
        "Parse '1, 2, three, 4, 5' and return the sum of numeric values. "
        "Start by running: total = sum(int(x.strip()) for x in '1, 2, three, 4, 5'.split(','))\n"
        "If that fails, fix the code and run it again. Return the final sum."
    )
    # Forces int("three") → ValueError on first call; model must fix and retry

    agent = Agent(
        model=LlmClient(FAST_MODEL),
        instructions=(
            "Use execute_python for date parsing. "
            "If you get a traceback, read it carefully and fix the code."
        ),
        max_steps=8,
        code_execution="e2b",
    )
    result = await agent.run(question)

    from agentkit.types import ToolResult
    errors = [
        item
        for e in result.context.events
        for item in e.content
        if isinstance(item, ToolResult) and item.name == "execute_python"
        and item.status == "success"
        and '"error":' in str(item.content[0])
        and '"type":' in str(item.content[0])
    ]

    print(f"\n  Steps: {result.context.current_step}  {_token_summary(result.context)}")
    print(f"  Code errors encountered: {len(errors)}")
    print(f"  Final answer: {str(result.output)[:200]}")
    print("\n  Full trace:")
    _print_trace(result.context)


# ── Section D — 46 presidential birthplaces ───────────────────────────────────


_PRESIDENTS_QUESTION = (
    "List the birthplace (city and state) for all 46 US presidents "
    "from George Washington (#1) to Joe Biden (#46), in order. "
    "Number each entry."
)

_CODE_ONLY_INSTRUCTIONS = (
    "You have Python code execution. Use it to build and format the complete list. "
    "The birthplaces are part of your training knowledge -- write Python code "
    "that constructs a dict of all 46 presidents and their birthplaces, "
    "then prints them in order."
)


async def _run_presidents(
    label: str,
    tools: list[Any] | None,
    use_code: bool,
    instructions: str,
    max_steps: int = 14,
) -> dict[str, Any]:
    agent = Agent(
        model=LlmClient(FAST_MODEL),
        tools=tools or [],
        instructions=instructions,
        max_steps=max_steps,
        code_execution="e2b" if use_code else None,
    )
    t0 = time.perf_counter()
    result = await agent.run(_PRESIDENTS_QUESTION)
    elapsed = time.perf_counter() - t0
    u = result.context.state.get("token_usage", {})

    output = str(result.output)
    # Count numbered lines
    numbered = [
        line for line in output.split("\n")
        if line.strip() and line.strip()[0].isdigit()
    ]
    return {
        "label": label,
        "steps": result.context.current_step,
        "input_tokens": u.get("input_tokens", 0),
        "elapsed": elapsed,
        "output": output,
        "numbered_entries": len(numbered),
        "context": result.context,
        "error": result.error,
    }


async def section_d() -> None:
    _hr("Section D -- 46 presidential birthplaces (search-only / search+code / code-only)")

    results = []

    # Config 1: search-only (baseline from block 12)
    async with McpToolset(*MCP_CMD) as ts:
        search_tools = list(load_mcp_tools(ts))
        print("\n  Running [search-only]...")
        r = await _run_presidents(
            "search-only",
            tools=search_tools,
            use_code=False,
            instructions="Search systematically. Number each entry explicitly.",
        )
        results.append(r)

        # Config 2: search + code
        print("  Running [search+code]...")
        r = await _run_presidents(
            "search+code",
            tools=search_tools,
            use_code=True,
            instructions=(
                "Search for the information, then use execute_python to build "
                "a structured list and format the output precisely."
            ),
        )
        results.append(r)

    # Config 3: code-only (no search tools)
    print("  Running [code-only]...")
    r = await _run_presidents(
        "code-only",
        tools=None,
        use_code=True,
        instructions=_CODE_ONLY_INSTRUCTIONS,
    )
    results.append(r)

    # Summary table
    print(f"\n  {'Config':<14} {'Steps':>6} {'Entries/46':>11} {'InTok':>7} {'Time':>6}")
    print(f"  {'-'*14} {'-'*6} {'-'*11} {'-'*7} {'-'*6}")
    for r in results:
        print(
            f"  {r['label']:<14} {r['steps']:>6} "
            f"{r['numbered_entries']:>7}/46    "
            f"{r['input_tokens']:>7} {r['elapsed']:>5.1f}s"
        )

    # Full traces
    for r in results:
        print(f"\n  {'─'*60}")
        print(f"  Trace [{r['label']}]:")
        _print_trace(r["context"])
        print(f"\n  Output [{r['label']}] (first 600 chars):")
        print(f"  {r['output'][:600]}")


# ── Section E — overhead ───────────────────────────────────────────────────────


async def section_e() -> None:
    _hr("Section E -- Overhead: '1234 * 5678' with and without code execution")

    import e2b_code_interpreter as e2b

    from agentkit.config import E2B_TIMEOUT

    question = "What is 1234 multiplied by 5678?"

    # Measure sandbox creation time separately
    t_create = time.perf_counter()
    sandbox = await e2b.AsyncSandbox.create(timeout=E2B_TIMEOUT)
    sandbox_create_ms = (time.perf_counter() - t_create) * 1000
    await sandbox.kill()
    print(f"\n  Sandbox creation time: {sandbox_create_ms:.0f} ms")

    for label, use_code in [("bare", False), ("with code", True)]:
        agent = Agent(
            model=LlmClient(FAST_MODEL),
            instructions="Answer arithmetic questions.",
            max_steps=6,
            code_execution="e2b" if use_code else None,
        )
        t0 = time.perf_counter()
        result = await agent.run(question)
        elapsed = time.perf_counter() - t0
        u = result.context.state.get("token_usage", {})

        from agentkit.types import ToolCall
        code_calls = sum(
            1 for e in result.context.events
            for item in e.content
            if isinstance(item, ToolCall) and item.name == "execute_python"
        )
        print(
            f"\n  [{label:<10}]  steps={result.context.current_step}"
            f"  in={u.get('input_tokens', 0)}  out={u.get('output_tokens', 0)}"
            f"  code_calls={code_calls}  ({elapsed:.1f}s)"
        )
        print(f"    Answer: {str(result.output)[:120]}")

    print(f"\n  Overhead = sandbox creation ({sandbox_create_ms:.0f} ms) + extra tokens")
    print("  For a task that needs real computation, this overhead is justified.")
    print("  For trivial arithmetic, bare agent is faster.")


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Block 17 code execution experiments")
    parser.add_argument(
        "--section", default="all",
        choices=["a", "b", "c", "d", "e", "all"],
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
        if args.section in ("e", "all"):
            await section_e()

    try:
        asyncio.run(_run_all())
    except KeyboardInterrupt:
        print("\n[interrupted]")
        sys.exit(1)

    print()


if __name__ == "__main__":
    main()
