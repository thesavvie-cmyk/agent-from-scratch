"""Chapter 7 reflection experiments (block 16).

Sections
--------
a) Broken tool recovery: get_wikipedia_page always raises RuntimeError.
   Instruction says "always try Wikipedia first". Compare without/with
   reflection. Key metric: how many times does the agent repeat the failing
   call before switching to search_web?
b) Synthesis: "Research recent developments in quantum computing" --
   without vs with reflection. Compare search count and answer quality.
c) Contradictory data: Moon distance -- search returns both mean (~384 400 km)
   and perigee (~362 600 km). Does reflection help the agent choose correctly?
d) SELF CHECK: Nobel Physics 2019-2021 laureates. Without reflection the agent
   may answer after finding only part of the data. With reflection it should
   verify completeness before calling final_answer.
e) Overhead: "1234 * 5678" with reflection -- how many extra steps/tokens
   compared to the bare agent?

Usage
-----
    uv run python experiments/ch07_reflection.py --section a
    uv run python experiments/ch07_reflection.py --section all

Requires ANTHROPIC_API_KEY + TAVILY_API_KEY (sections a-d).
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
from agentkit.tools.base import FunctionTool, tool
from agentkit.tools.mcp import load_mcp_tools
from agentkit.tools_manual import calculator

MCP_CMD = (find_uv(), ["run", "python", "-m", "agentkit.servers.tavily_server"])


def _hr(title: str) -> None:
    print(f"\n{'=' * 68}\n  {title}\n{'=' * 68}")


def _token_summary(ctx: Any) -> str:
    u = ctx.state.get("token_usage", {})
    return f"in={u.get('input_tokens', 0)}  out={u.get('output_tokens', 0)}"


def _count_repeated_tool_calls(ctx: Any) -> dict[str, int]:
    """Return {tool_name: extra_repeat_count} for calls repeated >1 time."""
    from agentkit.types import ToolCall
    seen: dict[str, int] = {}
    for event in ctx.events:
        for item in event.content:
            if isinstance(item, ToolCall):
                key = f"{item.name}:{sorted(item.arguments.items())!s}"
                seen[key] = seen.get(key, 0) + 1
    return {k: v - 1 for k, v in seen.items() if v > 1}


def _print_trace(ctx: Any, max_items: int = 40) -> None:
    """Print a compact trace: tool_call/tool_result/assistant lines."""
    from agentkit.types import Message, ToolCall, ToolResult
    printed = 0
    for i, event in enumerate(ctx.events):
        for item in event.content:
            if printed >= max_items:
                print(f"    ... (truncated at {max_items} items)")
                return
            if isinstance(item, ToolCall):
                args = str(item.arguments)[:80]
                print(f"    [{i}] CALL  {item.name}({args})")
                printed += 1
            elif isinstance(item, ToolResult):
                result_text = str(item.content[0])[:80] if item.content else ""
                status = item.status
                print(f"    [{i}] RESULT {item.name} [{status}] {result_text}")
                printed += 1
            elif isinstance(item, Message) and item.role == "assistant":
                print(f"    [{i}] THINK  {item.content[:120]}")
                printed += 1


# ── Section A — broken tool recovery ──────────────────────────────────────────


@tool
def get_wikipedia_page(title: str) -> str:
    """Fetch a Wikipedia article by title and return its text content."""
    msg = f"RuntimeError: Wikipedia service unavailable (title={title!r})"
    raise RuntimeError(msg)


async def _run_broken_wiki(
    label: str,
    search_tools: list[Any],
    use_reflection: bool,
) -> dict[str, Any]:
    tools = [get_wikipedia_page, *search_tools]
    agent = Agent(
        model=LlmClient(FAST_MODEL),
        tools=tools,
        instructions=(
            "STRICT TOOL POLICY: your FIRST tool call on every question "
            "MUST be get_wikipedia_page. No exceptions. Only call other "
            "tools after get_wikipedia_page has been attempted."
        ),
        max_steps=8,
        reflection=use_reflection,
    )
    t0 = time.perf_counter()
    result = await agent.run(
        "Use get_wikipedia_page to look up who won the 2025 Formula 1 "
        "World Championship. Give the driver's name and their constructor."
    )
    elapsed = time.perf_counter() - t0
    u = result.context.state.get("token_usage", {})
    repeated = _count_repeated_tool_calls(result.context)
    wiki_repeats = sum(v for k, v in repeated.items() if "wikipedia" in k.lower())
    return {
        "label": label,
        "steps": result.context.current_step,
        "input_tokens": u.get("input_tokens", 0),
        "output_tokens": u.get("output_tokens", 0),
        "elapsed": elapsed,
        "output": str(result.output),
        "context": result.context,
        "error": result.error,
        "wiki_repeats": wiki_repeats,
        "all_repeats": repeated,
    }


async def section_a() -> None:
    _hr("Section A -- Broken tool recovery (Wikipedia always raises)")

    async with McpToolset(*MCP_CMD) as ts:
        search_tools = list(load_mcp_tools(ts))
        rows = []
        for label, use_reflection in [
            ("without reflection", False),
            ("with reflection",    True),
        ]:
            print(f"\n  Running [{label}]...")
            row = await _run_broken_wiki(label, search_tools, use_reflection)
            rows.append(row)

    print()
    for row in rows:
        print(f"  [{row['label']}]")
        print(f"    Steps: {row['steps']}  {_token_summary(row['context'])}  ({row['elapsed']:.1f}s)")
        print(f"    Wikipedia repeated calls (extra): {row['wiki_repeats']}")
        if row["all_repeats"]:
            for k, v in row["all_repeats"].items():
                print(f"      repeated: {k} x{v+1}")
        print(f"    Answer: {row['output'][:200]}")
        print(f"\n  Full trace [{row['label']}]:")
        _print_trace(row["context"])
        print()

    print("  Summary: extra wiki calls (fewer=better with reflection)")
    print(f"  {'Config':<22} {'Steps':>6} {'WikiRepeats':>12} {'Switched to web':>16}")
    for row in rows:
        from agentkit.types import ToolCall
        web_calls = sum(
            1 for e in row["context"].events
            for item in e.content
            if isinstance(item, ToolCall) and "search" in item.name.lower()
        )
        print(
            f"  {row['label']:<22} {row['steps']:>6} "
            f"{row['wiki_repeats']:>12} {web_calls:>16}"
        )


# ── Section B — synthesis: quantum computing ──────────────────────────────────


async def section_b() -> None:
    _hr("Section B -- Synthesis: quantum computing developments")

    question = (
        "Research recent developments in quantum computing. "
        "Summarise the most significant advances from the past 2 years."
    )

    async with McpToolset(*MCP_CMD) as ts:
        search_tools = list(load_mcp_tools(ts))
        rows = []
        for label, use_reflection in [
            ("without reflection", False),
            ("with reflection",    True),
        ]:
            print(f"\n  Running [{label}]...")
            agent = Agent(
                model=LlmClient(FAST_MODEL),
                tools=list(search_tools),
                instructions="You are a research assistant. Search and synthesise facts carefully.",
                max_steps=10,
                reflection=use_reflection,
            )
            t0 = time.perf_counter()
            result = await agent.run(question)
            elapsed = time.perf_counter() - t0
            u = result.context.state.get("token_usage", {})

            from agentkit.types import ToolCall
            search_calls = sum(
                1 for e in result.context.events
                for item in e.content
                if isinstance(item, ToolCall) and "search" in item.name.lower()
            )
            rows.append({
                "label": label,
                "steps": result.context.current_step,
                "input_tokens": u.get("input_tokens", 0),
                "output_tokens": u.get("output_tokens", 0),
                "elapsed": elapsed,
                "output": str(result.output),
                "search_calls": search_calls,
                "repeats": _count_repeated_tool_calls(result.context),
            })

    print(f"\n  {'Config':<22} {'Steps':>6} {'Searches':>9} {'Repeats':>8} {'InTok':>7}")
    for r in rows:
        total_repeats = sum(r["repeats"].values())
        print(
            f"  {r['label']:<22} {r['steps']:>6} "
            f"{r['search_calls']:>9} {total_repeats:>8} {r['input_tokens']:>7}"
        )

    print("\n  Answers (first 400 chars):")
    for r in rows:
        print(f"\n  [{r['label']}]")
        print(f"  {r['output'][:400]}")


# ── Section C — contradictory data: Moon distance ─────────────────────────────


async def section_c() -> None:
    _hr("Section C -- Contradictory data: distance to the Moon")

    question = (
        "What is the distance from Earth to the Moon in kilometres? "
        "Give a precise number."
    )

    async with McpToolset(*MCP_CMD) as ts:
        search_tools = list(load_mcp_tools(ts))
        rows = []
        for label, use_reflection in [
            ("without reflection", False),
            ("with reflection",    True),
        ]:
            print(f"\n  Running [{label}]...")
            agent = Agent(
                model=LlmClient(FAST_MODEL),
                tools=list(search_tools),
                instructions="Answer precisely. If sources give different numbers, reconcile them.",
                max_steps=6,
                reflection=use_reflection,
            )
            t0 = time.perf_counter()
            result = await agent.run(question)
            elapsed = time.perf_counter() - t0
            u = result.context.state.get("token_usage", {})
            rows.append({
                "label": label,
                "steps": result.context.current_step,
                "input_tokens": u.get("input_tokens", 0),
                "elapsed": elapsed,
                "output": str(result.output),
                "repeats": _count_repeated_tool_calls(result.context),
                "context": result.context,
            })

    # Known correct values:
    # mean: ~384,400 km  perigee: ~362,600 km  apogee: ~405,500 km
    print()
    print("  Known values: mean ~384,400 km | perigee ~362,600 km | apogee ~405,500 km")
    for r in rows:
        out = r["output"]
        has_mean = any(v in out for v in ["384", "385"])
        has_perigee = "362" in out
        mentioned_both = has_mean and has_perigee
        print(
            f"\n  [{r['label']}]  steps={r['steps']}  in={r['input_tokens']}"
            f"  ({r['elapsed']:.1f}s)"
        )
        print(f"    Answer: {out[:300]}")
        print(
            f"    mean in answer: {'yes' if has_mean else 'no'}  "
            f"perigee in answer: {'yes' if has_perigee else 'no'}  "
            f"both mentioned: {'yes' if mentioned_both else 'no'}"
        )


# ── Section D — SELF CHECK: Nobel laureates ───────────────────────────────────


async def section_d() -> None:
    _hr("Section D -- SELF CHECK: Nobel Physics 2019-2021 (completeness check)")

    question = (
        "List all Nobel Prize in Physics laureates for 2019, 2020, and 2021. "
        "For each year list every winner by name."
    )
    # Expected: 2019: Peebles, Mayor, Queloz (3)
    #           2020: Penrose, Genzel, Ghez (3)
    #           2021: Syukuro Manabe, Klaus Hasselmann, Giorgio Parisi (3)

    async with McpToolset(*MCP_CMD) as ts:
        search_tools = list(load_mcp_tools(ts))
        rows = []
        for label, use_reflection in [
            ("without reflection", False),
            ("with reflection",    True),
        ]:
            print(f"\n  Running [{label}]...")
            agent = Agent(
                model=LlmClient(FAST_MODEL),
                tools=list(search_tools),
                instructions=(
                    "You are a research assistant. "
                    "Before answering, make sure you have found ALL winners for EACH year."
                ),
                max_steps=10,
                reflection=use_reflection,
            )
            t0 = time.perf_counter()
            result = await agent.run(question)
            elapsed = time.perf_counter() - t0
            u = result.context.state.get("token_usage", {})
            out = str(result.output).lower()

            # Check which names appear
            expected = {
                "peebles": "2019", "mayor": "2019", "queloz": "2019",
                "penrose": "2020", "genzel": "2020", "ghez": "2020",
                "manabe": "2021", "hasselmann": "2021", "parisi": "2021",
            }
            found = [name for name in expected if name in out]
            rows.append({
                "label": label,
                "steps": result.context.current_step,
                "input_tokens": u.get("input_tokens", 0),
                "elapsed": elapsed,
                "output": str(result.output),
                "found_names": found,
                "repeats": _count_repeated_tool_calls(result.context),
            })

    print("\n  Expected: 9 names across 3 years")
    print(f"  {'Config':<22} {'Steps':>6} {'Found/9':>8} {'Names found'}")
    for r in rows:
        print(
            f"  {r['label']:<22} {r['steps']:>6} "
            f"{len(r['found_names']):>6}/9  "
            f"{', '.join(r['found_names']) or 'none'}"
        )
    print("\n  Full answers:")
    for r in rows:
        print(f"\n  [{r['label']}]")
        print(f"  {r['output'][:500]}")


# ── Section E — overhead: trivial arithmetic ──────────────────────────────────


async def section_e() -> None:
    _hr("Section E -- Overhead: '1234 * 5678' with and without reflection")

    question = "What is 1234 multiplied by 5678?"
    calc = FunctionTool(calculator)

    for label, use_reflection in [
        ("bare",              False),
        ("with reflection",   True),
    ]:
        agent = Agent(
            model=LlmClient(FAST_MODEL),
            tools=[calc],
            instructions="Use the calculator.",
            max_steps=6,
            reflection=use_reflection,
        )
        result = await agent.run(question)
        u = result.context.state.get("token_usage", {})
        from agentkit.types import ToolCall
        reflection_calls = sum(
            1 for e in result.context.events
            for item in e.content
            if isinstance(item, ToolCall) and item.name == "reflection"
        )
        print(
            f"  [{label:<18}]  "
            f"steps={result.context.current_step}  "
            f"in={u.get('input_tokens', 0)}  "
            f"out={u.get('output_tokens', 0)}  "
            f"reflection_calls={reflection_calls}"
        )
        print(f"    Answer: {str(result.output)[:120]}")

    print("\n  Overhead = extra tokens from the reflection reminder instruction")
    print("  and any reflection() calls on a trivial task.")


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Block 16 reflection experiments")
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
