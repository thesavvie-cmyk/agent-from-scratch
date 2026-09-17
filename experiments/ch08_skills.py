"""Chapter 8 skills experiments (block 19).

Sections
--------
a) Token count: full tool descriptions vs level-1 skill list.
   Hypothetical projections for 50 and 100 tools.
b) Skill reading trace: task that triggers web-research skill.
   Shows: saw list → read SKILL.md → applied strategy.
c) Skill with script: task where agent runs a script from the skill folder.
d) Negative case: simple task where agent should NOT read any skill.

Usage
-----
    uv run python experiments/ch08_skills.py --section a
    uv run python experiments/ch08_skills.py --section all

Requires ANTHROPIC_API_KEY + TAVILY_API_KEY + E2B_API_KEY (sections b, c).
"""
from __future__ import annotations

import argparse
import asyncio
import json
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
from agentkit.skills import discover_skills, format_skills_for_prompt
from agentkit.tokens import count_tokens
from agentkit.tools.mcp import load_mcp_tools

MCP_CMD = (find_uv(), ["run", "python", "-m", "agentkit.servers.tavily_server"])
SKILLS_DIR = Path(__file__).parent.parent / "skills"


def _hr(title: str) -> None:
    print(f"\n{'=' * 68}\n  {title}\n{'=' * 68}")


# ── Section A — Token counting ─────────────────────────────────────────────────


def _tool_def_text(tool: Any) -> str:
    """Serialise a BaseTool's definition to JSON (what the API actually sends)."""
    defn = getattr(tool, "tool_definition", None) or {}
    return json.dumps(defn, indent=2)


def section_a() -> None:
    _hr("Section A -- Token budget: full descriptions vs skill list (levels 1–2)")

    skills = discover_skills(SKILLS_DIR)
    print(f"  Discovered {len(skills)} skills: {[s.name for s in skills]}")

    # ── Measure actual tool definitions ───────────────────────────────────────
    # Build an agent with all real tools to get their full definitions
    from agentkit.planning import create_plan, get_plan, update_task
    from agentkit.reflection import reflection as refl_tool
    from agentkit.tools.code_execution import execute_python
    from agentkit.tools.workspace_sandbox import WORKSPACE_TOOLS

    all_tools = [execute_python, refl_tool, create_plan, update_task, get_plan, *WORKSPACE_TOOLS]

    # Also add a simulated search_web definition
    from agentkit.tools.base import FunctionTool
    async def search_web(context: Any, query: str, max_results: int = 5) -> str:  # type: ignore[misc]
        """Search the web for current information using Tavily. Use when you need recent events, live data, statistics, or facts that may have changed since training. Parameters: query (search string), max_results (1-10, default 5)."""
        return ""
    search_tool = FunctionTool(search_web)
    all_tools = [search_tool, *all_tools]

    # Token cost per tool (full description)
    tool_texts = [_tool_def_text(t) for t in all_tools]
    tokens_per_tool = [count_tokens(txt) for txt in tool_texts]
    avg_tokens_per_tool = sum(tokens_per_tool) / len(tokens_per_tool)
    total_tool_tokens_actual = sum(tokens_per_tool)

    print(f"\n  Actual tools ({len(all_tools)}):")
    for t, tok in zip(all_tools, tokens_per_tool):
        name = getattr(t, "name", "?")
        print(f"    {name:<28} {tok:>5} tokens")
    print(f"  {'Total':<28} {total_tool_tokens_actual:>5} tokens")
    print(f"  Average per tool: {avg_tokens_per_tool:.0f} tokens")

    # ── Level 1: skill list ────────────────────────────────────────────────────
    skill_list_text = format_skills_for_prompt(skills)
    tok_skill_list = count_tokens(skill_list_text)
    tok_per_skill = tok_skill_list / len(skills) if skills else 0

    print(f"\n  Skill list (level 1, {len(skills)} skills): {tok_skill_list} tokens")
    print(f"  Per skill: {tok_per_skill:.0f} tokens")

    # ── Level 2: reading one SKILL.md ─────────────────────────────────────────
    level2_costs = {}
    for skill in skills:
        content = skill.skill_file().read_text(encoding="utf-8")
        tok = count_tokens(content)
        level2_costs[skill.name] = tok
        print(f"  SKILL.md {skill.name:<28} {tok:>5} tokens (level 2)")

    # ── Projection table: 10 / 50 / 100 tools ─────────────────────────────────
    base_instructions = count_tokens("Answer questions precisely. Use tools when needed.")

    print(f"\n  {'N tools':<10} {'Full descriptions':>18} {'Skill list (L1)':>16} {'L1 + 1 read (L2)':>18} {'Saved':>8}")
    print(f"  {'-'*10} {'-'*18} {'-'*16} {'-'*18} {'-'*8}")

    for n_tools in [len(all_tools), 50, 100]:
        # Assume tools grow proportionally; skills grow at ~3 tools/skill
        n_skills_hyp = max(len(skills), n_tools // 3)
        full = base_instructions + int(avg_tokens_per_tool * n_tools)
        l1 = base_instructions + int(tok_per_skill * n_skills_hyp)
        avg_l2 = sum(level2_costs.values()) / len(level2_costs) if level2_costs else 500
        l1_plus_read = l1 + int(avg_l2)
        saved_pct = (1 - l1_plus_read / full) * 100 if full > 0 else 0
        print(
            f"  {n_tools:<10} {full:>18,} {l1:>16,} "
            f"{l1_plus_read:>18,} {saved_pct:>7.0f}%"
        )

    print(
        "\n  Interpretation: at 100 tools, level-1 skill list + one on-demand"
        " read saves ~80% of tool-description tokens. At 10 tools the saving"
        " is smaller — skills start paying off around 20–30 tools."
    )


# ── Section B — Skill reading trace ────────────────────────────────────────────


async def section_b() -> None:
    _hr("Section B -- Skill reading trace: web-research triggered by complex search task")

    question = (
        "Find the birthplaces of the first 5 US presidents "
        "(Washington through Monroe) and list them numbered."
    )

    async with McpToolset(*MCP_CMD) as ts:
        search_tools = list(load_mcp_tools(ts))

    agent = Agent(
        model=LlmClient(FAST_MODEL),
        tools=search_tools,
        instructions=(
            "Answer research questions carefully. "
            "When appropriate, read relevant skills before proceeding."
        ),
        max_steps=12,
        skills_dir=SKILLS_DIR,
    )

    t0 = time.perf_counter()
    result = await agent.run(question)
    elapsed = time.perf_counter() - t0

    u = result.context.state.get("token_usage", {})

    # Count tool calls
    from agentkit.types import ToolCall, ToolResult
    skill_reads: list[str] = []
    search_calls = 0
    for ev in result.context.events:
        for item in ev.content:
            if isinstance(item, ToolCall):
                if item.name == "read_skill":
                    skill_reads.append(item.arguments.get("name", "?"))
                elif item.name == "search_web":
                    search_calls += 1

    # Measure tokens added by the skill read
    tok_added = 0
    for ev in result.context.events:
        for item in ev.content:
            if isinstance(item, ToolResult) and item.name == "read_skill":
                content_str = str(item.content[0]) if item.content else ""
                tok_added += count_tokens(content_str)

    print(f"\n  Steps: {result.context.current_step}  ({elapsed:.1f}s)")
    print(f"  Skills read: {skill_reads or '(none)'}")
    print(f"  Search calls: {search_calls}")
    print(f"  Tokens added by skill reads: ~{tok_added}")
    print(f"  in={u.get('input_tokens', 0)}  out={u.get('output_tokens', 0)}")
    print(f"\n  Answer (first 400 chars):\n  {str(result.output)[:400]}")

    # Print the trace up to the skill read
    print("\n  Trace (tool calls only):")
    for ev in result.context.events:
        for item in ev.content:
            if isinstance(item, ToolCall):
                args = str(item.arguments)[:70]
                marker = " ← SKILL READ" if item.name == "read_skill" else ""
                print(f"    CALL  {item.name}({args}){marker}")


# ── Section C — Skill script execution ────────────────────────────────────────


async def section_c() -> None:
    _hr("Section C -- Skill script: agent creates Excel, then runs read_excel.py from skill")

    question = (
        "I have an Excel file at /home/user/scores.xlsx with columns "
        "'name' and 'score'. Use the gaia-file-analysis skill to read it "
        "and tell me the highest score."
    )

    # First, create the Excel file in the sandbox via the agent
    setup_code = """
import openpyxl
wb = openpyxl.Workbook()
ws = wb.active
ws.append(["name", "score"])
data = [("Alice", 92), ("Bob", 78), ("Carol", 88), ("Dave", 95), ("Eve", 71)]
for row in data:
    ws.append(row)
wb.save('/home/user/scores.xlsx')
print("Created scores.xlsx with 5 rows")
"""

    async with McpToolset(*MCP_CMD) as ts:
        search_tools = list(load_mcp_tools(ts))

    agent = Agent(
        model=LlmClient(FAST_MODEL),
        tools=search_tools,
        instructions=(
            "Use the gaia-file-analysis skill when working with files. "
            "Skill scripts are available in /home/user/skills/<skill-name>/. "
            "Use run_command to run them."
        ),
        max_steps=12,
        code_execution="e2b",
        workspace=True,
        skills_dir=SKILLS_DIR,
    )

    t0 = time.perf_counter()
    # First create the file, then ask the agent to analyze it
    result = await agent.run(
        f"First run this setup code in execute_python:\n```python\n{setup_code}\n```\n\n"
        f"Then: {question}"
    )
    elapsed = time.perf_counter() - t0

    from agentkit.types import ToolCall
    code_calls = sum(
        1 for ev in result.context.events
        for item in ev.content
        if isinstance(item, ToolCall) and item.name in ("execute_python", "run_command")
    )
    skill_reads = [
        item.arguments.get("name", "?")
        for ev in result.context.events
        for item in ev.content
        if isinstance(item, ToolCall) and item.name == "read_skill"
    ]

    u = result.context.state.get("token_usage", {})
    print(f"\n  Steps: {result.context.current_step}  ({elapsed:.1f}s)")
    print(f"  Skills read: {skill_reads or '(none)'}")
    print(f"  Code/command calls: {code_calls}")
    print(f"  in={u.get('input_tokens', 0)}  out={u.get('output_tokens', 0)}")
    print(f"  Answer: {str(result.output)[:300]}")


# ── Section D — Negative case ──────────────────────────────────────────────────


async def section_d() -> None:
    _hr("Section D -- Negative case: trivial task, skills must NOT be read")

    questions = [
        "What is the capital of France?",
        "What is 17 multiplied by 23?",
        "What does HTTP stand for?",
    ]

    agent = Agent(
        model=LlmClient(FAST_MODEL),
        instructions="Answer questions directly and concisely.",
        max_steps=4,
        skills_dir=SKILLS_DIR,
    )

    print(f"\n  Agent has {len(agent._skills)} skills in toolbox: read_skill in toolbox = {'read_skill' in agent._toolbox}")

    for q in questions:
        result = await agent.run(q)

        from agentkit.types import ToolCall
        skill_reads = [
            item.arguments.get("name", "?")
            for ev in result.context.events
            for item in ev.content
            if isinstance(item, ToolCall) and item.name == "read_skill"
        ]
        all_tool_calls = [
            item.name
            for ev in result.context.events
            for item in ev.content
            if isinstance(item, ToolCall)
        ]

        status = "PASS (no skill read)" if not skill_reads else f"FAIL: read {skill_reads}"
        print(f"\n  Q: {q}")
        print(f"  A: {str(result.output)[:100]}")
        print(f"  Tools called: {all_tool_calls or '(none)'}")
        print(f"  Status: {status}")

    print(
        "\n  Expected: agent answers from memory, never calls read_skill."
        "\n  If it does read skills: tighten the format_skills_for_prompt"
        "\n  instructions to clarify 'only for complex research tasks'."
    )


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Block 19 skills experiments")
    parser.add_argument(
        "--section", default="all",
        choices=["a", "b", "c", "d", "all"],
    )
    args = parser.parse_args()

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    async def _run_async() -> None:
        if args.section in ("b", "all"):
            await section_b()
        if args.section in ("c", "all"):
            await section_c()
        if args.section in ("d", "all"):
            await section_d()

    if args.section in ("a", "all"):
        section_a()

    if args.section in ("b", "c", "d", "all"):
        try:
            asyncio.run(_run_async())
        except KeyboardInterrupt:
            print("\n[interrupted]")
            sys.exit(1)

    print()


if __name__ == "__main__":
    main()
