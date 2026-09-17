"""Chapter 8 workspace experiments (block 18).

Sections
--------
a) Bridge demo — inject search_web stub into sandbox, call it from Python code.
   Uses 127.0.0.1; requires a TAVILY_API_KEY.
b) Workspace file tools — write CSV to sandbox, process with execute_python,
   read output back with read_sandbox_file.
c) CLI tools — pip-install a package inside the sandbox, use in execute_python.
d) Full pipeline — agent searches for data, writes it to a sandbox file,
   processes with execute_python, and reads the result back.

Usage
-----
    uv run python experiments/ch08_workspace.py --section a
    uv run python experiments/ch08_workspace.py --section all

Requirements
------------
    ANTHROPIC_API_KEY + E2B_API_KEY (all sections).
    TAVILY_API_KEY required for sections a and d.

Bridge note (section a)
-----------------------
The stub code calls 127.0.0.1:<port>.  This works when the process and the
e2b template both resolve 127.0.0.1 to the same machine (local template or
same-host sandbox).  For remote e2b cloud sandboxes, expose the bridge with
an inbound tunnel and pass the public host to stub_code().
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

from agentkit.agent import Agent
from agentkit.config import FAST_MODEL, find_uv
from agentkit.llm import LlmClient
from agentkit.mcp_client import McpToolset
from agentkit.tools.mcp import load_mcp_tools

MCP_CMD = (find_uv(), ["run", "python", "-m", "agentkit.servers.tavily_server"])


def _hr(title: str) -> None:
    print(f"\n{'=' * 68}\n  {title}\n{'=' * 68}")


# ── Section A — Bridge demo ────────────────────────────────────────────────────


_QUERIES = [
    "Python asyncio best practices 2024",
    "LLM agent frameworks comparison 2024",
    "e2b sandbox code execution security",
    "Tavily search API documentation",
    "RAG retrieval augmented generation techniques",
    "vector databases comparison 2024",
    "LangChain vs LlamaIndex 2024",
    "OpenAI function calling guide",
    "Anthropic Claude tool use examples",
    "agent memory persistence patterns",
]


async def section_a() -> None:
    _hr("Section A -- Bridge env mode: 10 searches in one execute_python vs 10 agent rounds")

    import os

    import e2b_code_interpreter as e2b

    from agentkit.config import E2B_TIMEOUT
    from agentkit.context import ExecutionContext
    from agentkit.tools.mcp import load_mcp_tools
    from agentkit.tools.sandbox_bridge import SandboxBridge

    tavily_key = os.environ.get("TAVILY_API_KEY", "")
    if not tavily_key:
        print("  SKIP: TAVILY_API_KEY not set.")
        return

    async with McpToolset(*MCP_CMD) as ts:
        search_tools = list(load_mcp_tools(ts))
    search_tool = next((t for t in search_tools if t.name == "search_web"), None)
    if search_tool is None:
        print("  ERROR: search_web not found.")
        return

    # ── Config 1: 10 searches inside ONE execute_python call (env bridge) ─────
    print("\n  [1] 10 searches inside one execute_python (env bridge)...")
    sandbox = await e2b.AsyncSandbox.create(timeout=E2B_TIMEOUT)
    ctx = ExecutionContext()
    ctx.code_env = sandbox

    try:
        loop = asyncio.get_running_loop()
        bridge = SandboxBridge(
            [search_tool], ctx, loop, envs={"TAVILY_API_KEY": tavily_key}
        )
        host, port = bridge.start()  # no HTTP server (env-only)
        await sandbox.run_code(bridge.env_setup_code())
        await sandbox.run_code(bridge.stub_code(host, port))
        print(f"  Stubs injected. HTTP port={port} (0=env-only). {bridge.instructions_hint()}")

        queries_py = repr(_QUERIES)
        code = f"""
queries = {queries_py}
results = {{}}
for q in queries:
    r = search_web(query=q, max_results=3)
    first_line = r.split("\\n")[0] if r else "(empty)"
    results[q] = first_line
    print(f"  OK: {{q[:40]}}")
print(f"\\nTotal queries: {{len(results)}}")
"""
        t0 = time.perf_counter()
        execution = await sandbox.run_code(code)
        elapsed_single = time.perf_counter() - t0

        for line in execution.logs.stdout or []:
            print(f"    {line}")
        if execution.error:
            print(f"  ERROR: {execution.error.name}: {execution.error.value}")
    finally:
        bridge.stop()
        await sandbox.kill()

    # ── Config 2: 10 searches as 10 separate agent tool calls ─────────────────
    print("\n  [2] 10 searches as 10 separate agent tool-call rounds...")

    agent = Agent(
        model=LlmClient(FAST_MODEL),
        tools=search_tools,
        instructions="Run exactly the searches listed. Do not skip any. Collect all results and report done.",
        max_steps=20,
    )

    queries_str = "\n".join(f"- {q}" for q in _QUERIES)
    t0 = time.perf_counter()
    result = await agent.run(
        f"Search the web for each of the following queries (all 10):\n{queries_str}\n"
        "After all searches, reply 'Done: N searches completed'."
    )
    elapsed_rounds = time.perf_counter() - t0

    u = result.context.state.get("token_usage", {})
    from agentkit.types import ToolCall
    actual_searches = sum(
        1 for ev in result.context.events
        for item in ev.content
        if isinstance(item, ToolCall) and item.name == "search_web"
    )

    # ── Comparison table ───────────────────────────────────────────────────────
    print(f"\n  {'Config':<30} {'Queries':>8} {'Time':>8} {'LLM rounds':>11}")
    print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*11}")
    print(f"  {'10 searches in 1 execute_python':<30} {10:>8} {elapsed_single:>7.1f}s {'1 (code call)':>11}")
    print(
        f"  {'10 agent tool-call rounds':<30} {actual_searches:>8} "
        f"{elapsed_rounds:>7.1f}s {result.context.current_step:>10} steps"
    )
    print(f"\n  in={u.get('input_tokens', 0)}  out={u.get('output_tokens', 0)}  (agent config only)")
    print(
        "\n  Insight: env bridge lets the model issue a search loop inside one "
        "execute_python call — no extra agent steps, no token overhead per search."
    )


# ── Section B — Workspace file tools ──────────────────────────────────────────


async def section_b() -> None:
    _hr("Section B -- Workspace: write CSV → execute_python → read result")

    import e2b_code_interpreter as e2b

    from agentkit.config import E2B_TIMEOUT
    from agentkit.context import ExecutionContext
    from agentkit.tools.workspace_sandbox import (
        list_sandbox_files,
        read_sandbox_file,
        run_command,
        write_sandbox_file,
    )

    sandbox = await e2b.AsyncSandbox.create(timeout=E2B_TIMEOUT)
    ctx = ExecutionContext()
    ctx.code_env = sandbox

    try:
        csv_data = "name,score\nAlice,92\nBob,78\nCarol,88\nDave,95\nEve,71\n"

        # Write
        t0 = time.perf_counter()
        msg = await write_sandbox_file(context=ctx, path="/home/user/scores.csv", content=csv_data)
        print(f"  Write: {msg}")

        # List
        raw = await list_sandbox_files(context=ctx, path="/home/user")
        entries = json.loads(raw)
        print(f"  List ({len(entries)} entries):", [e["name"] for e in entries])

        # Shell: wc -l
        raw_cmd = await run_command(context=ctx, cmd="wc -l /home/user/scores.csv")
        cmd_data = json.loads(raw_cmd)
        print(f"  wc -l: {cmd_data['stdout'].strip()}")

        # Process with execute_python
        code = """
with open('/home/user/scores.csv') as f:
    lines = f.readlines()[1:]  # skip header
scores = [int(line.split(',')[1]) for line in lines if line.strip()]
avg = sum(scores) / len(scores)
print(f"Count: {len(scores)}, Avg: {avg:.1f}, Max: {max(scores)}")
with open('/home/user/summary.txt', 'w') as f:
    f.write(f"avg={avg:.1f} max={max(scores)} count={len(scores)}")
"""
        exec_result = await sandbox.run_code(code)
        for line in exec_result.logs.stdout or []:
            print(f"  execute_python: {line}")

        # Read result file
        summary = await read_sandbox_file(context=ctx, path="/home/user/summary.txt")
        print(f"  Read summary.txt: {summary}")

        elapsed = time.perf_counter() - t0
        print(f"\n  Total time: {elapsed:.1f}s")
    finally:
        await sandbox.kill()


# ── Section C — CLI tools ──────────────────────────────────────────────────────


async def section_c() -> None:
    _hr("Section C -- CLI: pip install httpx inside sandbox, use in execute_python")

    import e2b_code_interpreter as e2b

    from agentkit.config import E2B_TIMEOUT
    from agentkit.context import ExecutionContext
    from agentkit.tools.workspace_sandbox import run_command

    sandbox = await e2b.AsyncSandbox.create(timeout=E2B_TIMEOUT)
    ctx = ExecutionContext()
    ctx.code_env = sandbox

    try:
        # Install
        print("  Installing httpx...")
        t0 = time.perf_counter()
        raw = await run_command(context=ctx, cmd="pip install httpx --quiet")
        data = json.loads(raw)
        elapsed = time.perf_counter() - t0
        print(f"  pip exit_code={data['exit_code']}  ({elapsed:.1f}s)")
        if data["stderr"] and data["exit_code"] != 0:
            print(f"  stderr: {data['stderr'][:200]}")

        # Use in execute_python
        exec_result = await sandbox.run_code(
            "import httpx; print('httpx version:', httpx.__version__)"
        )
        for line in exec_result.logs.stdout or []:
            print(f"  execute_python: {line}")
        if exec_result.error:
            print(f"  ERROR: {exec_result.error.name}: {exec_result.error.value}")
        else:
            print("  SUCCESS: package installed and importable.")
    finally:
        await sandbox.kill()


# ── Section D — Full pipeline agent ───────────────────────────────────────────


async def section_d() -> None:
    _hr("Section D -- Full pipeline: agent searches, writes file, processes with code")

    question = (
        "Search for the top 5 programming languages by popularity in 2024. "
        "Write the results to /home/user/languages.txt inside the sandbox, "
        "one language per line. Then use execute_python to read the file, "
        "count the languages, and print each one numbered."
    )

    async with McpToolset(*MCP_CMD) as ts:
        search_tools = list(load_mcp_tools(ts))

    agent = Agent(
        model=LlmClient(FAST_MODEL),
        tools=search_tools,
        instructions=(
            "You have web search, file tools, and Python execution. "
            "Follow the task steps: search first, then write a file, then process it."
        ),
        max_steps=10,
        code_execution="e2b",
        workspace=True,
    )

    t0 = time.perf_counter()
    result = await agent.run(question)
    elapsed = time.perf_counter() - t0

    u = result.context.state.get("token_usage", {})
    print(f"\n  Steps: {result.context.current_step}  "
          f"in={u.get('input_tokens', 0)}  out={u.get('output_tokens', 0)}  ({elapsed:.1f}s)")
    print(f"  Answer: {str(result.output)[:500]}")
    if result.error:
        print(f"  Error: {result.error}")


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Block 18 workspace experiments")
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
