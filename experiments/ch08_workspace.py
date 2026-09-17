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


async def section_a() -> None:
    _hr("Section A -- Bridge: search_web callable from execute_python")

    import e2b_code_interpreter as e2b

    from agentkit.config import E2B_TIMEOUT
    from agentkit.context import ExecutionContext
    from agentkit.tools.mcp import load_mcp_tools
    from agentkit.tools.sandbox_bridge import SandboxBridge

    async with McpToolset(*MCP_CMD) as ts:
        search_tools = list(load_mcp_tools(ts))

    search_tool = next((t for t in search_tools if t.name == "search_web"), None)
    if search_tool is None:
        print("  ERROR: search_web tool not found in MCP server.")
        return

    sandbox = await e2b.AsyncSandbox.create(timeout=E2B_TIMEOUT)
    ctx = ExecutionContext()
    ctx.code_env = sandbox

    try:
        loop = asyncio.get_running_loop()
        bridge = SandboxBridge([search_tool], ctx, loop)
        host, port = bridge.start()
        print(f"  Bridge started at {host}:{port}")
        print(f"  Hint: {bridge.instructions_hint()}")

        stubs = bridge.stub_code(host, port)
        await sandbox.run_code(stubs)
        print("  Stubs injected into sandbox.")

        # Python code that calls search_web from inside the sandbox
        code = """
results = search_web(query="Python asyncio event loop basics")
import json
data = json.loads(results)
titles = [r.get('title', '') for r in data.get('results', data if isinstance(data, list) else [])]
print("Search returned", len(titles), "results")
for t in titles[:3]:
    print(" -", t)
"""
        t0 = time.perf_counter()
        execution = await sandbox.run_code(code)
        elapsed = time.perf_counter() - t0

        print(f"\n  execute_python({elapsed:.1f}s):")
        for line in (execution.logs.stdout or []):
            print(f"    {line}")
        if execution.error:
            print(f"  ERROR: {execution.error.name}: {execution.error.value}")
        else:
            print("  SUCCESS: search_web called from inside sandbox Python code.")
    finally:
        bridge.stop()
        await sandbox.kill()


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
