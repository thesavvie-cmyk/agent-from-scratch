"""Chapter 4 layer demo (block 7): BaseTool, LlmClient, MCP tools.

Sections:
  A -- @tool on calculator and search_web; compare schema with MCP server
  B -- @tool with context parameter: step_info(context)
  C -- LlmClient.generate: plain message
  D -- LlmClient.generate: question requiring calculator tool
  E -- MCP tools via load_mcp_tools; timing: persistent vs reconnecting
"""
from __future__ import annotations

import asyncio
import json
import sys
import time

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

from agentkit.config import FAST_MODEL
from agentkit.context import ExecutionContext
from agentkit.llm import LlmClient, LlmRequest
from agentkit.mcp_client import McpToolset
from agentkit.tools.base import FunctionTool, tool
from agentkit.tools.mcp import load_mcp_tools
from agentkit.tools.web import search_web
from agentkit.tools_manual import calculator
from agentkit.types import Message, ToolCall


def sep(title: str) -> None:
    print(f"\n{'=' * 65}")
    print(f"  {title}")
    print("=" * 65)


CUSTOM_CMD = ("uv", ["run", "python", "-m", "agentkit.servers.tavily_server"])


async def main() -> None:
    # ── A. @tool schemas ───────────────────────────────────────────────────────
    sep("A. @tool on calculator and search_web")

    calc_tool = FunctionTool(calculator)
    web_tool = FunctionTool(search_web)

    for ft in (calc_tool, web_tool):
        print(f"\n[{ft.name}]")
        print(f"  name       : {ft.name}")
        print(f"  description: {ft.description[:80]}")
        params = ft.tool_definition["function"]["parameters"].get("properties", {})
        print(f"  params     : {list(params.keys())}")

    print("\n[Schema comparison: FunctionTool vs MCP server for same functions]")
    async with McpToolset(*CUSTOM_CMD) as ts:
        mcp_tools = load_mcp_tools(ts)
        for mt in mcp_tools:
            ft_match = calc_tool if mt.name == "calculator" else web_tool
            print(f"\n  {mt.name}:")
            print(f"    FunctionTool desc  : {ft_match.description[:70]!r}")
            print(f"    MCP server desc    : {mt.description[:70]!r}")
            ft_params = list(ft_match.tool_definition["function"]["parameters"].get("properties", {}).keys())
            mcp_params = list(mt.tool_definition["function"]["parameters"].get("properties", {}).keys())
            print(f"    FunctionTool params: {ft_params}")
            print(f"    MCP server params  : {mcp_params}")

    # ── B. Tool with context parameter ────────────────────────────────────────
    sep("B. @tool with context: step_info(context)")

    @tool
    def step_info(context: ExecutionContext) -> str:
        """Return the current execution step number."""
        return f"current step: {context.current_step}"

    print(f"\nTool name: {step_info.name}")
    print(f"'context' in schema: {'context' in json.dumps(step_info.tool_definition)}")

    for steps in (0, 3, 7):
        ctx = ExecutionContext()
        for _ in range(steps):
            ctx.increment_step()
        result = await step_info.execute(ctx)
        print(f"  steps={steps} → {result!r}")

    # ── C. LlmClient: plain message ────────────────────────────────────────────
    sep("C. LlmClient.generate — plain message")

    client = LlmClient(FAST_MODEL)
    req = LlmRequest(
        instructions=["You are a helpful assistant. Be very brief."],
        contents=[Message(role="user", content="What is the capital of France? One word.")],
        tools=[],
    )
    resp = await client.generate(req)
    print(f"\nError    : {resp.error_message}")
    print(f"Content  : {resp.content}")
    print(f"Usage    : {resp.usage_metadata}")

    # ── D. LlmClient: calculator tool ─────────────────────────────────────────
    sep("D. LlmClient.generate — question requiring calculator")

    req_d = LlmRequest(
        instructions=["Use the calculator tool for arithmetic."],
        contents=[Message(role="user", content="What is 17 multiplied by 23?")],
        tools=[calc_tool],
    )
    resp_d = await client.generate(req_d)
    print(f"\nContent items: {len(resp_d.content)}")
    for item in resp_d.content:
        if isinstance(item, ToolCall):
            print(f"  ToolCall: name={item.name!r} args={item.arguments}")
        else:
            print(f"  {type(item).__name__}: {str(item)[:80]}")

    # ── E. MCP tools timing ────────────────────────────────────────────────────
    sep("E. MCP tools: persistent session vs reconnecting (5 calls each)")

    N = 5
    ctx_e = ExecutionContext()

    # Persistent session
    async with McpToolset(*CUSTOM_CMD) as ts:
        mcp_tools_e = load_mcp_tools(ts)
        calc_mcp = next(t for t in mcp_tools_e if t.name == "calculator")
        t0 = time.perf_counter()
        for i in range(N):
            await calc_mcp.execute(ctx_e, operator="add", first_number=float(i), second_number=1.0)
        persistent_ms = (time.perf_counter() - t0) * 1000

    # Reconnecting each time
    t0 = time.perf_counter()
    for i in range(N):
        async with McpToolset(*CUSTOM_CMD) as ts:
            tools_tmp = load_mcp_tools(ts)
            calc_tmp = next(t for t in tools_tmp if t.name == "calculator")
            await calc_tmp.execute(ctx_e, operator="add", first_number=float(i), second_number=1.0)
    reconnect_ms = (time.perf_counter() - t0) * 1000

    print(f"\n  Persistent session ({N} calls): {persistent_ms:.0f} ms  "
          f"({persistent_ms/N:.0f} ms/call)")
    print(f"  Reconnecting       ({N} calls): {reconnect_ms:.0f} ms  "
          f"({reconnect_ms/N:.0f} ms/call)")
    print(f"  Overhead per call : {(reconnect_ms - persistent_ms)/N:.0f} ms")
    print("\n  -> Each reconnection spawns a new process (uv run python),")
    print("     imports the server module, and does MCP handshake.")
    print("     Persistent session amortises this cost across all calls.")


if __name__ == "__main__":
    asyncio.run(main())
