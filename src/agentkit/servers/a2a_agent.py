"""Deployable A2A agent server (block 22).

Assembles an Agent with web search (Tavily via MCP), code execution (e2b),
and file tools, then serves it as an A2A-compliant HTTP server.

The McpToolset is kept alive for the lifetime of the server — opened once
at startup and closed on shutdown.  This avoids the session-lifetime bug
where tools fail immediately if the session is closed before use.

Usage
-----
    # localhost only, no auth (development)
    uv run --group a2a python -m agentkit.servers.a2a_agent

    # with Bearer-token auth
    A2A_TOKEN=my-secret uv run --group a2a python -m agentkit.servers.a2a_agent

    # full options
    uv run --group a2a python -m agentkit.servers.a2a_agent \\
        --port 8090 \\
        --token $A2A_TOKEN \\
        --budget-tokens 200000 \\
        --rate-limit 20 \\
        --model anthropic/claude-haiku-4-5

    # verify all components are present before starting
    uv run agent --diagnose

Security checklist before exposing to the network
--------------------------------------------------
1. Always set --token (or A2A_TOKEN env var).
2. Bind to 127.0.0.1, put nginx in front for TLS:
       location / { proxy_pass http://127.0.0.1:8090; }
3. Set --budget-tokens and --rate-limit to bound cost.
4. Keep AGENTKIT_CAPTURE_CONTENT unset (default false) so
   task inputs don't appear in trace files.
5. Run `uv run agent --diagnose` to confirm ANTHROPIC_API_KEY
   and TAVILY_API_KEY are present before opening traffic.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys
from pathlib import Path
from typing import Any

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="A2A agent server — serves a full agent over HTTP",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8090, help="TCP port (default: 8090)")
    parser.add_argument(
        "--token",
        default=os.getenv("A2A_TOKEN"),
        help="Bearer auth token (or set A2A_TOKEN env var; omit for no auth)",
    )
    parser.add_argument(
        "--budget-tokens",
        type=int,
        default=int(os.getenv("MAX_TOKENS_PER_HOUR", "500000")),
        help="Max tokens per hour (default: 500000)",
    )
    parser.add_argument(
        "--rate-limit",
        type=int,
        default=int(os.getenv("A2A_RATE_LIMIT", "60")),
        help="Max requests per IP per minute (default: 60)",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("AGENT_MODEL", "anthropic/claude-haiku-4-5"),
        help="LiteLLM model string (default: anthropic/claude-haiku-4-5)",
    )
    parser.add_argument(
        "--workspace",
        default=os.getenv("AGENT_WORKSPACE", str(Path.home() / ".agentkit" / "workspace")),
        help="Workspace directory for file tools",
    )
    parser.add_argument(
        "--no-code",
        action="store_true",
        default=False,
        help="Disable e2b code execution (no E2B_API_KEY needed)",
    )
    return parser.parse_args()


async def _main(args: argparse.Namespace) -> None:
    from dotenv import load_dotenv

    load_dotenv()

    from agentkit.a2a import A2AServer
    from agentkit.agent import Agent
    from agentkit.budget import BudgetGuard
    from agentkit.config import find_uv
    from agentkit.llm import LlmClient
    from agentkit.mcp_client import McpToolset
    from agentkit.tools.base import BaseTool
    from agentkit.tools.files import list_files, read_file
    from agentkit.tools.mcp import load_mcp_tools

    # ── Budget guard ──────────────────────────────────────────────────────────
    guard = BudgetGuard(max_tokens=args.budget_tokens)

    # ── LLM client ────────────────────────────────────────────────────────────
    model = LlmClient(args.model, budget_guard=guard)

    # ── Tool assembly — McpToolset stays alive for server lifetime ────────────
    mcp_cmd = (find_uv(), ["run", "python", "-m", "agentkit.servers.tavily_server"])

    async with McpToolset(*mcp_cmd) as ts:
        tools: list[BaseTool] = list(load_mcp_tools(ts))

        # File tools (sandboxed to workspace directory)
        ws_path = Path(args.workspace)
        ws_path.mkdir(parents=True, exist_ok=True)
        tools.extend([list_files, read_file])

        # Code execution (optional — requires E2B_API_KEY)
        if not args.no_code and os.getenv("E2B_API_KEY"):
            from agentkit.tools.code_execution import execute_python
            tools.append(execute_python)

        _tool_names = [t.name for t in tools]
        logger.info("Tools loaded: %s", _tool_names)

        # ── Agent ─────────────────────────────────────────────────────────────
        agent = Agent(
            model=model,
            tools=tools,
            name="a2a_agent",
            instructions=(
                "You are a capable AI assistant with access to web search, "
                "file reading, and (optionally) Python code execution.\n"
                "Use search_web for current information.\n"
                "Use read_file / list_files to inspect files in the workspace.\n"
                "Use execute_python for calculations and data processing.\n"
                "Be thorough and cite sources when using search results."
            ),
            max_steps=12,
        )

        # ── Server ────────────────────────────────────────────────────────────
        server = A2AServer(
            agent,
            name="a2a_agent",
            description=(
                "General-purpose agent with web search, file tools, "
                "and optional Python execution."
            ),
            host=args.host,
            port=args.port,
            auth_token=args.token,
            rate_limit=args.rate_limit,
            budget_guard=guard,
        )

        await server.start()

        auth_note = "auth=token" if args.token else "auth=NONE (dev mode)"
        print(f"A2A agent server running at {server.url}")
        print(f"  {auth_note}  rate_limit={args.rate_limit}/min  "
              f"budget={args.budget_tokens:,} tok/hr")
        print(f"  Tools: {_tool_names}")
        print(f"  AgentCard: {server.url}/.well-known/agent.json")
        if not args.token:
            print("\n  WARNING: No auth token set. Anyone who can reach this port")
            print("  can run tasks using your API keys.")
        print("\nPress Ctrl-C to stop.")

        # ── Wait for shutdown ─────────────────────────────────────────────────
        stop_event = asyncio.Event()

        def _on_signal(*_: Any) -> None:
            stop_event.set()

        if sys.platform != "win32":
            loop = asyncio.get_running_loop()
            loop.add_signal_handler(signal.SIGTERM, _on_signal)
            loop.add_signal_handler(signal.SIGINT, _on_signal)

        try:
            await stop_event.wait()
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass

        print("\nShutting down…")
        await server.stop()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    )
    args = _parse_args()
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try:
        asyncio.run(_main(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
