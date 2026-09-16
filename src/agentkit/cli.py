"""CLI entry point — `agent` command (block 9).

Usage:
    uv run agent ask "what is 2+2?"
    uv run agent chat
    uv run agent doctor [--json]
    uv run agent budget [--json]

Common flags (ask / chat):
    --model MODEL       Model ID (default: FAST_MODEL)
    --max-steps N       Max agent steps (default: 8)
    --no-tools          Disable web search / calculator tools
    --trace             Print execution trace after each response
    --json              Output as JSON {output, steps, tokens, elapsed}
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from typing import Any

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

from agentkit.agent import Agent
from agentkit.budget import BudgetGuard
from agentkit.config import FAST_MODEL
from agentkit.llm import LlmClient
from agentkit.mcp_client import McpToolset
from agentkit.prompts import GAIA_AGENT_PROMPT
from agentkit.tools.mcp import load_mcp_tools
from agentkit.utils import display_trace

_MCP_CMD = ("uv", ["run", "python", "-m", "agentkit.servers.tavily_server"])

logger = logging.getLogger(__name__)


# ── Agent factory ─────────────────────────────────────────────────────────────


def _make_agent(
    model: str,
    tools: list[Any],
    max_steps: int,
) -> Agent:
    guard = BudgetGuard()
    client = LlmClient(model, budget_guard=guard)
    return Agent(
        model=client,
        tools=tools,
        instructions=GAIA_AGENT_PROMPT,
        max_steps=max_steps,
    )


# ── ask ───────────────────────────────────────────────────────────────────────


async def cmd_ask(args: argparse.Namespace) -> int:
    model = args.model or FAST_MODEL
    t0 = time.perf_counter()

    if args.no_tools:
        agent = _make_agent(model, [], args.max_steps)
        result = await agent.run(args.question)
    else:
        async with McpToolset(*_MCP_CMD) as ts:
            tools = load_mcp_tools(ts)
            agent = _make_agent(model, tools, args.max_steps)
            result = await agent.run(args.question)

    elapsed = time.perf_counter() - t0
    usage = result.context.state.get("token_usage", {})

    if args.json_out:
        out: dict[str, Any] = {
            "output": str(result.output),
            "steps": result.context.current_step,
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "elapsed_s": round(elapsed, 2),
            "error": result.error,
        }
        print(json.dumps(out, ensure_ascii=False))
    else:
        if result.error:
            print(f"[error] {result.error}", file=sys.stderr)
        else:
            print(result.output)

    if args.trace:
        print(display_trace(result.context), file=sys.stderr)

    return 1 if result.error else 0


# ── chat ──────────────────────────────────────────────────────────────────────


async def cmd_chat(args: argparse.Namespace) -> None:
    model = args.model or FAST_MODEL
    print(f"Agent chat (model={model}, max_steps={args.max_steps}). Ctrl+C to quit.\n")

    async def _run_session(tools: list[Any]) -> None:
        agent = _make_agent(model, tools, args.max_steps)
        from agentkit.context import ExecutionContext

        ctx = ExecutionContext()

        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye.")
                break

            if not user_input:
                continue

            t0 = time.perf_counter()
            result = await agent.run(user_input, context=ctx)
            elapsed = time.perf_counter() - t0
            # Re-use same ctx so history accumulates across turns
            ctx = result.context

            if args.json_out:
                usage = ctx.state.get("token_usage", {})
                out: dict[str, Any] = {
                    "output": str(result.output),
                    "steps": ctx.current_step,
                    "input_tokens": usage.get("input_tokens", 0),
                    "output_tokens": usage.get("output_tokens", 0),
                    "elapsed_s": round(elapsed, 2),
                    "error": result.error,
                }
                print(json.dumps(out, ensure_ascii=False))
            else:
                if result.error:
                    print(f"[error] {result.error}")
                else:
                    print(f"Agent: {result.output}")

            if args.trace:
                print(display_trace(ctx), file=sys.stderr)

    if args.no_tools:
        await _run_session([])
    else:
        async with McpToolset(*_MCP_CMD) as ts:
            tools = load_mcp_tools(ts)
            await _run_session(tools)


# ── doctor ────────────────────────────────────────────────────────────────────


async def cmd_doctor(args: argparse.Namespace) -> int:
    from agentkit.diagnostics import run_doctor

    return await run_doctor(json_output=getattr(args, "json_out", False))


# ── budget ────────────────────────────────────────────────────────────────────


def cmd_budget(args: argparse.Namespace) -> None:
    guard = BudgetGuard()
    s = guard.status()

    if getattr(args, "json_out", False):
        print(json.dumps(s, ensure_ascii=False))
        return

    h = s["resets_in_seconds"] // 3600
    m = (s["resets_in_seconds"] % 3600) // 60
    sec = s["resets_in_seconds"] % 60
    reset_str = f"{h:02d}:{m:02d}:{sec:02d}"

    print("\n=== Budget Status ===\n")
    print(f"  Requests : {s['requests']:,} / {s['max_requests']:,}")
    print(f"  Tokens   : {s['tokens_used']:,} / {s['max_tokens']:,}")
    print(f"    input  : {s['input_tokens']:,}")
    print(f"    output : {s['output_tokens']:,}")
    print(f"  Resets in: {reset_str}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent",
        description="agentkit CLI — ask questions, chat, or check system health",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def _add_run_flags(p: argparse.ArgumentParser) -> None:
        p.add_argument("--model", default=None, help="Model ID (default: FAST_MODEL)")
        p.add_argument("--max-steps", type=int, default=8, dest="max_steps")
        p.add_argument("--no-tools", action="store_true", dest="no_tools")
        p.add_argument("--trace", action="store_true")
        p.add_argument("--json", action="store_true", dest="json_out")

    # ask
    ask_p = sub.add_parser("ask", help="Answer a single question")
    ask_p.add_argument("question", help="Question to answer")
    _add_run_flags(ask_p)

    # chat
    chat_p = sub.add_parser("chat", help="Interactive multi-turn chat")
    _add_run_flags(chat_p)

    # doctor
    doc_p = sub.add_parser("doctor", help="Check system health and connectivity")
    doc_p.add_argument("--json", action="store_true", dest="json_out")

    # budget
    bud_p = sub.add_parser("budget", help="Show current hourly budget consumption")
    bud_p.add_argument("--json", action="store_true", dest="json_out")

    return parser


def _setup_logging() -> None:
    import os

    level = os.getenv("LOG_LEVEL", "WARNING").upper()
    logging.basicConfig(
        level=getattr(logging, level, logging.WARNING),
        format="%(levelname)s %(name)s: %(message)s",
    )


def main() -> None:
    _setup_logging()
    parser = _build_parser()
    args = parser.parse_args()

    # Windows asyncio policy for subprocess support
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    if args.command == "ask":
        sys.exit(asyncio.run(cmd_ask(args)))
    elif args.command == "chat":
        asyncio.run(cmd_chat(args))
    elif args.command == "doctor":
        sys.exit(asyncio.run(cmd_doctor(args)))
    elif args.command == "budget":
        cmd_budget(args)
