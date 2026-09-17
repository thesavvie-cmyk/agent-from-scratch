"""CLI entry point — `agent` command (block 9).

Usage:
    uv run agent ask "what is 2+2?"
    uv run agent chat
    uv run agent chat --session <id>        # resume a persistent session
    uv run agent serve          # long-running daemon mode (used by systemd)
    uv run agent doctor [--json]
    uv run agent budget [--json]
    uv run agent sessions                   # list sessions for default user
    uv run agent sessions --show <id>       # show session details
    uv run agent sessions --delete <id>     # delete one session
    uv run agent sessions --delete-user <u> # delete all sessions for user

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
from agentkit.config import FAST_MODEL, find_uv
from agentkit.llm import LlmClient
from agentkit.mcp_client import McpToolset
from agentkit.prompts import GAIA_AGENT_PROMPT
from agentkit.tools.mcp import load_mcp_tools
from agentkit.utils import display_trace


def _mcp_cmd() -> tuple[str, list[str]]:
    return (find_uv(), ["run", "python", "-m", "agentkit.servers.tavily_server"])

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
        async with McpToolset(*_mcp_cmd()) as ts:
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
    session_id: str | None = getattr(args, "session", None)

    # Set up session store when --session is requested
    session = None
    store = None
    if session_id is not None:
        from agentkit.memory.session import SqliteSessionStore

        store = SqliteSessionStore()
        session = store.get(session_id)
        if session is None:
            # Create a new session with the requested ID is not possible; create fresh
            session = store.create(user_id="cli")
            print(
                f"  [session] Created new session {session.session_id} "
                f"('{session_id}' not found)"
            )
        else:
            print(
                f"  [session] Resumed session {session.session_id} "
                f"({len(session.events)} past events)"
            )

    print(f"Agent chat (model={model}, max_steps={args.max_steps}). Ctrl+C to quit.\n")

    async def _run_session(tools: list[Any]) -> None:
        nonlocal session
        if session is not None and store is not None:
            agent = _make_agent(model, tools, args.max_steps)
            agent._session_store = store  # attach store for persistence
        else:
            agent = _make_agent(model, tools, args.max_steps)

        from agentkit.context import ExecutionContext

        ctx: ExecutionContext | None = None

        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye.")
                break

            if not user_input:
                continue

            t0 = time.perf_counter()
            if session is not None:
                result = await agent.run(user_input, session=session)
                # Reload session so next turn sees persisted events
                if store is not None:
                    session = store.get(session.session_id) or session
                ctx = result.context
            else:
                result = await agent.run(user_input, context=ctx)
                ctx = result.context

            elapsed = time.perf_counter() - t0

            if args.json_out:
                usage = (ctx or result.context).state.get("token_usage", {})
                out: dict[str, Any] = {
                    "output": str(result.output),
                    "steps": (ctx or result.context).current_step,
                    "input_tokens": usage.get("input_tokens", 0),
                    "output_tokens": usage.get("output_tokens", 0),
                    "elapsed_s": round(elapsed, 2),
                    "error": result.error,
                    "session_id": session.session_id if session else None,
                }
                print(json.dumps(out, ensure_ascii=False))
            else:
                if result.error:
                    print(f"[error] {result.error}")
                else:
                    print(f"Agent: {result.output}")

            if args.trace:
                print(display_trace(ctx or result.context), file=sys.stderr)

    if args.no_tools:
        await _run_session([])
    else:
        async with McpToolset(*_mcp_cmd()) as ts:
            tools = load_mcp_tools(ts)
            await _run_session(tools)


# ── memory ────────────────────────────────────────────────────────────────────


def cmd_memory(args: argparse.Namespace) -> None:
    """List, search, or delete long-term memories."""
    from agentkit.memory.longterm import LongTermMemory

    # LongTermMemory lazy-loads ChromaDB; only imported when this subcommand runs
    mem = LongTermMemory()
    user_id: str = getattr(args, "user", "default")

    if getattr(args, "forget", None):
        ok = mem.delete(args.forget)
        if ok:
            print(f"Deleted memory {args.forget}")
        else:
            print(f"Memory not found: {args.forget}", file=sys.stderr)
            sys.exit(1)
        return

    if getattr(args, "forget_all", False):
        n = mem.delete_user(user_id)
        print(f"Deleted {n} memory record(s) for user '{user_id}'")
        return

    if getattr(args, "search_query", None):
        import datetime

        results = mem.search(user_id=user_id, query=args.search_query, min_score=0.0, top_k=10)
        if not results:
            print(f"No memories found for user '{user_id}'")
            return
        print(f"\n  Memories for '{user_id}' matching '{args.search_query}':\n")
        print(f"  {'Score':>6}  {'Updated':<20}  {'Text'}")
        print(f"  {'─'*6}  {'─'*20}  {'─'*50}")
        for m in results:
            updated = datetime.datetime.fromtimestamp(m.updated_at).strftime("%Y-%m-%d %H:%M:%S")
            print(f"  {m.score:>6.3f}  {updated:<20}  {m.text[:80]}")
        print()
        return

    # Default: list all
    import datetime

    memories = mem.list_all(user_id)
    if not memories:
        print(f"No memories for user '{user_id}'")
        return
    print(f"\n  Memories for user '{user_id}' ({len(memories)} total):\n")
    print(f"  {'ID':<38}  {'Updated':<20}  {'Text'}")
    print(f"  {'─'*38}  {'─'*20}  {'─'*50}")
    for m in memories:
        updated = datetime.datetime.fromtimestamp(m.updated_at).strftime("%Y-%m-%d %H:%M:%S")
        print(f"  {m.id:<38}  {updated:<20}  {m.text[:80]}")
    print()


# ── sessions ──────────────────────────────────────────────────────────────────


def cmd_sessions(args: argparse.Namespace) -> None:
    """List, inspect, or delete sessions."""
    from agentkit.memory.session import SqliteSessionStore

    store = SqliteSessionStore()

    if getattr(args, "delete", None):
        ok = store.delete(args.delete)
        if ok:
            print(f"Deleted session {args.delete}")
        else:
            print(f"Session not found: {args.delete}", file=sys.stderr)
            sys.exit(1)
        return

    if getattr(args, "delete_user", None):
        n = store.delete_user(args.delete_user)
        print(f"Deleted {n} session(s) for user '{args.delete_user}'")
        return

    if getattr(args, "show", None):
        session = store.get(args.show)
        if session is None:
            print(f"Session not found: {args.show}", file=sys.stderr)
            sys.exit(1)
        import datetime

        print(f"\n  Session   : {session.session_id}")
        print(f"  User      : {session.user_id}")
        created = datetime.datetime.fromtimestamp(session.created_at).strftime("%Y-%m-%d %H:%M:%S")
        updated = datetime.datetime.fromtimestamp(session.updated_at).strftime("%Y-%m-%d %H:%M:%S")
        print(f"  Created   : {created}")
        print(f"  Updated   : {updated}")
        print(f"  Events    : {len(session.events)}")
        print(f"  State keys: {list(session.state.keys())}")
        if session.metadata:
            print(f"  Metadata  : {session.metadata}")
        if session.events:
            print("\n  Event history:")
            for evt in session.events:
                for item in evt.content:
                    from agentkit.types import Message, ToolCall, ToolResult

                    if isinstance(item, Message):
                        snippet = item.content[:80].replace("\n", " ")
                        print(f"    [{item.role}] {snippet}")
                    elif isinstance(item, ToolCall):
                        print(f"    [tool_call] {item.name}({item.arguments})"[:80])
                    elif isinstance(item, ToolResult):
                        snippet = str(item.content)[:60].replace("\n", " ")
                        print(f"    [tool_result] {item.name}: {snippet}")
        print()
        return

    # Default: list sessions for 'default' user (or --user)
    user_id = getattr(args, "user", "default")
    sessions = store.list_sessions(user_id)
    if not sessions:
        print(f"No sessions found for user '{user_id}'")
        return

    import datetime

    print(f"\n  {'Session ID':<38} {'Events':>7} {'Updated':<20}")
    print(f"  {'─' * 38} {'─' * 7} {'─' * 20}")
    for s in sessions:
        updated = datetime.datetime.fromtimestamp(s.updated_at).strftime("%Y-%m-%d %H:%M:%S")
        print(f"  {s.session_id:<38} {len(s.events):>7} {updated}")
    print()


# ── serve (daemon mode for systemd) ──────────────────────────────────────────


async def cmd_serve() -> None:
    """Long-running daemon mode — blocks until SIGTERM/SIGINT.

    In block 22 this becomes an A2A server.  For now it just keeps the
    process alive so systemd can manage it as a persistent service.
    """
    import signal

    logger.info("agentkit serve: started, waiting for requests")
    print("agentkit serve: running (block 22 will add A2A server here)", flush=True)

    stop = asyncio.Event()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, stop.set)
        except (OSError, NotImplementedError):
            pass

    await stop.wait()
    logger.info("agentkit serve: shutting down")


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
    chat_p.add_argument(
        "--session",
        default=None,
        metavar="ID",
        help="Session ID to resume (creates new if not found)",
    )

    # memory
    mem_p = sub.add_parser("memory", help="List, search, or delete long-term memories")
    mem_p.add_argument("--user", default="default", help="User ID (default: 'default')")
    mem_p.add_argument("--list", action="store_true", dest="list_mem",
                       help="List all memories (default action)")
    mem_p.add_argument("--search", metavar="QUERY", dest="search_query",
                       help="Search memories by query")
    mem_p.add_argument("--forget", metavar="ID", help="Delete one memory by ID")
    mem_p.add_argument("--forget-all", action="store_true", dest="forget_all",
                       help="Delete all memories for user")

    # sessions
    sess_p = sub.add_parser("sessions", help="List, inspect, or delete sessions")
    sess_p.add_argument("--user", default="default", help="User ID (default: 'default')")
    sess_p.add_argument("--show", metavar="ID", help="Show full session history")
    sess_p.add_argument("--delete", metavar="ID", help="Delete a session by ID")
    sess_p.add_argument(
        "--delete-user", metavar="USER", dest="delete_user", help="Delete all sessions for user"
    )

    # serve
    sub.add_parser("serve", help="Long-running daemon mode (used by systemd)")

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
    elif args.command == "serve":
        asyncio.run(cmd_serve())
    elif args.command == "doctor":
        sys.exit(asyncio.run(cmd_doctor(args)))
    elif args.command == "budget":
        cmd_budget(args)
    elif args.command == "memory":
        cmd_memory(args)
    elif args.command == "sessions":
        cmd_sessions(args)
