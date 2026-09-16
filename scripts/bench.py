"""Benchmark script — measure agent performance on PC and server (block 9).

Runs 5 fixed questions (2 without tools, 3 with web search), records latency,
steps, tokens, and peak memory per question.  Saves results to
results/bench-<hostname>.json.

Usage:
    uv run python scripts/bench.py
    uv run python scripts/bench.py --compare results/bench-pc.json results/bench-server.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import sys
import time
from pathlib import Path
from typing import Any

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

# Resolve project root so the script works from any CWD
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from agentkit.agent import Agent
from agentkit.budget import BudgetGuard
from agentkit.config import FAST_MODEL, find_uv
from agentkit.llm import LlmClient
from agentkit.mcp_client import McpToolset
from agentkit.tools.mcp import load_mcp_tools

_MCP_CMD = (find_uv(), ["run", "python", "-m", "agentkit.servers.tavily_server"])

# ── Fixed question set ────────────────────────────────────────────────────────

QUESTIONS: list[dict[str, Any]] = [
    # No tools — model knowledge sufficient
    {"id": "q1_math", "q": "What is 1234 multiplied by 5678?", "needs_tools": False},
    {"id": "q2_fact", "q": "In what year did the Berlin Wall fall?", "needs_tools": False},
    # With tools — require current/external data
    {
        "id": "q3_current",
        "q": "What is the current price of gold per troy ounce in USD?",
        "needs_tools": True,
    },
    {
        "id": "q4_recent",
        "q": "Who won the most recent FIFA World Cup and in which country was it held?",
        "needs_tools": True,
    },
    {
        "id": "q5_weather",
        "q": "What is the current weather in London, UK (temperature in Celsius)?",
        "needs_tools": True,
    },
]


# ── Memory helper ─────────────────────────────────────────────────────────────


def _peak_rss_mb() -> float:
    """Return current RSS memory in MB (cross-platform via psutil)."""
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / 1e6
    except ImportError:
        return 0.0


# ── Single question runner ────────────────────────────────────────────────────


async def _run_question(
    q: dict[str, Any],
    agent: Agent,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    mem_before = _peak_rss_mb()
    t0 = time.perf_counter()
    error: str | None = None
    result = None

    async with semaphore:
        try:
            result = await agent.run(q["q"])
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"

    elapsed = time.perf_counter() - t0
    mem_after = _peak_rss_mb()

    usage = result.context.state.get("token_usage", {}) if result else {}
    steps = result.context.current_step if result else 0
    if result and result.error and not error:
        error = result.error

    return {
        "id": q["id"],
        "question": q["q"],
        "needs_tools": q["needs_tools"],
        "elapsed_s": round(elapsed, 3),
        "steps": steps,
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "memory_delta_mb": round(mem_after - mem_before, 1),
        "error": error,
        "output": str(result.output)[:200] if result else None,
    }


# ── Main benchmark ────────────────────────────────────────────────────────────


async def run_bench() -> dict[str, Any]:
    model = FAST_MODEL
    semaphore = asyncio.Semaphore(2)
    guard = BudgetGuard()

    no_tool_qs = [q for q in QUESTIONS if not q["needs_tools"]]
    tool_qs = [q for q in QUESTIONS if q["needs_tools"]]

    all_rows: list[dict[str, Any]] = []

    # ── No-tools questions ────────────────────────────────────────────────────
    agent_no_tools = Agent(
        model=LlmClient(model, budget_guard=guard),
        tools=[],
        max_steps=5,
    )
    for q in no_tool_qs:
        row = await _run_question(q, agent_no_tools, semaphore)
        print(f"  [{row['id']}] {row['elapsed_s']:.2f}s  {row['steps']} steps", flush=True)
        all_rows.append(row)

    # ── With-tools questions (single MCP session) ─────────────────────────────
    mcp_start = time.perf_counter()
    async with McpToolset(*_MCP_CMD) as ts:
        mcp_cold_start_s = round(time.perf_counter() - mcp_start, 2)
        tools = load_mcp_tools(ts)

        agent_tools = Agent(
            model=LlmClient(model, budget_guard=guard),
            tools=tools,
            max_steps=5,
        )
        tasks = [_run_question(q, agent_tools, semaphore) for q in tool_qs]
        tool_rows = await asyncio.gather(*tasks)
        for row in tool_rows:
            print(f"  [{row['id']}] {row['elapsed_s']:.2f}s  {row['steps']} steps", flush=True)
            all_rows.append(row)

    return {
        "hostname": socket.gethostname(),
        "model": model,
        "mcp_cold_start_s": mcp_cold_start_s,
        "questions": all_rows,
        "summary": _summarise(all_rows),
    }


def _summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {}
    return {
        "n": n,
        "total_elapsed_s": round(sum(r["elapsed_s"] for r in rows), 2),
        "avg_elapsed_s": round(sum(r["elapsed_s"] for r in rows) / n, 2),
        "avg_steps": round(sum(r["steps"] for r in rows) / n, 1),
        "total_input_tokens": sum(r["input_tokens"] for r in rows),
        "total_output_tokens": sum(r["output_tokens"] for r in rows),
        "avg_memory_delta_mb": round(sum(r["memory_delta_mb"] for r in rows) / n, 1),
        "errors": sum(1 for r in rows if r["error"]),
    }


# ── Compare ───────────────────────────────────────────────────────────────────


def _compare(file_a: str, file_b: str) -> None:
    a = json.loads(Path(file_a).read_text(encoding="utf-8"))
    b = json.loads(Path(file_b).read_text(encoding="utf-8"))

    host_a = a.get("hostname", file_a)
    host_b = b.get("hostname", file_b)

    rows_a = {r["id"]: r for r in a.get("questions", [])}
    rows_b = {r["id"]: r for r in b.get("questions", [])}

    ids = sorted(set(rows_a) | set(rows_b))

    print(f"\n{'=' * 72}")
    print(f"  {'ID':<18} {'Elapsed A':>10} {'Elapsed B':>10} {'Delta':>8} {'Steps A':>8} {'Steps B':>8}")
    print("=" * 72)
    for qid in ids:
        ra = rows_a.get(qid, {})
        rb = rows_b.get(qid, {})
        ea = ra.get("elapsed_s", 0.0)
        eb = rb.get("elapsed_s", 0.0)
        delta = eb - ea
        sa = ra.get("steps", 0)
        sb = rb.get("steps", 0)
        print(f"  {qid:<18} {ea:>10.2f} {eb:>10.2f} {delta:>+8.2f} {sa:>8} {sb:>8}")
    print("=" * 72)

    sa = a.get("summary", {})
    sb = b.get("summary", {})
    print(f"\n  {'':22} {host_a:>16} {host_b:>16}")
    print(f"  {'avg_elapsed_s':<22} {sa.get('avg_elapsed_s', 0):>16.2f} {sb.get('avg_elapsed_s', 0):>16.2f}")
    print(f"  {'avg_steps':<22} {sa.get('avg_steps', 0):>16.1f} {sb.get('avg_steps', 0):>16.1f}")
    print(
        f"  {'total_input_tokens':<22} {sa.get('total_input_tokens', 0):>16,} {sb.get('total_input_tokens', 0):>16,}"
    )
    print(
        f"  {'mcp_cold_start_s':<22} {a.get('mcp_cold_start_s', 0):>16.2f} {b.get('mcp_cold_start_s', 0):>16.2f}"
    )
    print()


# ── Entry point ───────────────────────────────────────────────────────────────


async def _main_async(args: argparse.Namespace) -> None:
    print(f"Running benchmark on {socket.gethostname()} (model={FAST_MODEL})...\n")
    data = await run_bench()

    out_dir = _ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"bench-{data['hostname']}.json"
    out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved to {out_path}")

    s = data["summary"]
    print(f"\n  Questions : {s['n']}")
    print(f"  Avg time  : {s['avg_elapsed_s']:.2f}s")
    print(f"  Avg steps : {s['avg_steps']}")
    print(f"  MCP start : {data['mcp_cold_start_s']:.2f}s")
    print(f"  Tokens    : {s['total_input_tokens']:,} in / {s['total_output_tokens']:,} out")
    if s["errors"]:
        print(f"  Errors    : {s['errors']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Agent benchmark")
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("FILE_A", "FILE_B"),
        help="Compare two saved bench results",
    )
    args = parser.parse_args()

    if args.compare:
        _compare(args.compare[0], args.compare[1])
        return

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    asyncio.run(_main_async(args))


if __name__ == "__main__":
    main()
