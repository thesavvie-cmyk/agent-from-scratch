"""Chapter 6 context management experiments (block 12).

Sections
--------
a) Context growth analysis: load ch04_traces and visualise token growth
b) TruncateOldToolResults: show before/after token counts at each step
c) SummarizeHistory: run agent with summarisation, print the summary message
d) API validity check: verify compacted context does not break Anthropic API

Usage
-----
    uv run python experiments/ch06_context.py --section a
    uv run python experiments/ch06_context.py --section all

Requires ANTHROPIC_API_KEY for sections c and d.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

TRACES_DIR = Path(__file__).parent.parent / "results" / "ch04_traces"


# ── helpers ────────────────────────────────────────────────────────────────────


def _hr(title: str) -> None:
    print(f"\n{'─' * 60}\n  {title}\n{'─' * 60}")


def _load_traces() -> list[dict[str, Any]]:
    traces = []
    for p in sorted(TRACES_DIR.glob("*.json")):
        d = json.loads(p.read_bytes().decode("utf-8"))
        d["_file"] = p.name
        traces.append(d)
    return traces


def _rebuild_contents(trace: dict[str, Any]) -> list[Any]:
    """Rebuild ContentItem list from a serialised trace."""
    from agentkit.types import Message, ToolCall, ToolResult

    contents: list[Any] = []
    for event in trace.get("events", []):
        for item in event.get("content", []):
            t = item.get("type")
            if t == "message":
                contents.append(Message(**item))
            elif t == "tool_call":
                contents.append(ToolCall(**item))
            elif t == "tool_result":
                contents.append(ToolResult(**item))
    return contents


# ── Section A — context growth analysis ───────────────────────────────────────


def section_a() -> None:
    _hr("Section A — Context growth analysis from ch04_traces")

    traces = _load_traces()
    # Only with_tools traces have interesting growth
    wt = [t for t in traces if "with_tools" in t.get("_file", "")]

    print(f"\n  Loaded {len(wt)} with_tools traces\n")

    # Summary stats
    steps_dist: dict[int, int] = {}
    for t in wt:
        s = t.get("steps", 0)
        steps_dist[s] = steps_dist.get(s, 0) + 1

    print("  Step distribution:")
    for s in sorted(steps_dist):
        bar = "█" * steps_dist[s]
        print(f"    {s:2d} steps: {steps_dist[s]:3d}  {bar}")

    # Top-5 deepest traces
    top5 = sorted(wt, key=lambda t: t.get("input_tokens", 0), reverse=True)[:5]
    print("\n  Top-5 by input tokens:")
    print(f"  {'File':<55} {'Steps':>5} {'In tok':>8} {'Correct':>8}")
    print(f"  {'─' * 55} {'─' * 5} {'─' * 8} {'─' * 8}")
    for t in top5:
        fname = t["_file"][:54]
        print(
            f"  {fname:<55} {t.get('steps', 0):>5} "
            f"{t.get('input_tokens', 0):>8} {t.get('correct', '?')!s:>8}"
        )

    # Token growth per step in the deepest trace
    deepest = top5[0]
    print(f"\n  Deepest trace: {deepest['_file']}")
    print(f"  Question: {deepest.get('question', '')[:80]}")
    from agentkit.memory.budget import estimate_context_tokens

    # Simulate step-by-step token growth
    # Events alternate: user, think, tool_results, think, tool_results, ...
    cumulative: list[tuple[int, int]] = []
    running: list[Any] = []
    for event in deepest.get("events", []):
        for item in event.get("content", []):
            from agentkit.types import Message, ToolCall, ToolResult
            t_type = item.get("type")
            if t_type == "message":
                running.append(Message(**item))
            elif t_type == "tool_call":
                running.append(ToolCall(**item))
            elif t_type == "tool_result":
                running.append(ToolResult(**item))
        tok = estimate_context_tokens(running)
        cumulative.append((len(running), tok))

    print("\n  Token growth (event cumulative):")
    print(f"  {'Event':>6} {'Items':>6} {'Est tokens':>12}")
    print(f"  {'─' * 6} {'─' * 6} {'─' * 12}")
    for idx, (items, tok) in enumerate(cumulative):
        print(f"  {idx + 1:>6} {items:>6} {tok:>12,}")


# ── Section B — TruncateOldToolResults ────────────────────────────────────────


async def section_b() -> None:
    _hr("Section B — TruncateOldToolResults: before/after comparison")

    traces = [
        t for t in _load_traces()
        if "with_tools" in t.get("_file", "") and t.get("steps", 0) >= 4
    ]
    if not traces:
        print("  No deep traces found — run ch04_gaia.py first")
        return

    from agentkit.context import ExecutionContext
    from agentkit.memory.budget import estimate_context_tokens
    from agentkit.memory.compaction import TruncateOldToolResults

    trace = max(traces, key=lambda t: t.get("input_tokens", 0))
    contents = _rebuild_contents(trace)

    print(f"\n  Trace: {trace['_file']}")
    print(f"  Steps: {trace.get('steps')}  Input tokens (real): {trace.get('input_tokens'):,}")
    print(f"  Items in context: {len(contents)}")

    ctx = ExecutionContext()
    print(f"\n  {'keep_recent':>12} {'Before tok':>12} {'After tok':>10} {'Saved':>8} {'Saved%':>7}")
    print(f"  {'─' * 12} {'─' * 12} {'─' * 10} {'─' * 8} {'─' * 7}")

    before = estimate_context_tokens(contents)
    for k in [1, 2, 3, 5]:
        strategy = TruncateOldToolResults(keep_recent=k)
        compacted = await strategy.apply(contents, ctx)
        after = estimate_context_tokens(compacted)
        saved = before - after
        pct = 100 * saved / before if before else 0
        print(f"  {k:>12} {before:>12,} {after:>10,} {saved:>8,} {pct:>6.1f}%")


# ── Section C — SummarizeHistory ──────────────────────────────────────────────


async def section_c() -> None:
    _hr("Section C — SummarizeHistory: live summary call")

    import os

    from agentkit.config import FAST_MODEL
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("  [SKIP] ANTHROPIC_API_KEY not set")
        return

    traces = [
        t for t in _load_traces()
        if "with_tools" in t.get("_file", "") and t.get("steps", 0) >= 4
    ]
    if not traces:
        print("  No deep traces found — run ch04_gaia.py first")
        return

    from agentkit.context import ExecutionContext
    from agentkit.llm import LlmClient
    from agentkit.memory.budget import estimate_context_tokens
    from agentkit.memory.compaction import SummarizeHistory

    trace = max(traces, key=lambda t: t.get("input_tokens", 0))
    contents = _rebuild_contents(trace)

    print(f"\n  Trace: {trace['_file']}")
    print(f"  Steps: {trace.get('steps')}  Items: {len(contents)}")
    before = estimate_context_tokens(contents)
    print(f"  Estimated tokens before: {before:,}")

    llm = LlmClient(FAST_MODEL)
    # Force trigger (threshold = 0 to always apply)
    strategy = SummarizeHistory(llm_client=llm, keep_recent=3, trigger_tokens=0)
    ctx = ExecutionContext()
    compacted = await strategy.apply(contents, ctx)

    after = estimate_context_tokens(compacted)
    print(f"  Estimated tokens after:  {after:,}  (saved {before - after:,})")
    print(f"  Items in compacted: {len(compacted)}")

    # Show the summary message
    from agentkit.types import Message
    for item in compacted:
        if isinstance(item, Message) and "[History summary]" in item.content:
            print(f"\n  Summary message:\n  {item.content[:600]}")
            break


# ── Section D — API validity check ────────────────────────────────────────────


async def section_d() -> None:
    _hr("Section D — API validity: send compacted context to Anthropic")

    import os

    from agentkit.config import FAST_MODEL
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("  [SKIP] ANTHROPIC_API_KEY not set")
        return

    traces = [
        t for t in _load_traces()
        if "with_tools" in t.get("_file", "") and t.get("steps", 0) >= 3
    ]
    if not traces:
        print("  No suitable traces found — run ch04_gaia.py first")
        return

    from agentkit.context import ExecutionContext
    from agentkit.llm import LlmClient, LlmRequest
    from agentkit.memory.compaction import TruncateOldToolResults

    trace = max(traces, key=lambda t: t.get("steps", 0))
    contents = _rebuild_contents(trace)

    strategy = TruncateOldToolResults(keep_recent=2)
    ctx = ExecutionContext()
    compacted = await strategy.apply(contents, ctx)

    print(f"  Original items: {len(contents)}, compacted: {len(compacted)}")
    print("  Sending compacted context to API…")

    llm = LlmClient(FAST_MODEL)
    request = LlmRequest(contents=compacted)
    response = await llm.generate(request)

    if response.error_message:
        print(f"  [FAIL] API returned error: {response.error_message}")
    else:
        from agentkit.types import Message
        out = next(
            (i.content for i in response.content if isinstance(i, Message)), ""
        )
        print("  [OK] API accepted compacted context")
        print(f"  Response: {out[:200]}")


# ── main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Block 12 context management experiments")
    parser.add_argument(
        "--section",
        default="all",
        choices=["a", "b", "c", "d", "all"],
    )
    args = parser.parse_args()

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    async def _run_all() -> None:
        if args.section in ("a", "all"):
            section_a()
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
