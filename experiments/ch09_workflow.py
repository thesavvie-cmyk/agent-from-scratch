"""Chapter 9 multi-agent workflow experiments (block 20).

Sections
--------
a) Blog pipeline: researcher → coder → writer → reviewer.
   Print what passes between steps and each agent's token cost.
b) Monolith comparison: same result via one agent with all roles in prompt.
   Table: total tokens, time, steps.  Both outputs printed in full.
c) Context isolation: same pipeline with share_context=True vs False.
   Show tokens seen by the last agent in each case.
d) Parallel researchers: three agents on different aspects, then merge.
   Compare elapsed time vs sequential.
e) Review loop: writer drafts, reviewer checks, writer revises if needed.
   Up to 2 iterations. Show whether the text improves.

Usage
-----
    uv run python experiments/ch09_workflow.py --section a
    uv run python experiments/ch09_workflow.py --section all

Requires ANTHROPIC_API_KEY + TAVILY_API_KEY (sections a–e).
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

from agentkit.agents.specialists import (
    make_researcher,
    make_reviewer,
    make_writer,
)
from agentkit.config import FAST_MODEL, find_uv
from agentkit.llm import LlmClient
from agentkit.mcp_client import McpToolset
from agentkit.tokens import count_tokens
from agentkit.tools.mcp import load_mcp_tools
from agentkit.types import ToolCall
from agentkit.workflow import SequentialWorkflow, WorkflowResult, WorkflowStep

MCP_CMD = (find_uv(), ["run", "python", "-m", "agentkit.servers.tavily_server"])
TOPIC = "how Python async/await works under the hood"


def _hr(title: str) -> None:
    print(f"\n{'=' * 68}\n  {title}\n{'=' * 68}")


def _print_workflow_trace(result: WorkflowResult, show_tokens: bool = True) -> None:
    """Print per-step summary: agent name, tokens, output preview."""
    tps = result.tokens_per_step()
    for i, (sr, tok) in enumerate(zip(result.step_results, tps)):
        name = sr.context.events[0].author if sr.context.events else f"step-{i}"
        out_preview = str(sr.output)[:120].replace("\n", " ")
        tool_calls = sum(
            1
            for ev in sr.context.events
            for item in ev.content
            if isinstance(item, ToolCall)
        )
        err = f"  ERROR: {sr.error}" if sr.error else ""
        tok_str = f"  in={tok['input_tokens']} out={tok['output_tokens']}" if show_tokens else ""
        print(
            f"\n  Step {i + 1} [{name}]  tools={tool_calls}{tok_str}{err}"
            f"\n    → {out_preview}…"
        )


# ── Section A — Blog pipeline ──────────────────────────────────────────────────


async def section_a(model: LlmClient, search_tools: list) -> None:
    _hr("Section A — Blog pipeline: researcher → writer → reviewer")

    wf = SequentialWorkflow([
        WorkflowStep(make_researcher(model, tools=search_tools)),
        WorkflowStep(make_writer(model)),
        WorkflowStep(make_reviewer(model)),
    ])

    t0 = time.perf_counter()
    result = await wf.run(TOPIC)
    elapsed = time.perf_counter() - t0

    _print_workflow_trace(result)
    total = result.total_tokens()
    print(
        f"\n  Total: steps={len(result.step_results)}  "
        f"in={total['input_tokens']}  out={total['output_tokens']}  "
        f"elapsed={elapsed:.1f}s"
    )
    print(f"\n  Authors in trace: {sorted({e.author for e in result.all_events})}")

    print(f"\n  === FINAL OUTPUT ===\n{result.output}")


# ── Section B — Monolith comparison ────────────────────────────────────────────


async def section_b(model: LlmClient, search_tools: list) -> None:
    _hr("Section B — Pipeline vs monolith comparison")

    from agentkit.agent import Agent

    monolith = Agent(
        model=model,
        tools=search_tools,
        name="monolith",
        instructions=(
            "You are a researcher, coder, writer, and reviewer combined.\n"
            "For the given topic: search for information, then write a complete, "
            "well-structured article. Review your own output for quality.\n"
            "Produce a polished final article."
        ),
        max_steps=12,
    )

    pipeline = SequentialWorkflow([
        WorkflowStep(make_researcher(model, tools=search_tools)),
        WorkflowStep(make_writer(model)),
        WorkflowStep(make_reviewer(model)),
    ])

    # Run monolith
    t0 = time.perf_counter()
    mono_result = await monolith.run(TOPIC)
    mono_elapsed = time.perf_counter() - t0
    mono_u = mono_result.context.state.get("token_usage", {})

    # Run pipeline
    t0 = time.perf_counter()
    pipe_result = await pipeline.run(TOPIC)
    pipe_elapsed = time.perf_counter() - t0
    pipe_u = pipe_result.total_tokens()

    print(f"\n  {'':20} {'Tokens in':>10} {'Tokens out':>11} {'Steps':>6} {'Time':>8}")
    print(f"  {'-'*20} {'-'*10} {'-'*11} {'-'*6} {'-'*8}")
    print(
        f"  {'Monolith':<20} {mono_u.get('input_tokens', 0):>10,} "
        f"{mono_u.get('output_tokens', 0):>11,} "
        f"{mono_result.context.current_step:>6}  {mono_elapsed:>6.1f}s"
    )
    print(
        f"  {'Pipeline':<20} {pipe_u['input_tokens']:>10,} "
        f"{pipe_u['output_tokens']:>11,} "
        f"{'—':>6}  {pipe_elapsed:>6.1f}s"
    )

    print("\n\n  === MONOLITH OUTPUT ===")
    print(mono_result.output)
    print("\n\n  === PIPELINE OUTPUT ===")
    print(pipe_result.output)


# ── Section C — Context isolation ─────────────────────────────────────────────


async def section_c(model: LlmClient, search_tools: list) -> None:
    _hr("Section C — Context isolation: share_context=True vs False")

    async def _run_pipeline(share: bool) -> WorkflowResult:
        steps = [
            WorkflowStep(make_researcher(model, tools=search_tools), share_context=share),
            WorkflowStep(make_writer(model), share_context=share),
            WorkflowStep(make_reviewer(model), share_context=share),
        ]
        return await SequentialWorkflow(steps).run(TOPIC)

    t0 = time.perf_counter()
    no_share = await _run_pipeline(share=False)
    t1 = time.perf_counter()
    with_share = await _run_pipeline(share=True)
    t2 = time.perf_counter()

    def _last_agent_tokens(wr: WorkflowResult) -> dict[str, int]:
        return wr.tokens_per_step()[-1]

    no_last = _last_agent_tokens(no_share)
    sh_last = _last_agent_tokens(with_share)

    print(f"\n  {'':30} {'share=False':>12} {'share=True':>12}")
    print(f"  {'-'*30} {'-'*12} {'-'*12}")
    print(
        f"  {'Last agent input tokens':<30} "
        f"{no_last['input_tokens']:>12,} "
        f"{sh_last['input_tokens']:>12,}"
    )
    print(
        f"  {'Last agent output tokens':<30} "
        f"{no_last['output_tokens']:>12,} "
        f"{sh_last['output_tokens']:>12,}"
    )
    print(
        f"  {'Total events in trace':<30} "
        f"{len(no_share.all_events):>12} "
        f"{len(with_share.all_events):>12}"
    )
    print(f"  {'Elapsed':<30} {t1-t0:>11.1f}s {t2-t1:>11.1f}s")
    print(
        "\n  Interpretation: share=True last agent sees the full conversation "
        "history → higher input token cost."
    )


# ── Section D — Parallel researchers ──────────────────────────────────────────


async def section_d(model: LlmClient, search_tools: list) -> None:
    _hr("Section D — Parallel researchers vs sequential")

    aspects = [
        f"the event loop and coroutine scheduler in {TOPIC}",
        f"practical performance implications of {TOPIC}",
        f"common pitfalls and debugging tips for {TOPIC}",
    ]

    def merge(outputs: list[str]) -> str:
        sections = ["# Research Summary\n"]
        titles = ["## Event Loop & Scheduler", "## Performance", "## Pitfalls & Tips"]
        for title, out in zip(titles, outputs):
            sections.append(f"{title}\n\n{out}")
        return "\n\n".join(sections)

    # ParallelWorkflow sends the same input to all; for different per-step inputs
    # we run them directly with asyncio.gather.
    t0 = time.perf_counter()
    tasks = [make_researcher(model, tools=search_tools).run(q) for q in aspects]
    raw = await asyncio.gather(*tasks, return_exceptions=True)
    par_elapsed = time.perf_counter() - t0

    outputs = [str(r.output) if not isinstance(r, Exception) else "" for r in raw]
    merged = merge(outputs)

    # Sequential (same three queries, one after another)
    t0 = time.perf_counter()
    seq_outputs: list[str] = []
    for q in aspects:
        r = await make_researcher(model, tools=search_tools).run(q)
        seq_outputs.append(str(r.output))
    seq_elapsed = time.perf_counter() - t0

    print(f"\n  Parallel elapsed:   {par_elapsed:.1f}s")
    print(f"  Sequential elapsed: {seq_elapsed:.1f}s")
    print(f"  Speed-up:           {seq_elapsed / par_elapsed:.1f}×")
    print(f"\n  Merged output ({count_tokens(merged)} tokens):")
    print(f"  {merged[:300]}…")


# ── Section E — Review loop ────────────────────────────────────────────────────


async def section_e(model: LlmClient) -> None:
    _hr("Section E — Writer/reviewer loop (up to 2 revisions)")

    writer = make_writer(model)
    reviewer = make_reviewer(model)

    drafts: list[str] = []
    reviews: list[str] = []

    # Iteration 0: writer creates initial draft
    research_summary = (
        f"Topic: {TOPIC}\n"
        "Key points: coroutines are paused functions; the event loop schedules "
        "them; await yields control; asyncio.gather runs tasks concurrently; "
        "avoid blocking calls inside async functions."
    )

    print("\n  Iter 0: writer produces initial draft…")
    draft_result = await writer.run(
        f"Write a 3-paragraph article based on this research:\n{research_summary}"
    )
    draft = str(draft_result.output)
    drafts.append(draft)
    in_tok = draft_result.context.state.get("token_usage", {}).get("input_tokens", 0)
    print(f"  Draft tokens in={in_tok}  length={len(draft)}")

    for iteration in range(1, 3):  # up to 2 review/rewrite cycles
        print(f"\n  Iter {iteration}: reviewer evaluates…")
        review_result = await reviewer.run(draft)
        review = str(review_result.output)
        reviews.append(review)
        print(f"  Review: {review[:120]}")

        if review.upper().startswith("APPROVED"):
            print("  → Approved. Stopping.")
            break

        # Writer revises
        print(f"  Iter {iteration}: writer revises based on feedback…")
        revise_result = await writer.run(
            f"Revise the following article based on this feedback:\n"
            f"FEEDBACK: {review}\n\nARTICLE:\n{draft}"
        )
        draft = str(revise_result.output)
        drafts.append(draft)
        in_tok = revise_result.context.state.get("token_usage", {}).get("input_tokens", 0)
        print(f"  Revised draft tokens in={in_tok}  length={len(draft)}")

    print(f"\n  Total drafts: {len(drafts)}  Total reviews: {len(reviews)}")
    print("\n  === DRAFT 0 ===")
    print(drafts[0])
    if len(drafts) > 1:
        print(f"\n  === DRAFT {len(drafts) - 1} (final) ===")
        print(drafts[-1])


# ── Main ───────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Block 20 workflow experiments")
    parser.add_argument(
        "--section", default="a",
        choices=["a", "b", "c", "d", "e", "all"],
    )
    return parser.parse_args()


async def _main(args: argparse.Namespace) -> None:
    async with McpToolset(*MCP_CMD) as ts:
        search_tools = list(load_mcp_tools(ts))

    model = LlmClient(FAST_MODEL)

    if args.section in ("a", "all"):
        await section_a(model, search_tools)
    if args.section in ("b", "all"):
        await section_b(model, search_tools)
    if args.section in ("c", "all"):
        await section_c(model, search_tools)
    if args.section in ("d", "all"):
        await section_d(model, search_tools)
    if args.section in ("e", "all"):
        await section_e(model)


def main() -> None:
    args = _parse_args()
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try:
        asyncio.run(_main(args))
    except KeyboardInterrupt:
        print("\n[interrupted]")
        sys.exit(1)
    print()


if __name__ == "__main__":
    main()
