"""Chapter 6 GAIA benchmark with context budget (block 12).

Compares two configurations on 20 GAIA tasks:
  - no_budget:   standard agent (no compaction)
  - with_budget: TruncateOldToolResults(keep_recent=3), max_tokens=30_000

Reports accuracy, steps, token usage, and compaction log summary.

Usage
-----
    uv run python experiments/ch06_gaia.py --limit 3   # smoke test
    uv run python experiments/ch06_gaia.py             # full 20-task run
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

from tqdm.asyncio import tqdm_asyncio

from agentkit.agent import Agent
from agentkit.config import FAST_MODEL, find_uv
from agentkit.gaia import GaiaOutput, is_correct, load_search_tasks
from agentkit.llm import LlmClient
from agentkit.mcp_client import McpToolset
from agentkit.memory.budget import ContextBudget
from agentkit.memory.compaction import TruncateOldToolResults
from agentkit.prompts import GAIA_AGENT_PROMPT
from agentkit.tools.mcp import load_mcp_tools

MCP_CMD = (find_uv(), ["run", "python", "-m", "agentkit.servers.tavily_server"])
RESULTS_DIR = Path(__file__).parent.parent / "results"
TRACES_DIR = RESULTS_DIR / "ch06_traces"
MAX_STEPS = 8
AGENT_CONCURRENCY = 3

_COST_PER_MTK = {"input": 0.80, "output": 4.00}  # haiku


# ── helpers ────────────────────────────────────────────────────────────────────


def _extract_prediction(output: Any) -> str | None:
    if isinstance(output, GaiaOutput):
        return output.final_answer if output.is_solvable else None
    text = str(output) if output else ""
    return None if text.startswith("[max_steps=") else text or None


def _serialize_context(ctx: Any) -> list[dict]:
    return [json.loads(e.model_dump_json()) for e in ctx.events]


async def _evaluate_task(
    problem: dict[str, Any],
    agent: Agent,
    semaphore: asyncio.Semaphore,
    config_name: str,
) -> dict[str, Any]:
    task_id: str = problem["task_id"]
    question: str = problem["Question"]
    answer: str = str(problem["Final answer"])

    start = time.perf_counter()
    error: str | None = None
    result = None

    async with semaphore:
        try:
            result = await agent.run(question)
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"

    latency = time.perf_counter() - start

    if result and result.error and not error:
        error = result.error
    output = result.output if result else None
    prediction = _extract_prediction(output) if (result and not result.error) else None
    token_usage = result.context.state.get("token_usage", {}) if result else {}
    compaction_log = result.context.state.get("compaction_log", []) if result else []
    input_tok = token_usage.get("input_tokens", 0)
    output_tok = token_usage.get("output_tokens", 0)
    steps = result.context.current_step if result else 0

    # Save trace
    if result:
        TRACES_DIR.mkdir(parents=True, exist_ok=True)
        trace_path = TRACES_DIR / f"{task_id}__{config_name}.json"
        trace_data = {
            "task_id": task_id,
            "config": config_name,
            "question": question,
            "answer": answer,
            "prediction": prediction,
            "correct": is_correct(prediction, answer),
            "steps": steps,
            "input_tokens": input_tok,
            "output_tokens": output_tok,
            "compaction_log": compaction_log,
            "events": _serialize_context(result.context),
        }
        trace_path.write_bytes(
            json.dumps(trace_data, indent=2, ensure_ascii=False).encode("utf-8")
        )

    return {
        "task_id": task_id,
        "config": config_name,
        "correct": is_correct(prediction, answer),
        "steps": steps,
        "hit_max_steps": steps >= MAX_STEPS,
        "input_tokens": input_tok,
        "output_tokens": output_tok,
        "latency_s": round(latency, 2),
        "compaction_count": len(compaction_log),
        "error": error,
    }


async def _run_config(
    config_name: str,
    tools: list[Any],
    problems: list[dict[str, Any]],
    semaphore: asyncio.Semaphore,
    context_budget: ContextBudget | None,
) -> list[dict[str, Any]]:
    agent = Agent(
        model=LlmClient(FAST_MODEL),
        tools=tools,
        instructions=GAIA_AGENT_PROMPT,
        max_steps=MAX_STEPS,
        output_type=GaiaOutput,
        name=config_name,
        context_budget=context_budget,
    )
    tasks = [
        _evaluate_task(p, agent, semaphore, config_name) for p in problems
    ]
    return await tqdm_asyncio.gather(*tasks, desc=config_name)


def _summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {}
    correct = sum(r["correct"] for r in rows)
    errors = sum(1 for r in rows if r["error"])
    total_in = sum(r["input_tokens"] for r in rows)
    total_out = sum(r["output_tokens"] for r in rows)
    avg_steps = sum(r["steps"] for r in rows) / n
    compactions = sum(r["compaction_count"] for r in rows)
    return {
        "n": n,
        "correct": correct,
        "accuracy": correct / n,
        "avg_steps": round(avg_steps, 1),
        "total_input_tokens": total_in,
        "total_output_tokens": total_out,
        "hit_max_steps": sum(r["hit_max_steps"] for r in rows),
        "errors": errors,
        "total_compactions": compactions,
    }


def _print_table(results: list[tuple[str, dict]]) -> None:
    print(f"\n{'=' * 90}")
    print(
        f"  {'Config':<22} {'Acc':>6} {'Steps':>6} {'In-Tok':>9} {'Out-Tok':>9} "
        f"{'Cost$':>7} {'MaxSt':>6} {'Compact':>8}"
    )
    print("=" * 90)
    for config_name, s in results:
        if not s:
            continue
        cost = (
            s["total_input_tokens"] * _COST_PER_MTK["input"]
            + s["total_output_tokens"] * _COST_PER_MTK["output"]
        ) / 1_000_000
        print(
            f"  {config_name:<22} {s['accuracy']:>6.1%} {s['avg_steps']:>6.1f} "
            f"{s['total_input_tokens']:>9,} {s['total_output_tokens']:>9,} "
            f"{cost:>7.3f} {s['hit_max_steps']:>6} {s['total_compactions']:>8}"
        )
    print("=" * 90)
    print("  Compact = number of compaction events fired across all tasks")


# ── main ───────────────────────────────────────────────────────────────────────


async def main(limit: int | None) -> None:
    semaphore = asyncio.Semaphore(AGENT_CONCURRENCY)

    print("Loading GAIA search tasks...")
    problems = load_search_tasks(limit=limit)
    print(f"Running {len(problems)} tasks (model={FAST_MODEL}, max_steps={MAX_STEPS}).\n")

    budget = ContextBudget(
        max_tokens=30_000,
        strategies=[TruncateOldToolResults(keep_recent=3)],
    )

    configs = [
        ("no_budget", None),
        ("with_budget", budget),
    ]

    all_results: list[tuple[str, dict]] = []

    async with McpToolset(*MCP_CMD) as ts:
        tools = load_mcp_tools(ts)

        for config_name, cb in configs:
            rows = await _run_config(config_name, tools, problems, semaphore, cb)
            s = _summarise(rows)
            all_results.append((config_name, s))

    _print_table(all_results)

    if limit and limit < 20:
        scale = 20 / limit
        print("\n  Cost projection for full 20 tasks:")
        for config_name, s in all_results:
            if not s:
                continue
            cost = (
                s["total_input_tokens"] * _COST_PER_MTK["input"]
                + s["total_output_tokens"] * _COST_PER_MTK["output"]
            ) / 1_000_000 * scale
            print(f"    {config_name:<22} ${cost:.2f}")

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(main(args.limit))
