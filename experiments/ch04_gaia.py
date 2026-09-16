"""Chapter 4 GAIA benchmark (block 8): Agent-loop evaluation.

Task selection
--------------
Uses load_search_tasks() from gaia.py — level-1 tasks where the annotator
used a web browser / search engine AND there is no file attachment and no
video/PDF/image parsing required.  This gives 20 tasks where a web-search
agent can plausibly outperform a tool-less baseline.

Configurations
--------------
  (FAST_MODEL | SMART_MODEL) x (with_tools | no_tools)

Baseline comparison
-------------------
The ch02 baseline used a direct response_format API call (no agent loop).
That is a different evaluation methodology, so ch02 numbers are shown as a
reference footnote only.  The internal baseline is the no_tools config of
the same run: it uses the identical agent code and prompt, differs only in
whether search tools are available.

Usage:
    uv run python experiments/ch04_gaia.py --limit 3   # smoke test
    uv run python experiments/ch04_gaia.py             # full 20-task run

Outputs:
    - Console table with accuracy / steps / tokens / cost / delta vs no_tools
    - Per-run traces saved to results/ch04_traces/
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
from agentkit.config import FAST_MODEL, SMART_MODEL, find_uv
from agentkit.gaia import GaiaOutput, is_correct, load_search_tasks
from agentkit.llm import LlmClient
from agentkit.mcp_client import McpToolset
from agentkit.prompts import GAIA_AGENT_PROMPT
from agentkit.tools.base import BaseTool
from agentkit.tools.mcp import load_mcp_tools

MCP_CMD = (find_uv(), ["run", "python", "-m", "agentkit.servers.tavily_server"])
RESULTS_DIR = Path(__file__).parent.parent / "results"
TRACES_DIR = RESULTS_DIR / "ch04_traces"
MAX_STEPS = 8
AGENT_CONCURRENCY = 3

# Approximate cost per million tokens (USD) — used for estimation only
_COST: dict[str, dict[str, float]] = {
    FAST_MODEL: {"input": 0.80, "output": 4.00},
    SMART_MODEL: {"input": 3.00, "output": 15.00},
}


# ── Evaluation helpers ─────────────────────────────────────────────────────────


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
    max_steps: int,
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
            "events": _serialize_context(result.context),
        }
        trace_path.write_text(
            json.dumps(trace_data, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    return {
        "task_id": task_id,
        "config": config_name,
        "correct": is_correct(prediction, answer),
        "prediction": prediction,
        "answer": answer,
        "steps": steps,
        "hit_max_steps": steps >= max_steps,
        "input_tokens": input_tok,
        "output_tokens": output_tok,
        "latency_s": round(latency, 2),
        "error": error,
    }


async def _run_config(
    config_name: str,
    model_name: str,
    tools: list[BaseTool],
    problems: list[dict[str, Any]],
    semaphore: asyncio.Semaphore,
    max_steps: int,
) -> list[dict[str, Any]]:
    agent = Agent(
        model=LlmClient(model_name),
        tools=tools,
        instructions=GAIA_AGENT_PROMPT,
        max_steps=max_steps,
        output_type=GaiaOutput,
        name=config_name,
    )
    tasks = [
        _evaluate_task(p, agent, semaphore, config_name, max_steps) for p in problems
    ]
    return await tqdm_asyncio.gather(*tasks, desc=config_name)


# ── Reporting ──────────────────────────────────────────────────────────────────


def _summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {}
    correct = sum(r["correct"] for r in rows)
    errors = sum(1 for r in rows if r["error"])
    hit_max = sum(r["hit_max_steps"] for r in rows)
    total_in = sum(r["input_tokens"] for r in rows)
    total_out = sum(r["output_tokens"] for r in rows)
    avg_steps = sum(r["steps"] for r in rows) / n
    return {
        "n": n,
        "correct": correct,
        "accuracy": correct / n,
        "avg_steps": round(avg_steps, 1),
        "total_input_tokens": total_in,
        "total_output_tokens": total_out,
        "hit_max_steps": hit_max,
        "errors": errors,
    }


def _estimate_cost(summary: dict, model: str) -> float:
    rates = _COST.get(model, {"input": 1.0, "output": 5.0})
    return (
        summary["total_input_tokens"] * rates["input"]
        + summary["total_output_tokens"] * rates["output"]
    ) / 1_000_000


def _print_table(
    all_summaries: list[tuple[str, str, dict]],
    internal_baseline: float | None,
) -> None:
    """Print results table.

    Delta column is relative to haiku+no_tools from the SAME run — the only
    valid internal comparison (same agent code, same prompt, same task set).
    ch02 numbers are printed as a separate footnote since they used a
    different evaluation methodology (direct response_format, no agent loop).
    """
    print(f"\n{'=' * 87}")
    print(
        f"  {'Config':<32} {'Acc':>6} {'Delta*':>7} {'Steps':>6} "
        f"{'In-Tok':>8} {'Out-Tok':>8} {'Cost$':>7} {'MaxSt':>6}"
    )
    print("=" * 87)
    for config_name, model, s in all_summaries:
        if not s:
            continue
        acc = s["accuracy"]
        delta = (
            f"{acc - internal_baseline:+.1%}"
            if internal_baseline is not None
            else "  n/a"
        )
        cost = _estimate_cost(s, model)
        print(
            f"  {config_name:<32} {acc:>6.1%} {delta:>7} {s['avg_steps']:>6.1f} "
            f"{s['total_input_tokens']:>8,} {s['total_output_tokens']:>8,} "
            f"{cost:>7.3f} {s['hit_max_steps']:>6}"
        )
    print("=" * 87)
    if internal_baseline is not None:
        print("  * Delta vs haiku+no_tools (same run, same methodology)")


# ── Main ───────────────────────────────────────────────────────────────────────


async def main(limit: int | None, max_steps: int) -> None:
    semaphore = asyncio.Semaphore(AGENT_CONCURRENCY)

    # Load search-only tasks (web browser/search engine, no file attachments)
    print("Loading GAIA search tasks...")
    problems = load_search_tasks(limit=limit)
    print(f"Running {len(problems)} tasks (max_steps={max_steps}).\n")

    all_results: list[tuple[str, str, dict]] = []
    all_rows_by_config: dict[str, list[dict]] = {}

    # ── with-tools configs (both models share one MCP session) ─────────────────
    async with McpToolset(*MCP_CMD) as ts:
        mcp_tools = load_mcp_tools(ts)

        for model_name in [FAST_MODEL, SMART_MODEL]:
            short = model_name.split("/")[-1]
            config_name = f"{short}__with_tools"
            rows = await _run_config(
                config_name, model_name, mcp_tools, problems, semaphore, max_steps
            )
            s = _summarise(rows)
            all_results.append((config_name, model_name, s))
            all_rows_by_config[config_name] = rows

    # ── no-tools configs ────────────────────────────────────────────────────────
    for model_name in [FAST_MODEL, SMART_MODEL]:
        short = model_name.split("/")[-1]
        config_name = f"{short}__no_tools"
        rows = await _run_config(
            config_name, model_name, [], problems, semaphore, max_steps
        )
        s = _summarise(rows)
        all_results.append((config_name, model_name, s))
        all_rows_by_config[config_name] = rows

    # Internal baseline = haiku + no_tools (same methodology as all other configs)
    haiku_short = FAST_MODEL.split("/")[-1]
    baseline_key = f"{haiku_short}__no_tools"
    baseline_s = next((s for cn, _, s in all_results if cn == baseline_key), {})
    internal_baseline: float | None = baseline_s.get("accuracy") if baseline_s else None

    _print_table(all_results, internal_baseline)

    # Cost projection for full 20-task run (only shown when running with --limit)
    if limit and limit < 20:
        print(f"\n  Cost projection for full 20 tasks (max_steps={max_steps}):")
        scale = 20 / limit
        for config_name, model_name, s in all_results:
            if not s:
                continue
            projected = _estimate_cost(s, model_name) * scale
            print(f"    {config_name:<38} ${projected:.2f}")

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N search tasks (default: all 20)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=MAX_STEPS,
        help=f"Max agent steps per task (default: {MAX_STEPS})",
    )
    args = parser.parse_args()
    asyncio.run(main(args.limit, args.max_steps))
