"""Chapter 4 GAIA benchmark (block 8): Agent-loop evaluation.

Runs 20 fixed level-1 tasks (same task_ids as ch02 baseline) with four
configurations: (FAST|SMART) x (with_tools|no_tools).

Usage:
    uv run python experiments/ch04_gaia.py --limit 3   # smoke test
    uv run python experiments/ch04_gaia.py             # full 20-task run

Outputs:
    - Console table with accuracy / steps / tokens / cost / delta vs baseline
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
from agentkit.config import FAST_MODEL, SMART_MODEL
from agentkit.gaia import GaiaOutput, is_correct, load_level1
from agentkit.llm import LlmClient
from agentkit.mcp_client import McpToolset
from agentkit.prompts import GAIA_AGENT_PROMPT
from agentkit.tools.base import BaseTool
from agentkit.tools.mcp import load_mcp_tools

# ── Fixed task IDs from ch02 baseline (do not change — ensures comparability) ──
BASELINE_TASK_IDS: list[str] = [
    "e1fc63a2-da7a-432f-be78-7c4a95598703",
    "8e867cd7-cff9-4e6c-867a-ff5ddc2550be",
    "ec09fa32-d03f-4bf8-84b0-1f16922c3ae4",
    "5d0080cb-90d7-4712-bc33-848150e917d3",
    "a1e91b78-d3d8-4675-bb8d-62741b4b68a6",
    "46719c30-f4c3-4cad-be07-d5cb21eee6bb",
    "4b6bb5f7-f634-410e-815d-e673ab7f8632",
    "cffe0e32-c9a6-4c52-9877-78ceb4aaa9fb",
    "2d83110e-a098-4ebb-9987-066c06fa42d0",
    "5cfb274c-0207-4aa7-9575-6ac0bd95d9b2",
    "27d5d136-8563-469e-92bf-fd103c28b57c",
    "dc28cf18-6431-458b-83ef-64b3ce566c10",
    "b816bfce-3d80-4913-a07d-69b752ce6377",
    "72e110e7-464c-453c-a309-90a95aed6538",
    "42576abe-0deb-4869-8c63-225c2d75a95a",
    "b415aba4-4b68-4fc6-9b89-2c812e55a3e1",
    "cca530fc-4052-43b2-b130-b30968d8aa44",
    "935e2cff-ae78-4218-b3f5-115589b19dae",
    "4fc2f1ae-8625-45b5-ab34-ad4433bc21f8",
    "5188369a-3bbe-43d8-8b94-11558f909a08",
]

MCP_CMD = ("uv", ["run", "python", "-m", "agentkit.servers.tavily_server"])
RESULTS_DIR = Path(__file__).parent.parent / "results"
TRACES_DIR = RESULTS_DIR / "ch04_traces"
BASELINE_PATH = RESULTS_DIR / "ch02_baseline.json"
MAX_STEPS = 15
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
    # treat error/max_steps outputs as non-answers
    return None if text.startswith("[") else text or None


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

    output = result.output if result else None
    prediction = _extract_prediction(output)
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
        "hit_max_steps": steps >= MAX_STEPS,
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
) -> list[dict[str, Any]]:
    agent = Agent(
        model=LlmClient(model_name),
        tools=tools,
        instructions=GAIA_AGENT_PROMPT,
        max_steps=MAX_STEPS,
        output_type=GaiaOutput,
        name=config_name,
    )
    tasks = [
        _evaluate_task(p, agent, semaphore, config_name) for p in problems
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


def _load_baseline() -> dict[str, float]:
    """Return {task_id: correct} from ch02 baseline for haiku only."""
    if not BASELINE_PATH.exists():
        return {}
    data = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    return {
        r["task_id"]: r["correct"]
        for r in data
        if r.get("model") == FAST_MODEL
    }


def _print_table(
    all_summaries: list[tuple[str, str, dict]],
    baseline_acc: float | None,
) -> None:
    print(f"\n{'=' * 85}")
    print(
        f"  {'Config':<30} {'Acc':>6} {'Delta':>7} {'Steps':>6} "
        f"{'In-Tok':>8} {'Out-Tok':>8} {'Cost$':>7} {'MaxSt':>6}"
    )
    print("=" * 85)
    for config_name, model, s in all_summaries:
        if not s:
            continue
        acc = s["accuracy"]
        delta = f"{acc - baseline_acc:+.1%}" if baseline_acc is not None else "  n/a"
        cost = _estimate_cost(s, model)
        print(
            f"  {config_name:<30} {acc:>6.1%} {delta:>7} {s['avg_steps']:>6.1f} "
            f"{s['total_input_tokens']:>8,} {s['total_output_tokens']:>8,} "
            f"{cost:>7.3f} {s['hit_max_steps']:>6}"
        )
    print("=" * 85)
    if baseline_acc is not None:
        print(f"  Baseline (ch02, haiku, no tools): {baseline_acc:.1%}")


# ── Main ───────────────────────────────────────────────────────────────────────


async def main(limit: int | None) -> None:
    semaphore = asyncio.Semaphore(AGENT_CONCURRENCY)

    # Load and filter problems
    print("Loading GAIA dataset...")
    all_problems = load_level1()
    target_ids = set(BASELINE_TASK_IDS[:limit] if limit else BASELINE_TASK_IDS)
    problems = [p for p in all_problems if p["task_id"] in target_ids]
    # preserve original ordering
    id_order = {tid: i for i, tid in enumerate(BASELINE_TASK_IDS)}
    problems.sort(key=lambda p: id_order.get(p["task_id"], 999))
    print(f"Running {len(problems)} tasks ({limit=}).\n")

    # Baseline accuracy (haiku only, for delta column)
    baseline_map = _load_baseline()
    if baseline_map and problems:
        baseline_correct = sum(baseline_map.get(p["task_id"], False) for p in problems)
        baseline_acc: float | None = baseline_correct / len(problems)
    else:
        baseline_acc = None

    all_results: list[tuple[str, str, dict]] = []

    # ── with-tools configs (both models share one MCP session) ─────────────────
    async with McpToolset(*MCP_CMD) as ts:
        mcp_tools = load_mcp_tools(ts)

        for model_name in [FAST_MODEL, SMART_MODEL]:
            short = model_name.split("/")[-1]
            config_name = f"{short}__with_tools"
            rows = await _run_config(config_name, model_name, mcp_tools, problems, semaphore)
            s = _summarise(rows)
            all_results.append((config_name, model_name, s))

    # ── no-tools configs ────────────────────────────────────────────────────────
    for model_name in [FAST_MODEL, SMART_MODEL]:
        short = model_name.split("/")[-1]
        config_name = f"{short}__no_tools"
        rows = await _run_config(config_name, model_name, [], problems, semaphore)
        s = _summarise(rows)
        all_results.append((config_name, model_name, s))

    _print_table(all_results, baseline_acc)

    # Cost projection for full 20-task run (only shown when running with --limit)
    if limit and limit < 20:
        print("\n  Cost projection for full 20 tasks:")
        scale = 20 / limit
        for config_name, model_name, s in all_results:
            if not s:
                continue
            projected = _estimate_cost(s, model_name) * scale
            print(f"    {config_name:<35} ${projected:.2f}")

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N tasks from the baseline set (default: all 20)",
    )
    args = parser.parse_args()
    asyncio.run(main(args.limit))
