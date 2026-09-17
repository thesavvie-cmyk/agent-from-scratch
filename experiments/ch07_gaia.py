"""Chapter 7 GAIA benchmark with planning (block 15).

Configurations
--------------
  haiku + tools          (baseline, reproduces block-8 haiku+with_tools)
  haiku + tools + plan   (planning=True added)

Special focus
-------------
The 9 tasks that hit max_steps=8 in block 8 are tracked separately.
They are identified from the saved ch04 traces in results/ch04_traces/.
This is the main test of the hypothesis: does planning reduce looping?

Usage
-----
    uv run python experiments/ch07_gaia.py --limit 3   # smoke
    uv run python experiments/ch07_gaia.py             # full 20 tasks
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
from agentkit.prompts import GAIA_AGENT_PROMPT
from agentkit.tools.base import BaseTool
from agentkit.tools.mcp import load_mcp_tools

MCP_CMD = (find_uv(), ["run", "python", "-m", "agentkit.servers.tavily_server"])
RESULTS_DIR = Path(__file__).parent.parent / "results"
TRACES_DIR = RESULTS_DIR / "ch04_traces"
CH07_TRACES_DIR = RESULTS_DIR / "ch07_traces"
MAX_STEPS = 12  # raised from 8 to give planning room to work
AGENT_CONCURRENCY = 2  # lower to avoid rate limits

_COST = {FAST_MODEL: {"input": 0.80, "output": 4.00}}


# ── Load block-8 hard tasks ────────────────────────────────────────────────────


def _load_ch04_hard_task_ids() -> set[str]:
    """Return task_ids that hit max_steps in block-8 haiku+with_tools run."""
    if not TRACES_DIR.exists():
        return set()
    hard: set[str] = set()
    for p in TRACES_DIR.glob("*__haiku*with_tools*.json"):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if data.get("steps", 0) >= 8:
                hard.add(data["task_id"])
        except Exception:  # noqa: BLE001, S112
            continue
    return hard


# ── Evaluation helpers ─────────────────────────────────────────────────────────


def _extract_prediction(output: Any) -> str | None:
    if isinstance(output, GaiaOutput):
        return output.final_answer if output.is_solvable else None
    text = str(output) if output else ""
    return None if text.startswith("[max_steps=") else text or None


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
    input_tok = token_usage.get("input_tokens", 0)
    output_tok = token_usage.get("output_tokens", 0)
    steps = result.context.current_step if result else 0

    if result:
        CH07_TRACES_DIR.mkdir(parents=True, exist_ok=True)
        trace_path = CH07_TRACES_DIR / f"{task_id}__{config_name}.json"
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
            "hit_max_steps": steps >= MAX_STEPS,
            "plan": result.context.state.get("plan"),
            "events": [json.loads(e.model_dump_json()) for e in result.context.events],
        }
        trace_path.write_text(
            json.dumps(trace_data, indent=2, ensure_ascii=False), encoding="utf-8"
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
        "error": error,
    }


async def _run_config(
    config_name: str,
    tools: list[BaseTool],
    problems: list[dict[str, Any]],
    semaphore: asyncio.Semaphore,
    planning: bool = False,
) -> list[dict[str, Any]]:
    agent = Agent(
        model=LlmClient(FAST_MODEL),
        tools=tools,
        instructions=GAIA_AGENT_PROMPT,
        max_steps=MAX_STEPS,
        output_type=GaiaOutput,
        name=config_name,
        planning=planning,
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
    }


def _cost(s: dict, model: str = FAST_MODEL) -> float:
    rates = _COST.get(model, {"input": 1.0, "output": 5.0})
    return (
        s["total_input_tokens"] * rates["input"]
        + s["total_output_tokens"] * rates["output"]
    ) / 1_000_000


def _print_table(
    all_results: list[tuple[str, list[dict]]],
    hard_ids: set[str],
) -> None:
    print(f"\n{'=' * 80}")
    print(
        f"  {'Config':<24} {'Acc':>6} {'Hard9-Acc':>10} {'Steps':>6} "
        f"{'HitMax':>7} {'InTok':>8} {'Cost$':>7}"
    )
    print("=" * 80)
    for config_name, rows in all_results:
        s = _summarise(rows)
        if not s:
            continue
        hard_rows = [r for r in rows if r["task_id"] in hard_ids]
        hard_s = _summarise(hard_rows)

        hard_acc = f"{hard_s['accuracy']:.0%} ({hard_s['correct']}/{hard_s['n']})" if hard_s else "  n/a"
        cost = _cost(s)
        print(
            f"  {config_name:<24} {s['accuracy']:>5.0%}  "
            f"{hard_acc:>10}  {s['avg_steps']:>6}  "
            f"{s['hit_max_steps']:>7}  {s['total_input_tokens']:>8}  "
            f"${cost:>5.2f}"
        )
    print("=" * 80)

    if hard_ids:
        print("\n  Hard-9 task_ids (hit max_steps=8 in block 8):")
        for tid in sorted(hard_ids):
            print(f"    {tid}")
    else:
        print("\n  Note: ch04 traces not found — Hard-9 column unavailable.")
        print("  Run ch04_gaia.py first to generate block-8 traces.")


# ── Main ───────────────────────────────────────────────────────────────────────


async def run(limit: int | None) -> None:
    problems = load_search_tasks()
    if limit:
        problems = problems[:limit]

    print(f"Running {len(problems)} tasks, max_steps={MAX_STEPS}, model={FAST_MODEL}")

    hard_ids = _load_ch04_hard_task_ids()
    if hard_ids:
        print(f"Found {len(hard_ids)} hard tasks from block-8 traces.")
    else:
        print("No block-8 traces found; hard-task column will show n/a.")

    semaphore = asyncio.Semaphore(AGENT_CONCURRENCY)
    all_results: list[tuple[str, list[dict]]] = []

    async with McpToolset(*MCP_CMD) as ts:
        search_tools = load_mcp_tools(ts)

        for config_name, planning in [
            ("haiku+tools", False),
            ("haiku+tools+plan", True),
        ]:
            rows = await _run_config(
                config_name, list(search_tools), problems, semaphore, planning=planning
            )
            all_results.append((config_name, rows))

    _print_table(all_results, hard_ids)

    # Per-task delta for hard tasks
    if hard_ids and len(all_results) == 2:
        baseline_map = {r["task_id"]: r for r in all_results[0][1]}
        plan_map = {r["task_id"]: r for r in all_results[1][1]}
        hard_in_run = hard_ids & set(baseline_map)
        if hard_in_run:
            print(f"\n  Per-task comparison on hard-9 tasks ({len(hard_in_run)} found in this run):")
            print(f"  {'task_id':<36} {'base':>5} {'plan':>5} {'base_steps':>11} {'plan_steps':>11}")
            for tid in sorted(hard_in_run):
                b = baseline_map[tid]
                p = plan_map.get(tid, {})
                b_ok = "ok" if b["correct"] else "--"
                p_ok = "ok" if p.get("correct") else "--"
                print(
                    f"  {tid:<36} {b_ok:>5} {p_ok:>5} "
                    f"{b['steps']:>11} {p.get('steps', '?'):>11}"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="Block 15 GAIA planning benchmark")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    try:
        asyncio.run(run(args.limit))
    except KeyboardInterrupt:
        print("\n[interrupted]")
        sys.exit(1)


if __name__ == "__main__":
    main()
