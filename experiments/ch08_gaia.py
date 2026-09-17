"""Chapter 8 GAIA benchmark — 4 configs with code execution (block 18).

Configs
-------
baseline     haiku + search (no code, no reflection) — block 8 baseline
+refl        haiku + search + reflection              — block 16 best
+code        haiku + search + code execution          — block 17 best
+code+refl   haiku + search + code + reflection       — block 18

Metrics
-------
  Accuracy   — exact / normalised match against ground truth
  HitMax     — tasks that hit max_steps (agent got stuck)
  CodeCalls  — number of execute_python calls (new in this block)
  DupCalls   — repeated identical tool calls

Usage
-----
    uv run python experiments/ch08_gaia.py --tasks 20
    uv run python experiments/ch08_gaia.py --tasks 5 --config code

Requires ANTHROPIC_API_KEY + TAVILY_API_KEY + E2B_API_KEY.
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

from agentkit.agent import Agent
from agentkit.config import FAST_MODEL, find_uv
from agentkit.llm import LlmClient
from agentkit.mcp_client import McpToolset
from agentkit.reflection import REFLECTION_INSTRUCTIONS
from agentkit.tools.mcp import load_mcp_tools

MCP_CMD = (find_uv(), ["run", "python", "-m", "agentkit.servers.tavily_server"])
RESULTS_DIR = Path(__file__).parent.parent / "results"
MAX_STEPS = 8


# ── GAIA loader ────────────────────────────────────────────────────────────────


def _load_gaia(n: int) -> list[dict[str, Any]]:
    data_file = Path(__file__).parent.parent / "data" / "gaia_validation.jsonl"
    tasks: list[dict[str, Any]] = []
    with data_file.open() as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
            if len(tasks) >= n:
                break
    return tasks


# ── Answer normalisation ───────────────────────────────────────────────────────


def _normalise(text: str) -> str:
    return (
        text.lower()
        .replace(",", "")
        .replace("$", "")
        .replace("%", "")
        .strip()
        .rstrip(".")
    )


def _is_correct(prediction: str, gold: str) -> bool:
    pred = _normalise(str(prediction))
    gold_n = _normalise(gold)
    return pred == gold_n or gold_n in pred


# ── Trace analysis ─────────────────────────────────────────────────────────────


def _count_dup_calls(events: list[Any]) -> int:
    from agentkit.types import ToolCall
    seen: set[str] = set()
    dups = 0
    for ev in events:
        for item in ev.content:
            if isinstance(item, ToolCall):
                key = f"{item.name}:{json.dumps(item.arguments, sort_keys=True)}"
                if key in seen:
                    dups += 1
                seen.add(key)
    return dups


def _count_code_calls(events: list[Any]) -> int:
    from agentkit.types import ToolCall
    return sum(
        1 for ev in events
        for item in ev.content
        if isinstance(item, ToolCall) and item.name == "execute_python"
    )


# ── Single task runner ─────────────────────────────────────────────────────────


async def _run_task(
    task: dict[str, Any],
    search_tools: list[Any],
    use_reflection: bool,
    use_code: bool,
) -> dict[str, Any]:
    instructions = "Answer questions precisely. Give only the final answer — no explanation."
    if use_reflection:
        instructions += f"\n\n{REFLECTION_INSTRUCTIONS}"
    if use_code:
        instructions += (
            "\n\nUse execute_python for calculations, data processing, and tasks "
            "requiring structured output or precise computation."
        )

    agent = Agent(
        model=LlmClient(FAST_MODEL),
        tools=search_tools,
        instructions=instructions,
        max_steps=MAX_STEPS,
        reflection=use_reflection,
        code_execution="e2b" if use_code else None,
    )

    t0 = time.perf_counter()
    result = await agent.run(task["Question"])
    elapsed = time.perf_counter() - t0

    hit_max = result.context.current_step >= MAX_STEPS and result.output == (
        f"[max_steps={MAX_STEPS} reached without final answer]"
    )
    correct = _is_correct(str(result.output), task.get("Final answer", ""))
    dup_calls = _count_dup_calls(result.context.events)
    code_calls = _count_code_calls(result.context.events)

    return {
        "task_id": task.get("task_id", ""),
        "question": task["Question"][:80],
        "gold": task.get("Final answer", ""),
        "prediction": str(result.output)[:200],
        "correct": correct,
        "hit_max": hit_max,
        "steps": result.context.current_step,
        "dup_calls": dup_calls,
        "code_calls": code_calls,
        "elapsed": elapsed,
        "error": result.error,
    }


# ── Config runner ──────────────────────────────────────────────────────────────


async def _run_config(
    label: str,
    tasks: list[dict[str, Any]],
    search_tools: list[Any],
    use_reflection: bool,
    use_code: bool,
) -> list[dict[str, Any]]:
    print(f"\n  Running [{label}] ({len(tasks)} tasks)...")
    results = []
    for i, task in enumerate(tasks):
        r = await _run_task(task, search_tools, use_reflection, use_code)
        tick = "✓" if r["correct"] else ("!" if r["hit_max"] else "✗")
        code_marker = f" code={r['code_calls']}" if r["code_calls"] else ""
        print(
            f"    [{i + 1:2d}/{len(tasks)}] {tick} steps={r['steps']}"
            f"{code_marker}  {r['question'][:50]}"
        )
        results.append(r)
    return results


def _summarise(label: str, results: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(results)
    acc = sum(r["correct"] for r in results) / n if n else 0
    hit_max = sum(r["hit_max"] for r in results)
    dup_calls = sum(r["dup_calls"] for r in results)
    code_calls = sum(r["code_calls"] for r in results)
    return {
        "label": label,
        "n": n,
        "accuracy": acc,
        "hit_max": hit_max,
        "dup_calls": dup_calls,
        "code_calls": code_calls,
    }


# ── Main ───────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Block 18 GAIA benchmark")
    parser.add_argument("--tasks", type=int, default=20)
    parser.add_argument(
        "--config", default="all",
        choices=["baseline", "refl", "code", "code_refl", "all"],
    )
    return parser.parse_args()


async def _main(args: argparse.Namespace) -> None:
    tasks = _load_gaia(args.tasks)
    print(f"\nLoaded {len(tasks)} GAIA tasks  model={FAST_MODEL}  max_steps={MAX_STEPS}")

    summaries: list[dict[str, Any]] = []

    async with McpToolset(*MCP_CMD) as ts:
        search_tools = list(load_mcp_tools(ts))

        configs = [
            ("baseline",   False, False),
            ("refl",       True,  False),
            ("code",       False, True),
            ("code+refl",  True,  True),
        ]
        if args.config != "all":
            label_map = {"baseline": "baseline", "refl": "refl", "code": "code", "code_refl": "code+refl"}
            target = label_map[args.config]
            configs = [(l, r, c) for l, r, c in configs if l == target]

        all_results: dict[str, list[dict]] = {}
        for label, use_refl, use_code in configs:
            results = await _run_config(label, tasks, search_tools, use_refl, use_code)
            all_results[label] = results
            summaries.append(_summarise(label, results))

            # Save per-config results
            RESULTS_DIR.mkdir(exist_ok=True)
            out_path = RESULTS_DIR / f"ch08_gaia_{label}.json"
            out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
            print(f"  Saved → {out_path}")

    # Summary table
    print(f"\n  {'Config':<14} {'Acc':>6} {'HitMax':>7} {'CodeCalls':>10} {'DupCalls':>9}")
    print(f"  {'-'*14} {'-'*6} {'-'*7} {'-'*10} {'-'*9}")
    for s in summaries:
        print(
            f"  {s['label']:<14} {s['accuracy']:>5.0%}   "
            f"{s['hit_max']:>5}/{s['n']}  "
            f"{s['code_calls']:>10}  "
            f"{s['dup_calls']:>9}"
        )

    # Show tasks where code made a difference
    if "baseline" in all_results and "code" in all_results:
        code_helped = [
            r for r, b in zip(all_results["code"], all_results["baseline"])
            if r["correct"] and not b["correct"] and r["code_calls"] > 0
        ]
        if code_helped:
            print(f"\n  Tasks solved by code but NOT by baseline ({len(code_helped)}):")
            for r in code_helped[:5]:
                print(f"    code_calls={r['code_calls']}  {r['question']}")


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
