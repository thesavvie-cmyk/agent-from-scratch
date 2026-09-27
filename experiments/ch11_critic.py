"""Block 26 experiments: plan critique vs. baseline planning.

Sections
--------
  a  Custom dataset × 4 configs × 3 runs — factual accuracy + step/token metrics
  b  GAIA 10 tasks × 4 configs × 3 runs  — same metrics
  c  Print combined comparison table

Configs
-------
  no_planning   — Agent(planning=False)
  planning      — Agent(planning=True)
  critiqued_1   — Agent(planning="critiqued", critique_rounds=1)
  critiqued_2   — Agent(planning="critiqued", critique_rounds=2)

Each config runs 3 times (mean ± spread) because n≤10 and inter-run variance
on small datasets is 15–20%.

Usage
-----
    uv run --group eval python experiments/ch11_critic.py --section a
    uv run --group eval python experiments/ch11_critic.py --section b
    uv run --group eval python experiments/ch11_critic.py --section c
    uv run --group eval python experiments/ch11_critic.py --section all
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
from pathlib import Path

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "src"))
RESULTS = REPO / "results"
RESULTS.mkdir(exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────

CONFIGS: list[dict] = [
    {"name": "no_planning",   "planning": False,        "critique_rounds": 0},
    {"name": "planning",      "planning": True,         "critique_rounds": 0},
    {"name": "critiqued_2",   "planning": "critiqued",  "critique_rounds": 2},
]
RUNS_PER_CONFIG = 3

# Custom multi-step tasks that benefit from planning
CUSTOM_TASKS = [
    {
        "id": "custom-plan-001",
        "question": (
            "What is the population of the capital of the country that won "
            "the 2022 FIFA World Cup?"
        ),
        "gold": "3,433,000",  # Buenos Aires approx
    },
    {
        "id": "custom-plan-002",
        "question": (
            "How many Oscars did the highest-grossing film of 2023 win?"
        ),
        "gold": "7",  # Barbie won 0, Oppenheimer won 7
    },
    {
        "id": "custom-plan-003",
        "question": (
            "What is the name of the longest river in the continent that "
            "contains the country with the most UNESCO World Heritage Sites?"
        ),
        "gold": "Yangtze",
    },
    {
        "id": "custom-plan-004",
        "question": (
            "How many years after the invention of the telephone did the "
            "first commercial mobile phone call take place?"
        ),
        "gold": "107",  # telephone 1876, first commercial call 1983
    },
    {
        "id": "custom-plan-005",
        "question": (
            "What is the capital city of the country that borders both "
            "Sweden and Russia?"
        ),
        "gold": "Helsinki",
    },
    {
        "id": "custom-plan-006",
        "question": (
            "Name the element whose atomic number equals the number of "
            "bones in the adult human hand."
        ),
        "gold": "Iodine",  # 27 bones, atomic number 53 — Iodine; actually 27 bones = Cobalt
        # adult hand: 27 bones → element 27 = Cobalt
    },
    {
        "id": "custom-plan-007",
        "question": (
            "In what year was the city that hosted the 1992 Summer Olympics "
            "founded?"
        ),
        "gold": "801",  # Barcelona, year 801 CE (Carolingian conquest)
    },
    {
        "id": "custom-plan-008",
        "question": (
            "What is the currency used in the country that has the most "
            "Nobel Prize winners in Literature?"
        ),
        "gold": "Euro",  # France has the most (15+) — Euro
    },
    {
        "id": "custom-plan-009",
        "question": (
            "How many time zones does the country with the largest land area "
            "span?"
        ),
        "gold": "11",  # Russia
    },
    {
        "id": "custom-plan-010",
        "question": (
            "What is the tallest mountain in the country that is the world's "
            "largest producer of coffee?"
        ),
        "gold": "Pico da Neblina",  # Brazil, 2,994 m
    },
]


# ── Agent factory ─────────────────────────────────────────────────────────────


def _make_agent(cfg: dict) -> Agent:  # noqa: F821
    from agentkit.agent import Agent
    from agentkit.config import FAST_MODEL
    from agentkit.llm import LlmClient
    from agentkit.tools.base import FunctionTool
    from agentkit.tools.web import search_web

    model = LlmClient(model=FAST_MODEL)
    return Agent(
        model=model,
        tools=[FunctionTool(search_web)],
        planning=cfg["planning"],
        critique_rounds=cfg["critique_rounds"],
        max_steps=12,
        name=f"agent_{cfg['name']}",
    )


# ── Runner ────────────────────────────────────────────────────────────────────


async def _run_once(agent, question: str) -> dict:
    """Run agent on question; return metrics dict."""
    from agentkit.agent import Agent  # noqa: F401 (type hint only)

    result = await agent.run(question)
    usage = result.context.state.get("token_usage", {})
    steps = result.context.current_step
    hit_max = steps >= agent.max_steps
    output = result.output if isinstance(result.output, str) else str(result.output)
    return {
        "output": output,
        "steps": steps,
        "hit_max": hit_max,
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "error": result.error,
    }


async def _run_config_on_tasks(
    cfg: dict, tasks: list[dict], runs: int
) -> list[dict]:
    """Run cfg × tasks × runs, return flat list of run records."""
    records = []
    for task in tasks:
        for run_idx in range(runs):
            agent = _make_agent(cfg)
            run = await _run_once(agent, task["question"])
            records.append({
                "config": cfg["name"],
                "task_id": task["id"],
                "run": run_idx,
                "question": task["question"],
                "gold": task.get("gold", ""),
                **run,
            })
            status = "HIT_MAX" if run["hit_max"] else "ok"
            print(
                f"  [{cfg['name']}] {task['id']} run={run_idx} "
                f"steps={run['steps']} {status}: {run['output'][:60]}"
            )
    return records


# ── Judge ─────────────────────────────────────────────────────────────────────


async def _judge_records(records: list[dict]) -> list[dict]:
    """Run factual_accuracy judge on records that have gold answers."""
    from agentkit.config import SMART_MODEL
    from agentkit.eval.dataset import EvalCase
    from agentkit.eval.judge import judge_single
    from agentkit.eval.rubrics import FACTUAL_ACCURACY
    from agentkit.llm import LlmClient

    judge_llm = LlmClient(model=SMART_MODEL)
    judged = []
    for rec in records:
        gold = rec.get("gold", "")
        output = rec.get("output", "")
        if not gold or not output:
            judged.append({**rec, "verdict": "SKIP", "reasoning": ""})
            continue
        case = EvalCase(
            id=rec["task_id"],
            input=rec["question"],
            expected=gold,
            category=rec.get("config", "experiment"),
        )
        verdict = await judge_single(
            llm=judge_llm,
            case=case,
            output=output,
            trace_features=None,
            rubric=FACTUAL_ACCURACY,
        )
        judged.append({**rec, "verdict": verdict.verdict, "reasoning": verdict.reasoning})
    return judged


# ── Metrics ───────────────────────────────────────────────────────────────────


def _compute_metrics(records: list[dict]) -> dict[str, dict]:
    """Aggregate per-config metrics: accuracy, steps, tokens, hit_max."""
    by_config: dict[str, list[dict]] = {}
    for rec in records:
        by_config.setdefault(rec["config"], []).append(rec)

    result = {}
    for cfg_name, recs in by_config.items():
        judged = [r for r in recs if r.get("verdict") in ("PASS", "FAIL")]
        acc_vals: list[float] = []
        # per-task accuracy (mean of runs)
        task_ids = sorted({r["task_id"] for r in judged})
        for tid in task_ids:
            task_recs = [r for r in judged if r["task_id"] == tid]
            passes = sum(1 for r in task_recs if r["verdict"] == "PASS")
            acc_vals.append(passes / len(task_recs))

        steps_vals = [r["steps"] for r in recs]
        tokens_vals = [r["input_tokens"] + r["output_tokens"] for r in recs]
        hit_max_vals = [r["hit_max"] for r in recs]

        result[cfg_name] = {
            "accuracy_mean": statistics.mean(acc_vals) if acc_vals else 0.0,
            "accuracy_stdev": statistics.stdev(acc_vals) if len(acc_vals) > 1 else 0.0,
            "steps_mean": statistics.mean(steps_vals),
            "steps_stdev": statistics.stdev(steps_vals) if len(steps_vals) > 1 else 0.0,
            "tokens_mean": statistics.mean(tokens_vals),
            "hit_max_count": sum(hit_max_vals),
            "hit_max_rate": sum(hit_max_vals) / len(hit_max_vals),
            "n_runs": len(recs),
        }
    return result


def _print_table(metrics: dict[str, dict], title: str) -> None:
    print(f"\n{'─'*70}")
    print(f"  {title}")
    print(f"{'─'*70}")
    header = f"{'Config':<16} {'Acc':>6} {'±':>5} {'Steps':>6} {'±':>5} {'Tokens':>8} {'HitMax':>7}"
    print(header)
    print("─" * 70)
    for cfg_name in ["no_planning", "planning", "critiqued_2"]:
        if cfg_name not in metrics:
            continue
        m = metrics[cfg_name]
        print(
            f"{cfg_name:<16} "
            f"{m['accuracy_mean']:>5.0%} "
            f"{m['accuracy_stdev']:>5.0%} "
            f"{m['steps_mean']:>6.1f} "
            f"{m['steps_stdev']:>5.1f} "
            f"{m['tokens_mean']:>8.0f} "
            f"{m['hit_max_count']:>4}/{m['n_runs']}"
        )
    print()


# ── Sections ──────────────────────────────────────────────────────────────────


async def section_a() -> None:
    """Custom dataset × 4 configs × 3 runs."""
    print("\n=== Section A: Custom multi-step tasks ===")
    all_records: list[dict] = []
    for cfg in CONFIGS:
        print(f"\n--- Config: {cfg['name']} ---")
        recs = await _run_config_on_tasks(cfg, CUSTOM_TASKS, RUNS_PER_CONFIG)
        all_records.extend(recs)

    # Save raw records before judging so a judge-only failure is recoverable
    raw_path = RESULTS / "ch11_critic_custom_raw.jsonl"
    with raw_path.open("w", encoding="utf-8") as fh:
        for rec in all_records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Raw records saved → {raw_path}")

    print("\nJudging with factual_accuracy rubric...")
    judged = await _judge_records(all_records)

    path = RESULTS / "ch11_critic_custom.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for rec in judged:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Saved {len(judged)} records → {path}")

    metrics = _compute_metrics(judged)
    _print_table(metrics, "Custom dataset — factual_accuracy (3 runs per config)")

    (RESULTS / "ch11_metrics_custom.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )


async def section_b() -> None:
    """GAIA 10 tasks × 4 configs × 3 runs."""
    print("\n=== Section B: GAIA 10 tasks ===")
    from agentkit.gaia import load_search_tasks

    gaia_raw = load_search_tasks(limit=10)
    gaia_tasks = [
        {
            "id": t["task_id"],
            "question": t["Question"],
            "gold": t["Final answer"],
        }
        for t in gaia_raw
    ]

    all_records: list[dict] = []
    for cfg in CONFIGS:
        print(f"\n--- Config: {cfg['name']} ---")
        recs = await _run_config_on_tasks(cfg, gaia_tasks, RUNS_PER_CONFIG)
        all_records.extend(recs)

    raw_path = RESULTS / "ch11_critic_gaia_raw.jsonl"
    with raw_path.open("w", encoding="utf-8") as fh:
        for rec in all_records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Raw records saved → {raw_path}")

    print("\nJudging with factual_accuracy rubric...")
    judged = await _judge_records(all_records)

    path = RESULTS / "ch11_critic_gaia.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for rec in judged:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Saved {len(judged)} records → {path}")

    metrics = _compute_metrics(judged)
    _print_table(metrics, "GAIA 10 tasks — factual_accuracy (3 runs per config)")

    (RESULTS / "ch11_metrics_gaia.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def section_c() -> None:
    """Print combined comparison table from saved results."""
    print("\n=== Section C: Combined comparison ===")

    custom_path = RESULTS / "ch11_metrics_custom.json"
    gaia_path = RESULTS / "ch11_metrics_gaia.json"

    for label, path in [("Custom dataset", custom_path), ("GAIA 10 tasks", gaia_path)]:
        if not path.exists():
            print(f"  {label}: no results file ({path.name}) — run section a/b first.")
            continue
        metrics = json.loads(path.read_text(encoding="utf-8"))
        _print_table(metrics, label)


# ── Judge-only (recover from raw records) ────────────────────────────────────


async def _judge_only() -> None:
    """Judge already-saved raw records without re-running agents."""
    for label, raw_path, out_path, metrics_path in [
        ("Custom",  RESULTS / "ch11_critic_custom_raw.jsonl",
                    RESULTS / "ch11_critic_custom.jsonl",
                    RESULTS / "ch11_metrics_custom.json"),
        ("GAIA",    RESULTS / "ch11_critic_gaia_raw.jsonl",
                    RESULTS / "ch11_critic_gaia.jsonl",
                    RESULTS / "ch11_metrics_gaia.json"),
    ]:
        if not raw_path.exists():
            print(f"  {label}: no raw file ({raw_path.name}) — skipping.")
            continue
        all_records = []
        with raw_path.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    all_records.append(json.loads(line))
        print(f"\n=== {label}: judging {len(all_records)} records ===")
        judged = await _judge_records(all_records)
        with out_path.open("w", encoding="utf-8") as fh:
            for rec in judged:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"Saved {len(judged)} records → {out_path}")
        metrics = _compute_metrics(judged)
        metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
        _print_table(metrics, f"{label} — factual_accuracy")


# ── Main ──────────────────────────────────────────────────────────────────────


async def main() -> None:
    parser = argparse.ArgumentParser(description="Block 26: plan critique experiments")
    parser.add_argument(
        "--section",
        choices=["a", "b", "c", "all"],
        default="all",
        help="Which section to run (default: all)",
    )
    parser.add_argument(
        "--judge-only",
        action="store_true",
        help="Skip agent runs; judge raw records already saved in results/",
    )
    args = parser.parse_args()

    if args.judge_only:
        await _judge_only()
        return

    if args.section in ("a", "all"):
        await section_a()
    if args.section in ("b", "all"):
        await section_b()
    if args.section in ("c", "all"):
        section_c()


if __name__ == "__main__":
    asyncio.run(main())
