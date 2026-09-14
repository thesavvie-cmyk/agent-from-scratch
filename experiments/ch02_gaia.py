"""Chapter 2: GAIA level-1 baseline evaluation.

Usage:
    uv run python experiments/ch02_gaia.py --limit 20
    uv run python experiments/ch02_gaia.py --limit 20 --models anthropic/claude-haiku-4-5
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from collections import Counter
from pathlib import Path

from agentkit.config import FAST_MODEL, SMART_MODEL
from agentkit.gaia import load_level1, run_experiment


def _normalize_loose(s: str) -> str:
    """Strip whitespace, commas, currency symbols; lowercase."""
    return re.sub(r"[\s,$€£¥₹]", "", s.lower())


def _categorize_reason(reason: str) -> str:
    r = reason.lower()
    if any(kw in r for kw in ["search", "web", "internet", "browse", "url", "real-time", "current"]):
        return "web search"
    if any(kw in r for kw in ["file", "image", "pdf", "document", "attachment", "spreadsheet", "read"]):
        return "file reading"
    return "other"


def print_table(results_by_model: dict[str, list[dict]]) -> None:
    col = 44
    header = f"{'Model':<{col}} {'Acc':>9} {'Solvable':>10} {'Declined':>10} {'Refusals':>10} {'Errors':>7} {'Tokens':>9} {'Time':>7}"
    bar = "=" * len(header)
    print(f"\n{bar}\n{header}\n{bar}")
    for model, results in results_by_model.items():
        n = len(results)
        if n == 0:
            continue
        correct = sum(r["correct"] for r in results)
        solvable = sum(r["is_solvable"] for r in results)
        # declined = model said is_solvable=False (needs tools/search) — expected good behaviour
        declined = sum(1 for r in results if not r["is_solvable"] and not r["error"])
        # refusals = actual API-level safety refusal (finish_reason=="refusal")
        refusals = sum(1 for r in results if r.get("api_refusal"))
        errors = sum(1 for r in results if r["error"])
        tokens = sum(r["input_tokens"] + r["output_tokens"] for r in results)
        elapsed = sum(r["latency_s"] for r in results)
        short = model.split("/")[-1][:col]
        print(
            f"{short:<{col}} {correct}/{n} ({100 * correct // n}%){'':<1}"
            f" {solvable:>6}/{n}"
            f" {declined:>10}"
            f" {refusals:>10}"
            f" {errors:>7}"
            f" {tokens:>9}"
            f" {elapsed:>6.1f}s"
        )
    print(bar)


def print_unsolvable_breakdown(results_by_model: dict[str, list[dict]]) -> None:
    print("\n── Unsolvable breakdown ──────────────────────────────────")
    for model, results in results_by_model.items():
        unsolvable = [r for r in results if not r["is_solvable"] and not r["error"]]
        if not unsolvable:
            continue
        counts: Counter[str] = Counter(
            _categorize_reason(r["unsolvable_reason"]) for r in unsolvable
        )
        print(f"  {model.split('/')[-1]}: {dict(counts)}")


def print_near_correct(results_by_model: dict[str, list[dict]]) -> None:
    print("\n── Near-correct (exact match fails, loose match passes) ──")
    found = False
    for model, results in results_by_model.items():
        near = [
            r for r in results
            if not r["correct"]
            and r["prediction"] is not None
            and _normalize_loose(r["prediction"]) == _normalize_loose(r["answer"])
        ]
        for r in near:
            found = True
            print(
                f"  [{model.split('/')[-1][:20]}]"
                f"  pred={r['prediction']!r:<24}"
                f"  ans={r['answer']!r}"
            )
    if not found:
        print("  (none found)")


def main() -> None:
    parser = argparse.ArgumentParser(description="GAIA level-1 baseline")
    parser.add_argument("--limit", type=int, default=20, help="Number of problems")
    parser.add_argument(
        "--models", nargs="+", default=[FAST_MODEL, SMART_MODEL], help="Model IDs"
    )
    parser.add_argument("--out", default="results/ch02_baseline.json", help="Output path")
    args = parser.parse_args()

    print(f"Loading {args.limit} GAIA level-1 problems...")
    try:
        problems = load_level1(limit=args.limit)
    except Exception as exc:  # noqa: BLE001
        print(f"Error loading dataset: {exc}", file=sys.stderr)
        sys.exit(1)
    print(f"Loaded {len(problems)} problems. Models: {args.models}\n")

    results_by_model = asyncio.run(run_experiment(problems, args.models))

    print_table(results_by_model)
    print_unsolvable_breakdown(results_by_model)
    print_near_correct(results_by_model)

    # Save JSON
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    all_results = [r for rows in results_by_model.values() for r in rows]
    out_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False))
    print(f"\nSaved {len(all_results)} results → {out_path}")

    # Save CSV
    try:
        import pandas as pd

        csv_path = out_path.with_suffix(".csv")
        pd.DataFrame(all_results).to_csv(csv_path, index=False)
        print(f"Saved CSV        → {csv_path}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
