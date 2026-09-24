"""Collect failed eval cases for open coding (block 25).

Reads an eval report JSON, extracts cases where any rubric returned FAIL,
and writes them to a JSONL file ready for open_coding.py or manual review.

Usage
-----
    uv run python scripts/collect_failures.py \\
        --report results/ch10_section_a_report.json \\
        --output results/failures.jsonl

    # Only collect failures for a specific rubric:
    uv run python scripts/collect_failures.py \\
        --report results/ch10_section_a_report.json \\
        --rubric factual_accuracy \\
        --output results/factual_failures.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract failed cases from eval report")
    parser.add_argument("--report", required=True, type=Path, help="Eval report JSON")
    parser.add_argument("--output", required=True, type=Path, help="Output JSONL for open coding")
    parser.add_argument(
        "--rubric", default=None,
        help="Only collect failures for this rubric (default: any rubric)",
    )
    args = parser.parse_args()

    data = json.loads(args.report.read_text(encoding="utf-8"))
    results = data.get("results", [])

    # Group by case_id
    by_case: dict[str, dict] = {}
    for r in results:
        cid = r["case_id"]
        if args.rubric and r["rubric_name"] != args.rubric:
            continue
        if r["verdict"] == "FAIL":
            if cid not in by_case:
                by_case[cid] = {
                    "case_id": cid,
                    "category": r.get("category", ""),
                    "output": r.get("output", ""),
                    "failed_rubrics": [],
                }
            by_case[cid]["failed_rubrics"].append({
                "rubric": r["rubric_name"],
                "reasoning": r.get("reasoning", ""),
            })

    if not by_case:
        print("No failures found.")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        for case in by_case.values():
            fh.write(json.dumps(case, ensure_ascii=False) + "\n")

    print(f"Wrote {len(by_case)} failed cases to {args.output}")
    print("\nFailed cases by rubric:")
    from collections import Counter
    rubric_counts: Counter = Counter()
    for case in by_case.values():
        for fr in case["failed_rubrics"]:
            rubric_counts[fr["rubric"]] += 1
    for rubric, count in rubric_counts.most_common():
        print(f"  {rubric}: {count}")


if __name__ == "__main__":
    main()
