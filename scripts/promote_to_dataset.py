"""Promote a production failure to the eval dataset (block 25).

After running collect_failures.py and open_coding.py on a failure case,
this script adds it to a dataset JSONL as a new EvalCase with source=production.

Usage
-----
    # Add a single case interactively:
    uv run python scripts/promote_to_dataset.py \\
        --dataset data/custom_dataset.jsonl \\
        --input "What is the latest version of Python?" \\
        --expected "3.13" \\
        --category core \\
        --tags search,version \\
        --source production

    # Promote all cases from an open_codes JSONL (from open_coding.py):
    uv run python scripts/promote_to_dataset.py \\
        --dataset data/custom_dataset.jsonl \\
        --from-codes results/open_codes.jsonl \\
        --source production

The --from-codes mode reads every entry that has a label field (annotated
during open coding) and adds it as a new case.  Existing case IDs are
skipped to avoid duplicates.
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]


def _load_existing_ids(dataset_path: Path) -> set[str]:
    if not dataset_path.exists():
        return set()
    ids = set()
    with dataset_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    ids.add(json.loads(line)["id"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return ids


def _append_case(dataset_path: Path, case: dict) -> None:
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    with dataset_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(case, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote production failure to eval dataset")
    parser.add_argument("--dataset", required=True, type=Path, help="Target dataset JSONL")
    parser.add_argument("--source", default="production", help="Source label (default: production)")

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--input", help="Task input string (single-case mode)")
    mode.add_argument("--from-codes", type=Path, help="Open-coding JSONL (batch mode)")

    parser.add_argument("--expected", default=None, help="Expected answer (single-case mode)")
    parser.add_argument("--category", default="core", help="Case category (single-case mode)")
    parser.add_argument("--tags", default="", help="Comma-separated tags (single-case mode)")
    args = parser.parse_args()

    existing = _load_existing_ids(args.dataset)
    added = 0

    if args.input:
        # Single-case mode
        case_id = str(uuid.uuid4())
        tags = [t.strip() for t in args.tags.split(",") if t.strip()]
        case = {
            "id": case_id,
            "input": args.input,
            "expected": args.expected,
            "category": args.category,
            "tags": tags,
            "metadata": {"source": args.source},
        }
        _append_case(args.dataset, case)
        print(f"Added case {case_id} to {args.dataset}")
        added = 1

    elif args.from_codes:
        # Batch mode from open_coding.py output
        if not args.from_codes.exists():
            print(f"Error: {args.from_codes} not found", file=sys.stderr)
            sys.exit(1)
        with args.from_codes.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Only promote annotated entries
                if not entry.get("label"):
                    continue

                case_id = entry.get("case_id", str(uuid.uuid4()))
                if case_id in existing:
                    print(f"  Skipping {case_id} (already in dataset)")
                    continue

                case = {
                    "id": case_id,
                    "input": entry.get("input", ""),
                    "expected": entry.get("expected"),
                    "category": entry.get("category", "core"),
                    "tags": entry.get("tags", []),
                    "metadata": {
                        "source": args.source,
                        "open_code_label": entry.get("label", ""),
                        "open_code_note": entry.get("note", ""),
                    },
                }
                _append_case(args.dataset, case)
                existing.add(case_id)
                added += 1
                print(f"  Added {case_id}: {entry.get('input', '')[:60]}")

    print(f"\nTotal added: {added}")


if __name__ == "__main__":
    main()
