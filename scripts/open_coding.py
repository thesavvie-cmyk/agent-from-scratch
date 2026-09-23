"""Open coding — manual trace annotation CLI (block 24).

Usage
-----
    uv run --group eval python scripts/open_coding.py \\
        --traces results/traces.jsonl \\
        --output results/open_codes.jsonl

Workflow
--------
For each trace in the input file the script displays:
  - task input
  - agent output (prediction)
  - trace summary (tool calls, steps, tokens)

The annotator enters a label and an optional note.  Labels are free-form
strings (e.g. "hallucination", "correct_but_unsourced", "tool_error").
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_traces(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def _load_done(output_path: Path) -> set[str]:
    """Return set of already-annotated task IDs."""
    done: set[str] = set()
    if not output_path.exists():
        return done
    with output_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    entry = json.loads(line)
                    tid = entry.get("task_id") or entry.get("id")
                    if tid:
                        done.add(tid)
                except json.JSONDecodeError:
                    pass
    return done


def _append(output_path: Path, entry: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _fmt_trace(record: dict) -> str:
    parts = []
    steps = record.get("steps", "?")
    parts.append(f"Steps: {steps}")
    events = record.get("events", [])
    tool_counts: dict[str, int] = {}
    for ev in events:
        for item in ev.get("content", []):
            if item.get("type") == "tool_call":
                name = item.get("name", "unknown")
                tool_counts[name] = tool_counts.get(name, 0) + 1
    if tool_counts:
        breakdown = ", ".join(f"{k}×{v}" for k, v in sorted(tool_counts.items()))
        parts.append(f"Tool calls: {breakdown}")
    else:
        parts.append("Tool calls: 0")
    in_tok = record.get("input_tokens", 0)
    out_tok = record.get("output_tokens", 0)
    if in_tok or out_tok:
        parts.append(f"Tokens: {in_tok} in / {out_tok} out")
    return " | ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Open coding — manual trace annotation")
    parser.add_argument("--traces", required=True, type=Path, help="Input JSONL (legacy results)")
    parser.add_argument("--output", required=True, type=Path, help="Output JSONL for annotations")
    parser.add_argument("--limit", type=int, default=None, help="Annotate at most N traces")
    args = parser.parse_args()

    records = _load_traces(args.traces)
    done = _load_done(args.output)

    remaining = [r for r in records if (r.get("task_id") or r.get("id")) not in done]
    if args.limit:
        remaining = remaining[: args.limit]

    if not remaining:
        print("Nothing left to annotate.")
        return

    print(f"Annotating {len(remaining)} traces  (Ctrl-C to stop, progress saved)\n")
    print("Commands: enter a label string, or 's' to skip, 'q' to quit.\n")

    for idx, record in enumerate(remaining, 1):
        task_id = record.get("task_id") or record.get("id") or f"unknown-{idx}"
        question = record.get("question", record.get("input", "(no question)"))
        prediction = record.get("prediction", record.get("output", "(no output)"))
        trace_str = _fmt_trace(record)

        print("─" * 70)
        print(f"[{idx}/{len(remaining)}] Task: {task_id}")
        print(f"\nQ: {question[:300]}")
        print(f"\nA: {prediction[:400]}")
        print(f"\nTrace: {trace_str}")
        print()

        try:
            label = input("Label (or 's'=skip, 'q'=quit): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nStopped.")
            break

        if label.lower() == "q":
            break
        if label.lower() == "s":
            continue

        try:
            note = input("Note (optional, Enter to skip): ").strip()
        except (EOFError, KeyboardInterrupt):
            note = ""

        entry = {
            "task_id": task_id,
            "label": label,
            "note": note,
        }
        _append(args.output, entry)
        print(f"  → saved: {label!r}\n")

    print(f"\nDone.  Annotations saved to {args.output}")


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    main()
