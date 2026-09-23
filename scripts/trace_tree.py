"""trace_tree.py — print a JSONL span file as a terminal tree.

Usage
-----
    uv run python scripts/trace_tree.py results/traces/trace.jsonl
    uv run python scripts/trace_tree.py results/traces/trace.jsonl --trace <trace_id>

Each line in the JSONL file must be a span dict produced by
``agentkit.telemetry.FileSpanExporter`` (or any compatible exporter).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]


def _load(path: Path) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    spans.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    print(f"  [warn] skipping malformed line: {exc}", file=sys.stderr)
    return spans


def _build_tree(
    spans: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Return parent_span_id → [children] mapping (None key = roots)."""
    children: dict[str | None, list[dict[str, Any]]] = defaultdict(list)
    for span in spans:
        children[span.get("parent_span_id")].append(span)
    # Sort children by start_time within each parent
    for lst in children.values():
        lst.sort(key=lambda s: s.get("start_time") or 0)
    return children  # type: ignore[return-value]


def _fmt_duration(ms: float | None) -> str:
    if ms is None:
        return ""
    if ms >= 1000:
        return f"{ms / 1000:.2f}s"
    return f"{ms:.1f}ms"


def _fmt_attrs(attrs: dict[str, Any], max_keys: int = 3) -> str:
    if not attrs:
        return ""
    items = list(attrs.items())[:max_keys]
    parts = [f"{k}={v!r}" for k, v in items]
    if len(attrs) > max_keys:
        parts.append(f"…+{len(attrs) - max_keys}")
    return "  {" + ", ".join(parts) + "}"


def _print_tree(
    span: dict[str, Any],
    children: dict[str | None, list[dict[str, Any]]],
    prefix: str = "",
    is_last: bool = True,
) -> None:
    connector = "└── " if is_last else "├── "
    duration = _fmt_duration(span.get("duration_ms"))
    dur_str = f"  [{duration}]" if duration else ""
    attrs_str = _fmt_attrs(span.get("attributes", {}))
    status = span.get("status", "UNSET")
    status_str = f"  ({status})" if status not in ("UNSET", "OK", "UNSET_STATUS_CODE") else ""
    span_id = span.get("span_id", "")[:8]

    print(f"{prefix}{connector}{span['name']}{dur_str}{attrs_str}{status_str}  [{span_id}]")

    # Print span events indented under the span
    for evt in span.get("events", []):
        evt_prefix = prefix + ("    " if is_last else "│   ")
        evt_attrs = _fmt_attrs(evt.get("attributes", {}))
        print(f"{evt_prefix}    event: {evt['name']}{evt_attrs}")

    child_prefix = prefix + ("    " if is_last else "│   ")
    kids = children.get(span.get("span_id"))
    if kids:
        for i, child in enumerate(kids):
            _print_tree(child, children, child_prefix, is_last=(i == len(kids) - 1))


def print_traces(spans: list[dict[str, Any]], trace_id: str | None = None) -> None:
    if trace_id:
        spans = [s for s in spans if s.get("trace_id") == trace_id]

    if not spans:
        print("No spans found.")
        return

    # Group by trace_id
    by_trace: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for span in spans:
        by_trace[span.get("trace_id", "unknown")].append(span)

    for tid, trace_spans in by_trace.items():
        children = _build_tree(trace_spans)
        roots = children.get(None, [])
        print(f"\ntrace_id={tid}  ({len(trace_spans)} spans)")
        print("─" * 68)
        for i, root in enumerate(roots):
            _print_tree(root, children, is_last=(i == len(roots) - 1))


def main() -> None:
    parser = argparse.ArgumentParser(description="Print JSONL span file as a tree")
    parser.add_argument("file", type=Path, help="Path to the JSONL trace file")
    parser.add_argument("--trace", help="Filter to a specific trace_id prefix")
    args = parser.parse_args()

    if not args.file.exists():
        print(f"File not found: {args.file}", file=sys.stderr)
        sys.exit(1)

    spans = _load(args.file)
    if not spans:
        print("No spans in file.")
        return

    trace_id = args.trace
    if trace_id and len(trace_id) < 32:
        # Prefix match
        all_ids = {s.get("trace_id", "") for s in spans}
        matches = [t for t in all_ids if t.startswith(trace_id)]
        if len(matches) == 1:
            trace_id = matches[0]
        elif len(matches) > 1:
            print(f"Ambiguous prefix {trace_id!r}: matches {matches}", file=sys.stderr)
            sys.exit(1)

    print_traces(spans, trace_id=trace_id)


if __name__ == "__main__":
    main()
