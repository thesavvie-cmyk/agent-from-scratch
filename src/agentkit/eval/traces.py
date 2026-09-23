"""Trace loading and feature extraction (block 24).

Supports two trace formats:
  - OTel JSONL (produced by agentkit.telemetry.FileSpanExporter, block 23)
  - Legacy result JSON (produced by ch04/ch07/ch08 experiment runners,
    containing an "events" list of agentkit.types.Event dicts)

TraceFeatures
-------------
Normalised per-trace statistics used by trajectory rubrics and the eval
runner to summarise what the agent actually did.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ── Data types ────────────────────────────────────────────────────────────────


@dataclass
class TraceFeatures:
    """Normalised statistics extracted from one agent trace."""

    task_id: str = ""
    steps: int = 0
    tool_calls: dict[str, int] = field(default_factory=dict)   # name → call count
    tool_errors: int = 0
    hit_max_steps: bool = False
    input_tokens: int = 0
    output_tokens: int = 0
    used_code: bool = False
    search_calls: int = 0

    # ── derived properties ────────────────────────────────────────────────────

    @property
    def total_tool_calls(self) -> int:
        return sum(self.tool_calls.values())

    @property
    def used_search(self) -> bool:
        return self.search_calls > 0

    def summary_text(self) -> str:
        """One-paragraph human-readable summary for the judge prompt."""
        parts = [
            f"Steps: {self.steps}",
            f"Tool calls: {self.total_tool_calls}",
        ]
        if self.tool_calls:
            calls_str = ", ".join(
                f"{k}×{v}" for k, v in sorted(self.tool_calls.items())
            )
            parts.append(f"Breakdown: {calls_str}")
        if self.tool_errors:
            parts.append(f"Tool errors: {self.tool_errors}")
        if self.hit_max_steps:
            parts.append("Hit max_steps: YES")
        if self.input_tokens or self.output_tokens:
            parts.append(
                f"Tokens: {self.input_tokens} in / {self.output_tokens} out"
            )
        if self.used_code:
            parts.append("Used code execution: YES")
        return " | ".join(parts)


# ── Feature extraction from agentkit Event lists ──────────────────────────────

_SEARCH_TOOL_NAMES = frozenset({
    "search_web",
    "tavily_search",
    "web_search",
})
_CODE_TOOL_NAMES = frozenset({
    "execute_python",
    "run_python",
    "python",
})


def extract_features(
    events: list[dict[str, Any]],
    hit_max_steps: bool = False,
    input_tokens: int = 0,
    output_tokens: int = 0,
    task_id: str = "",
) -> TraceFeatures:
    """Extract TraceFeatures from a list of agentkit Event dicts.

    Each event dict must have a ``content`` list of ContentItem dicts
    (``type`` ∈ ``message | tool_call | tool_result``).
    """
    tool_calls: dict[str, int] = {}
    tool_errors = 0
    step_authors: set[str] = set()

    for ev in events:
        author = ev.get("author", "")
        if author and author not in ("user",):
            step_authors.add(author)
        for item in ev.get("content", []):
            item_type = item.get("type", "")
            if item_type == "tool_call":
                name = item.get("name", "unknown")
                tool_calls[name] = tool_calls.get(name, 0) + 1
            elif item_type == "tool_result":
                # status may be "error", or content may start with "Error:"
                status = item.get("status", "success")
                if status == "error":
                    tool_errors += 1
                else:
                    content_parts = item.get("content", [])
                    if content_parts:
                        first = content_parts[0]
                        text = first.get("text", "") if isinstance(first, dict) else str(first)
                        if text.startswith("Error:"):
                            tool_errors += 1

    search_calls = sum(v for k, v in tool_calls.items() if k in _SEARCH_TOOL_NAMES)
    used_code = any(k in _CODE_TOOL_NAMES for k in tool_calls)

    return TraceFeatures(
        task_id=task_id,
        steps=len(step_authors) if step_authors else len(events),
        tool_calls=tool_calls,
        tool_errors=tool_errors,
        hit_max_steps=hit_max_steps,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        used_code=used_code,
        search_calls=search_calls,
    )


# ── Load legacy result JSONs (ch04 / ch07 / ch08 format) ──────────────────────


def load_legacy_results(path: Path) -> list[dict[str, Any]]:
    """Load results from a legacy JSON file (list of result dicts).

    Format produced by ch04/ch08 experiment runners:
      [{"task_id": ..., "question": ..., "gold": ..., "prediction": ...,
        "correct": ..., "hit_max": ..., "steps": ..., "events": [...], ...}]
    """
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    if isinstance(data, list):
        return data
    # Some runs saved a dict mapping task_id → result
    if isinstance(data, dict):
        return list(data.values())
    return []


def features_from_legacy(result: dict[str, Any]) -> TraceFeatures:
    """Extract TraceFeatures from a legacy result dict (ch04/ch08 format)."""
    return extract_features(
        events=result.get("events", []),
        hit_max_steps=bool(result.get("hit_max", False)),
        input_tokens=int(result.get("input_tokens", 0)),
        output_tokens=int(result.get("output_tokens", 0)),
        task_id=result.get("task_id", ""),
    )


# ── Load OTel JSONL traces (block 23 format) ──────────────────────────────────


def load_otel_spans(path: Path) -> list[dict[str, Any]]:
    """Load spans from a FileSpanExporter JSONL file."""
    path = Path(path)
    spans: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    spans.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return spans


def features_from_otel_spans(
    spans: list[dict[str, Any]],
    trace_id: str,
) -> TraceFeatures:
    """Extract TraceFeatures from OTel spans for a given trace_id.

    Inspects ``tool.execute`` spans for tool names and status, and
    ``agent.run`` / ``agent.step`` spans for step counts and tokens.
    """
    trace_spans = [s for s in spans if s.get("trace_id") == trace_id]
    tool_calls: dict[str, int] = {}
    tool_errors = 0
    steps = 0
    input_tokens = 0
    output_tokens = 0
    hit_max = False

    for span in trace_spans:
        name = span.get("name", "")
        attrs = span.get("attributes", {})

        if name == "tool.execute":
            tool_name = attrs.get("tool.name", "unknown")
            tool_calls[tool_name] = tool_calls.get(tool_name, 0) + 1
            if attrs.get("tool.status") == "error":
                tool_errors += 1

        elif name == "agent.step":
            steps += 1

        elif name == "gen_ai.chat":
            input_tokens += int(attrs.get("gen_ai.usage.input_tokens", 0))
            output_tokens += int(attrs.get("gen_ai.usage.output_tokens", 0))

        elif name == "agent.run":
            steps_used = attrs.get("agent.steps_used")
            max_steps = attrs.get("agent.max_steps")
            if steps_used is not None and max_steps is not None and int(steps_used) >= int(max_steps):
                hit_max = True

    search_calls = sum(v for k, v in tool_calls.items() if k in _SEARCH_TOOL_NAMES)
    used_code = any(k in _CODE_TOOL_NAMES for k in tool_calls)

    return TraceFeatures(
        task_id=trace_id,
        steps=steps,
        tool_calls=tool_calls,
        tool_errors=tool_errors,
        hit_max_steps=hit_max,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        used_code=used_code,
        search_calls=search_calls,
    )
