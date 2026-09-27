"""Self-maintenance analysis (block 26).

Reads OTel JSONL traces and eval result files from a time window, then uses
an LLM to synthesize a markdown maintenance report covering:

  * Recurring failure patterns (tool errors, hit_max, wrong answers)
  * High-error tools (tools that appear most often in error events)
  * Potential dataset gaps (FAIL cases not yet in training dataset)
  * Skill / documentation improvement suggestions

Also writes case drafts (JSONL) for cases worth adding to the eval dataset.

No automatic changes are made — the report and drafts are for human review.

Usage
-----
    from agentkit.maintenance import run_maintenance

    await run_maintenance(
        trace_dir=Path("traces/"),
        eval_results=Path("results/ch11_critic_gaia.jsonl"),
        output_report=Path("results/maintenance_report.md"),
        output_drafts=Path("results/maintenance_drafts.jsonl"),
    )
"""
from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Trace loading ─────────────────────────────────────────────────────────────


def _load_spans(trace_dir: Path, since_hours: int = 24) -> list[dict[str, Any]]:
    """Load OTel span dicts from all JSONL files in *trace_dir*.

    Only spans whose start_time is within *since_hours* of now are returned.
    Returns an empty list if no files found (graceful — not an error).
    """
    cutoff = datetime.now(tz=UTC) - timedelta(hours=since_hours)
    spans: list[dict[str, Any]] = []

    for jsonl_path in sorted(trace_dir.glob("*.jsonl")):
        try:
            for line in jsonl_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                span = json.loads(line)
                start_raw = span.get("start_time", "")
                # OTel FileSpanExporter stores start_time as ISO string
                try:
                    start = datetime.fromisoformat(start_raw.rstrip("Z")).replace(
                        tzinfo=UTC
                    )
                    if start < cutoff:
                        continue
                except (ValueError, AttributeError):
                    pass  # include spans with unparseable timestamps
                spans.append(span)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping trace file %s: %s", jsonl_path.name, exc)

    logger.debug("Loaded %d spans from %s", len(spans), trace_dir)
    return spans


# ── Trace analysis ────────────────────────────────────────────────────────────


def _tool_error_counts(spans: list[dict]) -> dict[str, int]:
    """Count how many times each tool emitted an error."""
    counts: dict[str, int] = {}
    for span in spans:
        if span.get("name") != "tool.execute":
            continue
        attrs = span.get("attributes", {})
        if attrs.get("tool.status") == "error":
            tool_name = attrs.get("tool.name", "unknown")
            counts[tool_name] = counts.get(tool_name, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def _hit_max_count(spans: list[dict]) -> int:
    """Count agent.run spans where steps_used >= max_steps."""
    count = 0
    for span in spans:
        if span.get("name") != "agent.run":
            continue
        attrs = span.get("attributes", {})
        steps = attrs.get("agent.steps_used", 0)
        max_s = attrs.get("agent.max_steps", 99)
        if steps >= max_s:
            count += 1
    return count


def _compaction_events(spans: list[dict]) -> int:
    """Count context.compacted events across all spans."""
    count = 0
    for span in spans:
        for evt in span.get("events", []):
            if evt.get("name") == "context.compacted":
                count += 1
    return count


def _agent_errors(spans: list[dict]) -> list[str]:
    """Collect agent.error attribute values from agent.run spans."""
    errors = []
    for span in spans:
        if span.get("name") == "agent.run":
            err = span.get("attributes", {}).get("agent.error")
            if err:
                errors.append(err)
    return errors


# ── Eval result loading ───────────────────────────────────────────────────────


def _load_eval_records(path: Path) -> list[dict]:
    """Load eval records from a JSONL or JSON file."""
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8").strip()
    # Try JSONL first
    if text.startswith("{"):
        records = []
        for line in text.splitlines():
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return records
    # Try JSON array
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "results" in data:
            return data["results"]
    except json.JSONDecodeError:
        pass
    return []


def _extract_fail_cases(records: list[dict]) -> list[dict]:
    """Return eval records with verdict=FAIL that are not yet confirmed as dataset items."""
    return [r for r in records if r.get("verdict") == "FAIL"]


# ── LLM synthesis ─────────────────────────────────────────────────────────────


async def _synthesize_report(
    llm: Any,
    trace_summary: str,
    fail_cases: list[dict],
) -> str:
    """Ask the LLM to write a maintenance report from structured summaries."""
    from agentkit.llm import LlmRequest
    from agentkit.types import Message

    fail_excerpt = ""
    if fail_cases:
        excerpts = []
        for rec in fail_cases[:10]:  # cap at 10 to stay within context
            excerpts.append(
                f"  task={rec.get('task_id', '?')[:20]} "
                f"config={rec.get('config', '?')[:15]} "
                f"q={rec.get('question', '')[:80]} "
                f"ans={rec.get('output', '')[:60]}"
            )
        fail_excerpt = "Recent FAIL cases (up to 10):\n" + "\n".join(excerpts)

    user_msg = (
        "You are a software maintenance analyst for an AI agent system. "
        "Based on the following operational data, write a concise markdown "
        "maintenance report with these sections:\n"
        "1. **Summary** — one paragraph overview\n"
        "2. **Top Issues** — bullet list of the most important problems\n"
        "3. **Tool Health** — which tools are failing and why\n"
        "4. **Dataset Gaps** — which failure patterns need new eval cases\n"
        "5. **Recommendations** — concrete next steps for the team\n\n"
        "---\n"
        f"{trace_summary}\n\n"
        f"{fail_excerpt}\n\n"
        "Write the report now."
    )

    request = LlmRequest(
        instructions=[],
        contents=[Message(role="user", content=user_msg)],
        tools=[],
        tool_choice=None,
    )
    response = await llm.generate(request)
    if response.error_message:
        return f"# Maintenance Report\n\nLLM synthesis failed: {response.error_message}\n"
    for item in response.content:
        from agentkit.types import Message as Msg
        if isinstance(item, Msg):
            return item.content
    return "# Maintenance Report\n\nNo content returned.\n"


# ── Case draft generation ─────────────────────────────────────────────────────


def _make_case_drafts(fail_cases: list[dict]) -> list[dict]:
    """Convert FAIL eval records into draft EvalCase dicts for human review."""
    drafts = []
    seen_questions: set[str] = set()
    for rec in fail_cases:
        q = rec.get("question", "").strip()
        if not q or q in seen_questions:
            continue
        seen_questions.add(q)
        drafts.append({
            "id": f"draft-{rec.get('task_id', 'unknown')}",
            "question": q,
            "gold": rec.get("gold", ""),
            "agent_output": rec.get("output", ""),
            "verdict": "FAIL",
            "config": rec.get("config", ""),
            "source": "maintenance_auto_draft",
            "status": "pending_human_review",
            "notes": rec.get("reasoning", ""),
        })
    return drafts


# ── Public API ────────────────────────────────────────────────────────────────


async def run_maintenance(
    trace_dir: Path | None = None,
    eval_results: Path | None = None,
    output_report: Path | None = None,
    output_drafts: Path | None = None,
    since_hours: int = 24,
    llm: Any | None = None,
) -> dict[str, Any]:
    """Analyse traces + eval results and produce a maintenance report.

    Parameters
    ----------
    trace_dir:
        Directory containing OTel JSONL trace files.  If None or missing,
        trace analysis is skipped.
    eval_results:
        Path to a JSONL eval results file (output of ch10/ch11 experiments).
        If None or missing, no FAIL cases are loaded.
    output_report:
        Where to write the markdown report.  Defaults to
        results/maintenance_report.md.
    output_drafts:
        Where to write the case drafts JSONL.  Defaults to
        results/maintenance_drafts.jsonl.
    since_hours:
        Only include spans from the past N hours (default 24).
    llm:
        LlmClient to use for synthesis.  If None, uses SMART_MODEL.

    Returns a dict with keys: report_path, drafts_path, fail_count, span_count.
    """
    from agentkit.config import SMART_MODEL
    from agentkit.llm import LlmClient

    if llm is None:
        llm = LlmClient(model=SMART_MODEL)

    repo_root = Path(__file__).parent.parent.parent
    results_dir = repo_root / "results"
    results_dir.mkdir(exist_ok=True)

    if output_report is None:
        output_report = results_dir / "maintenance_report.md"
    if output_drafts is None:
        output_drafts = results_dir / "maintenance_drafts.jsonl"

    # ── Load spans ────────────────────────────────────────────────────────────
    spans: list[dict] = []
    if trace_dir and trace_dir.exists():
        spans = _load_spans(trace_dir, since_hours=since_hours)

    tool_errors = _tool_error_counts(spans)
    hit_max = _hit_max_count(spans)
    compactions = _compaction_events(spans)
    agent_errors = _agent_errors(spans)

    trace_summary = (
        f"Trace window: past {since_hours}h — {len(spans)} spans loaded.\n"
        f"HitMax events: {hit_max}\n"
        f"Context compactions: {compactions}\n"
        f"Agent LLM errors: {len(agent_errors)}\n"
        f"Tool errors by tool: "
        + (
            ", ".join(f"{k}={v}" for k, v in tool_errors.items())
            if tool_errors else "none"
        )
    )
    logger.info(trace_summary)

    # ── Load eval records ─────────────────────────────────────────────────────
    eval_records: list[dict] = []
    if eval_results and eval_results.exists():
        eval_records = _load_eval_records(eval_results)
    fail_cases = _extract_fail_cases(eval_records)

    # ── Synthesize report ─────────────────────────────────────────────────────
    report_md = await _synthesize_report(llm, trace_summary, fail_cases)

    # Prepend metadata header
    now = datetime.now(tz=UTC).strftime("%Y-%m-%d %H:%M UTC")
    header = (
        f"<!-- generated by agentkit.maintenance at {now} -->\n"
        f"<!-- spans={len(spans)} fail_cases={len(fail_cases)} -->\n\n"
    )
    full_report = header + report_md

    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(full_report, encoding="utf-8")
    logger.info("Maintenance report → %s", output_report)

    # ── Case drafts ───────────────────────────────────────────────────────────
    drafts = _make_case_drafts(fail_cases)
    output_drafts.parent.mkdir(parents=True, exist_ok=True)
    with output_drafts.open("w", encoding="utf-8") as fh:
        for draft in drafts:
            fh.write(json.dumps(draft, ensure_ascii=False) + "\n")
    logger.info("Case drafts (%d) → %s", len(drafts), output_drafts)

    return {
        "report_path": str(output_report),
        "drafts_path": str(output_drafts),
        "fail_count": len(fail_cases),
        "span_count": len(spans),
        "tool_errors": tool_errors,
        "hit_max": hit_max,
        "draft_count": len(drafts),
    }
