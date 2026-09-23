"""Eval runner — dataset execution, verdict caching, metrics, reports (block 24).

Design decisions
----------------
1. Verdict cache (JSONL) — each verdict is keyed by SHA256(case_id+rubric+output)[:16].
   On re-run the runner checks the cache first; only uncached (case, rubric) pairs are
   sent to the LLM judge.  This keeps re-runs cheap.

2. Agent and judge are separated — the runner calls agent.run() for every case and
   collects the output + trace.  Only afterwards does it invoke the judge per rubric.
   This allows running the agent once and re-judging without re-running the agent.

3. Concurrency — cases are run sequentially to avoid overwhelming the LLM API.
   Within a case, all rubrics are judged concurrently (asyncio.gather).

4. Metrics — pass_rate is computed three ways:
     - overall
     - per rubric
     - per category (core / edge / adversarial)
   Rubrics that require a trace are skipped (verdict=UNCLEAR) when no trace is available.

5. Reports — JSON (machine-readable) and markdown (human-readable) are written together.
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agentkit.eval.judge import judge_single, verdict_cache_key
from agentkit.eval.rubrics import DEFAULT_RUBRICS
from agentkit.eval.traces import TraceFeatures, extract_features

if TYPE_CHECKING:
    from agentkit.agent import Agent
    from agentkit.eval.dataset import EvalCase, EvalDataset
    from agentkit.eval.rubrics import Rubric
    from agentkit.llm import LlmClient


# ── Data types ─────────────────────────────────────────────────────────────────


@dataclass
class EvalResult:
    """Result for one (case, rubric) pair."""

    case_id: str
    category: str
    output: str
    rubric_name: str
    verdict: str          # PASS / FAIL / UNCLEAR
    reasoning: str
    from_cache: bool = False
    trace_features: TraceFeatures | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "case_id": self.case_id,
            "category": self.category,
            "output": self.output,
            "rubric_name": self.rubric_name,
            "verdict": self.verdict,
            "reasoning": self.reasoning,
            "from_cache": self.from_cache,
        }
        if self.trace_features is not None:
            d["trace_summary"] = self.trace_features.summary_text()
        return d


@dataclass
class EvalReport:
    """Aggregated metrics and all individual results."""

    run_id: str
    timestamp: str
    total_cases: int
    total_rubrics: int
    results: list[EvalResult] = field(default_factory=list)

    # Metrics filled in by _compute_metrics()
    pass_rate_overall: float = 0.0
    pass_rate_by_rubric: dict[str, float] = field(default_factory=dict)
    pass_rate_by_category: dict[str, float] = field(default_factory=dict)
    unclear_count: int = 0
    cached_count: int = 0

    # ── serialisation ──────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "total_cases": self.total_cases,
            "total_rubrics": self.total_rubrics,
            "pass_rate_overall": self.pass_rate_overall,
            "pass_rate_by_rubric": self.pass_rate_by_rubric,
            "pass_rate_by_category": self.pass_rate_by_category,
            "unclear_count": self.unclear_count,
            "cached_count": self.cached_count,
            "results": [r.to_dict() for r in self.results],
        }

    def save_json(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def save_markdown(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self._to_markdown(), encoding="utf-8")

    def _to_markdown(self) -> str:
        lines: list[str] = []
        lines.append(f"# Eval Report — {self.run_id}")
        lines.append(f"\n**Timestamp:** {self.timestamp}")
        lines.append(
            f"**Cases:** {self.total_cases}  |  "
            f"**Rubrics:** {self.total_rubrics}  |  "
            f"**Cached verdicts:** {self.cached_count}  |  "
            f"**UNCLEAR:** {self.unclear_count}"
        )
        lines.append(f"\n**Overall pass rate:** {self.pass_rate_overall:.1%}\n")

        # By rubric
        lines.append("## Pass Rate by Rubric\n")
        lines.append("| Rubric | Pass rate |")
        lines.append("|--------|-----------|")
        for rubric, rate in sorted(self.pass_rate_by_rubric.items()):
            lines.append(f"| {rubric} | {rate:.1%} |")

        # By category
        lines.append("\n## Pass Rate by Category\n")
        lines.append("| Category | Pass rate |")
        lines.append("|----------|-----------|")
        for cat, rate in sorted(self.pass_rate_by_category.items()):
            lines.append(f"| {cat} | {rate:.1%} |")

        # Individual results
        lines.append("\n## Individual Results\n")
        for r in self.results:
            icon = {"PASS": "✓", "FAIL": "✗", "UNCLEAR": "?"}.get(r.verdict, "?")
            lines.append(
                f"### {icon} {r.case_id} / {r.rubric_name} — {r.verdict}"
            )
            if r.from_cache:
                lines.append("*(from cache)*")
            lines.append(f"\n**Output (excerpt):** {r.output[:200]!r}")
            if r.reasoning:
                lines.append(f"\n**Reasoning:** {r.reasoning}")
            lines.append("")

        return "\n".join(lines)


# ── Verdict cache ──────────────────────────────────────────────────────────────


class VerdictCache:
    """Simple JSONL-backed verdict cache keyed by SHA256 hash."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._data: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        with self._path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    self._data[entry["key"]] = entry
                except (json.JSONDecodeError, KeyError):
                    pass

    def get(self, key: str) -> dict[str, Any] | None:
        return self._data.get(key)

    def put(self, key: str, verdict: str, reasoning: str, rubric_name: str) -> None:
        entry = {
            "key": key,
            "verdict": verdict,
            "reasoning": reasoning,
            "rubric_name": rubric_name,
        }
        self._data[key] = entry
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ── Runner ─────────────────────────────────────────────────────────────────────


class EvalRunner:
    """Run a dataset through an agent and judge the outputs.

    Parameters
    ----------
    agent:
        The agent to evaluate.  Must have an async ``run(question) -> AgentResult``
        method.  Pass ``None`` to skip agent execution and judge pre-supplied outputs.
    judge_llm:
        LLM client for the judge (should be SMART_MODEL).
    rubrics:
        List of rubrics to apply.  Defaults to DEFAULT_RUBRICS.
    cache_path:
        Path for the JSONL verdict cache.  Defaults to ``results/verdict_cache.jsonl``.
    """

    def __init__(
        self,
        agent: Agent | None,
        judge_llm: LlmClient,
        rubrics: list[Rubric] | None = None,
        cache_path: Path | None = None,
    ) -> None:
        self._agent = agent
        self._judge_llm = judge_llm
        self._rubrics: list[Rubric] = rubrics if rubrics is not None else list(DEFAULT_RUBRICS)
        self._cache = VerdictCache(
            cache_path or Path("results/verdict_cache.jsonl")
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    async def run_dataset(
        self,
        dataset: EvalDataset,
        run_id: str | None = None,
        pre_outputs: dict[str, str] | None = None,
        pre_traces: dict[str, TraceFeatures] | None = None,
    ) -> EvalReport:
        """Evaluate every case in *dataset*.

        Parameters
        ----------
        dataset:
            The dataset to evaluate.
        run_id:
            Human-readable identifier for this run.  Auto-generated if not supplied.
        pre_outputs:
            Optional mapping {case_id → output string}.  When supplied the agent is
            not called for cases whose id is in this dict.
        pre_traces:
            Optional mapping {case_id → TraceFeatures}.  Used for trace-based rubrics.
        """
        import uuid

        run_id = run_id or f"run-{uuid.uuid4().hex[:8]}"
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        all_results: list[EvalResult] = []

        cases = list(dataset)
        for case in cases:
            output, trace = await self._get_output(
                case,
                pre_outputs=pre_outputs or {},
                pre_traces=pre_traces or {},
            )
            case_results = await self._judge_case(case, output, trace)
            all_results.extend(case_results)

        report = EvalReport(
            run_id=run_id,
            timestamp=timestamp,
            total_cases=len(cases),
            total_rubrics=len(self._rubrics),
            results=all_results,
        )
        _compute_metrics(report)
        return report

    async def run_case(
        self,
        case: EvalCase,
        output: str,
        trace: TraceFeatures | None = None,
    ) -> list[EvalResult]:
        """Judge a single (case, output) pair under all rubrics."""
        return await self._judge_case(case, output, trace)

    # ── Internals ──────────────────────────────────────────────────────────────

    async def _get_output(
        self,
        case: EvalCase,
        pre_outputs: dict[str, str],
        pre_traces: dict[str, TraceFeatures],
    ) -> tuple[str, TraceFeatures | None]:
        """Return (output, trace) for a case, running the agent if needed."""
        if case.id in pre_outputs:
            return pre_outputs[case.id], pre_traces.get(case.id)

        if self._agent is None:
            return "", None

        try:
            result = await self._agent.run(case.input)
        except Exception as exc:  # noqa: BLE001
            return f"[Agent error: {exc}]", None

        output = result.output or ""
        trace: TraceFeatures | None = None
        if hasattr(result, "events") and result.events:
            trace = extract_features(
                events=[asdict(e) if hasattr(e, "__dataclass_fields__") else e
                        for e in result.events],
                hit_max_steps=bool(getattr(result, "hit_max_steps", False)),
                input_tokens=int(getattr(result, "input_tokens", 0)),
                output_tokens=int(getattr(result, "output_tokens", 0)),
                task_id=case.id,
            )
        return output, trace

    async def _judge_case(
        self,
        case: EvalCase,
        output: str,
        trace: TraceFeatures | None,
    ) -> list[EvalResult]:
        """Judge one case under all rubrics concurrently."""
        tasks = [
            self._judge_one(case, output, trace, rubric)
            for rubric in self._rubrics
        ]
        return list(await asyncio.gather(*tasks))

    async def _judge_one(
        self,
        case: EvalCase,
        output: str,
        trace: TraceFeatures | None,
        rubric: Rubric,
    ) -> EvalResult:
        """Judge one (case, output, rubric) triple, using cache when available."""
        key = verdict_cache_key(case.id, output, rubric.name)
        cached = self._cache.get(key)

        if cached is not None:
            return EvalResult(
                case_id=case.id,
                category=case.category,
                output=output,
                rubric_name=rubric.name,
                verdict=cached["verdict"],
                reasoning=cached["reasoning"],
                from_cache=True,
                trace_features=trace,
            )

        # Skip trace-based rubrics when no trace is available
        if rubric.requires_trace and trace is None:
            return EvalResult(
                case_id=case.id,
                category=case.category,
                output=output,
                rubric_name=rubric.name,
                verdict="UNCLEAR",
                reasoning="No trace available for this rubric.",
                from_cache=False,
                trace_features=None,
            )

        verdict_obj = await judge_single(
            llm=self._judge_llm,
            case=case,
            output=output,
            trace_features=trace,
            rubric=rubric,
        )

        self._cache.put(key, verdict_obj.verdict, verdict_obj.reasoning, rubric.name)

        return EvalResult(
            case_id=case.id,
            category=case.category,
            output=output,
            rubric_name=rubric.name,
            verdict=verdict_obj.verdict,
            reasoning=verdict_obj.reasoning,
            from_cache=False,
            trace_features=trace,
        )


# ── Metrics ────────────────────────────────────────────────────────────────────


def _compute_metrics(report: EvalReport) -> None:
    """Populate pass rate fields on *report* in-place."""
    results = report.results

    # Overall (exclude UNCLEAR from denominator)
    decided = [r for r in results if r.verdict in ("PASS", "FAIL")]
    report.pass_rate_overall = (
        sum(1 for r in decided if r.verdict == "PASS") / len(decided)
        if decided
        else 0.0
    )
    report.unclear_count = sum(1 for r in results if r.verdict == "UNCLEAR")
    report.cached_count = sum(1 for r in results if r.from_cache)

    # By rubric
    rubric_names = {r.rubric_name for r in results}
    for name in rubric_names:
        subset = [r for r in results if r.rubric_name == name and r.verdict in ("PASS", "FAIL")]
        report.pass_rate_by_rubric[name] = (
            sum(1 for r in subset if r.verdict == "PASS") / len(subset)
            if subset
            else 0.0
        )

    # By category
    categories = {r.category for r in results}
    for cat in categories:
        subset = [r for r in results if r.category == cat and r.verdict in ("PASS", "FAIL")]
        report.pass_rate_by_category[cat] = (
            sum(1 for r in subset if r.verdict == "PASS") / len(subset)
            if subset
            else 0.0
        )


# ── Convenience: judge pre-existing outputs ────────────────────────────────────


async def judge_legacy_results(
    legacy_results: list[dict[str, Any]],
    judge_llm: LlmClient,
    rubrics: list[Rubric] | None = None,
    cache_path: Path | None = None,
) -> EvalReport:
    """Judge outputs from legacy result dicts (ch04/ch08 format).

    This is the entry point for section (a) of ch10_eval.py: re-judging
    saved GAIA results with the LLM judge instead of exact match.
    """
    from agentkit.eval.dataset import EvalCase, EvalDataset
    from agentkit.eval.traces import features_from_legacy

    ds = EvalDataset()
    pre_outputs: dict[str, str] = {}
    pre_traces: dict[str, TraceFeatures] = {}

    for res in legacy_results:
        task_id = res.get("task_id", "")
        question = res.get("question", "")
        gold = res.get("gold", None)
        prediction = res.get("prediction", "")

        case = EvalCase(
            id=task_id,
            input=question,
            expected=gold or None,
            category="core",
            tags=["gaia"],
            metadata={"source": "gaia", "exact_match": res.get("correct", False)},
        )
        ds.add(case)
        pre_outputs[task_id] = prediction or ""
        tf = features_from_legacy(res)
        pre_traces[task_id] = tf

    runner = EvalRunner(
        agent=None,
        judge_llm=judge_llm,
        rubrics=rubrics,
        cache_path=cache_path,
    )
    return await runner.run_dataset(
        dataset=ds,
        run_id="legacy-gaia",
        pre_outputs=pre_outputs,
        pre_traces=pre_traces,
    )


def verdicts_changed(
    legacy_results: list[dict[str, Any]],
    report: EvalReport,
    rubric_name: str = "answer_relevance",
) -> dict[str, Any]:
    """Compare LLM-judge verdicts vs exact-match labels.

    Returns a summary dict with changed/same counts and the list of changed cases.
    """
    exact: dict[str, bool] = {
        r["task_id"]: bool(r.get("correct", False)) for r in legacy_results
    }
    llm_verdicts: dict[str, str] = {
        r.case_id: r.verdict
        for r in report.results
        if r.rubric_name == rubric_name and r.verdict in ("PASS", "FAIL")
    }

    changed = []
    same = 0
    for case_id, exact_ok in exact.items():
        llm_v = llm_verdicts.get(case_id)
        if llm_v is None:
            continue
        exact_v = "PASS" if exact_ok else "FAIL"
        if llm_v != exact_v:
            changed.append({"case_id": case_id, "exact": exact_v, "llm": llm_v})
        else:
            same += 1

    return {
        "rubric": rubric_name,
        "same": same,
        "changed": len(changed),
        "changed_cases": changed,
    }
