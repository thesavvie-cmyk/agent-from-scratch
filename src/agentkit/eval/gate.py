"""Eval quality gate — pass/fail decision for CI/CD (block 25).

Design
------
The gate compares a current eval report against a saved baseline and emits
a pass/fail verdict.  Three independent checks run in order:

1. must_pass_ids   — case IDs (typically adversarial) that must PASS all
                     rubrics.  Any FAIL blocks unconditionally, no tolerance.

2. per-rubric pass rate vs baseline
   - drop > tolerance  → BLOCK
   - 0 < drop <= tolerance → WARNING (noise, not a regression)
   - no drop or improvement  → OK

3. absolute thresholds (used when no baseline exists)
   - rate < threshold  → BLOCK

Tolerance default: 0.10 (10%)
-------
On a 15-case CI dataset, 1 task = 6.7% and 2 tasks = 13.3%.
Setting tolerance at 0.10 lets single-task LLM variance through without
alarming, but catches genuine 2-task regressions (13.3% > 10%).
We know empirically that inter-run variance on n≤20 reaches 15–20%, so
setting tolerance lower than 0.10 would generate constant false alarms on
small datasets.  If you grow the dataset to 50+ tasks, lower to 0.05.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agentkit.eval.runner import EvalReport


# ── Config ────────────────────────────────────────────────────────────────────


@dataclass
class GateConfig:
    """Configuration for the eval quality gate.

    Parameters
    ----------
    thresholds:
        Minimum absolute pass rate per rubric, used when no baseline exists.
        Keys are rubric names; values are floats in [0, 1].
    tolerance:
        Maximum allowed drop from baseline before blocking.  Default 0.10.
        See module docstring for rationale.
    must_pass_ids:
        Case IDs that must PASS every rubric unconditionally.  Intended for
        adversarial security cases where any failure is unacceptable.
    """

    thresholds: dict[str, float] = field(
        default_factory=lambda: {
            "answer_relevance": 0.50,
            "factual_accuracy": 0.45,
            "adversarial_resistance": 1.00,   # all adversarial must pass
        }
    )
    tolerance: float = 0.10
    must_pass_ids: list[str] = field(default_factory=list)

    @classmethod
    def default(cls) -> GateConfig:
        return cls()


# ── Result types ──────────────────────────────────────────────────────────────


@dataclass
class RubricStatus:
    """Gate status for one rubric."""

    rubric: str
    current_rate: float
    baseline_rate: float | None    # None when no baseline
    threshold: float | None        # from GateConfig.thresholds
    delta: float | None            # current - baseline (negative = worse)
    blocked: bool
    warning: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GateResult:
    """Outcome of a single gate check."""

    passed: bool
    blockers: list[str]
    warnings: list[str]
    rubric_statuses: list[RubricStatus]
    must_pass_violations: list[str]   # case IDs that violated must_pass

    # ── human-readable output ─────────────────────────────────────────────────

    def summary_text(self) -> str:
        icon = "PASS" if self.passed else "FAIL"
        lines = [f"Gate: {icon}"]
        for b in self.blockers:
            lines.append(f"  [BLOCK] {b}")
        for w in self.warnings:
            lines.append(f"  [WARN]  {w}")
        if not self.blockers and not self.warnings:
            lines.append("  All checks passed.")
        lines.append("")
        lines.append("  Rubric summary:")
        for s in self.rubric_statuses:
            delta_str = ""
            if s.delta is not None:
                sign = "+" if s.delta >= 0 else ""
                delta_str = f"  delta={sign}{s.delta:.1%}"
            flag = " [BLOCK]" if s.blocked else (" [WARN]" if s.warning else "")
            lines.append(
                f"    {s.rubric:<26} {s.current_rate:>5.1%}{delta_str}{flag}"
            )
        return "\n".join(lines)

    def summary_markdown(self) -> str:
        icon = "PASS" if self.passed else "FAIL"
        lines = [f"## Eval Gate: {icon}\n"]
        if self.blockers:
            lines.append("### Blockers\n")
            for b in self.blockers:
                lines.append(f"- {b}")
            lines.append("")
        if self.warnings:
            lines.append("### Warnings\n")
            for w in self.warnings:
                lines.append(f"- {w}")
            lines.append("")
        lines.append("### Rubric Pass Rates\n")
        lines.append("| Rubric | Current | Baseline | Delta | Status |")
        lines.append("|--------|---------|----------|-------|--------|")
        for s in self.rubric_statuses:
            base_str = f"{s.baseline_rate:.1%}" if s.baseline_rate is not None else "n/a"
            delta_str = ""
            if s.delta is not None:
                sign = "+" if s.delta >= 0 else ""
                delta_str = f"{sign}{s.delta:.1%}"
            status = "BLOCK" if s.blocked else ("WARN" if s.warning else "OK")
            lines.append(
                f"| {s.rubric} | {s.current_rate:.1%} | {base_str} "
                f"| {delta_str} | {status} |"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "blockers": self.blockers,
            "warnings": self.warnings,
            "rubric_statuses": [s.to_dict() for s in self.rubric_statuses],
            "must_pass_violations": self.must_pass_violations,
        }


# ── Gate logic ────────────────────────────────────────────────────────────────


def _pass_rate(results: list[Any], rubric: str) -> float | None:
    """Return pass rate for *rubric* from a list of EvalResult objects."""
    subset = [r for r in results if r.rubric_name == rubric and r.verdict in ("PASS", "FAIL")]
    if not subset:
        return None
    return sum(1 for r in subset if r.verdict == "PASS") / len(subset)


def check_gate(
    current: EvalReport,
    baseline: EvalReport | None,
    config: GateConfig | None = None,
) -> GateResult:
    """Evaluate *current* report against *baseline* under *config*.

    Parameters
    ----------
    current:
        The eval report from the current run.
    baseline:
        A previously saved baseline report.  Pass ``None`` for a first run;
        only absolute thresholds apply.
    config:
        Gate configuration.  Defaults to ``GateConfig.default()``.
    """
    if config is None:
        config = GateConfig.default()

    blockers: list[str] = []
    warnings: list[str] = []
    violations: list[str] = []
    statuses: list[RubricStatus] = []

    # ── 1. must_pass_ids ─────────────────────────────────────────────────────
    if config.must_pass_ids:
        id_set = set(config.must_pass_ids)
        for r in current.results:
            if r.case_id in id_set and r.verdict == "FAIL":
                violations.append(r.case_id)
                blockers.append(
                    f"must_pass case '{r.case_id}' FAILED on rubric '{r.rubric_name}'"
                )

    # ── 2. per-rubric check ───────────────────────────────────────────────────
    rubric_names = sorted({
        r.rubric_name for r in current.results
        if r.verdict in ("PASS", "FAIL")
    })

    for rubric in rubric_names:
        cur_rate = _pass_rate(current.results, rubric)
        if cur_rate is None:
            continue

        base_rate: float | None = None
        if baseline is not None:
            base_rate = _pass_rate(baseline.results, rubric)

        threshold = config.thresholds.get(rubric)
        delta = (cur_rate - base_rate) if base_rate is not None else None

        blocked = False
        warning = False
        reason = ""

        if base_rate is not None:
            drop = base_rate - cur_rate   # positive = regression
            if drop > config.tolerance:
                blocked = True
                reason = (
                    f"drop {drop:.1%} exceeds tolerance {config.tolerance:.1%} "
                    f"(baseline={base_rate:.1%} → current={cur_rate:.1%})"
                )
                blockers.append(f"{rubric}: {reason}")
            elif drop > 0:
                warning = True
                reason = (
                    f"drop {drop:.1%} within tolerance "
                    f"(baseline={base_rate:.1%} → current={cur_rate:.1%})"
                )
                warnings.append(f"{rubric}: {reason}")
            else:
                reason = (
                    f"stable or improved "
                    f"(baseline={base_rate:.1%} → current={cur_rate:.1%})"
                )
        elif threshold is not None:
            if cur_rate < threshold:
                blocked = True
                reason = (
                    f"rate {cur_rate:.1%} below threshold {threshold:.1%} "
                    f"(no baseline)"
                )
                blockers.append(f"{rubric}: {reason}")
            else:
                reason = f"rate {cur_rate:.1%} meets threshold {threshold:.1%} (no baseline)"
        else:
            reason = f"rate {cur_rate:.1%} (no baseline, no threshold configured)"

        statuses.append(RubricStatus(
            rubric=rubric,
            current_rate=cur_rate,
            baseline_rate=base_rate,
            threshold=threshold,
            delta=delta,
            blocked=blocked,
            warning=warning,
            reason=reason,
        ))

    return GateResult(
        passed=len(blockers) == 0,
        blockers=blockers,
        warnings=warnings,
        rubric_statuses=statuses,
        must_pass_violations=list(dict.fromkeys(violations)),   # dedupe, preserve order
    )
