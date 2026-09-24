"""Eval quality gate CLI — for CI/CD pipelines (block 25).

Loads a saved eval report, compares it to a baseline, and exits with
code 0 (gate passed) or 1 (gate failed).

Usage
-----
    # First run — no baseline yet, only absolute thresholds apply:
    uv run python scripts/eval_gate.py \\
        --report results/ch10_section_a_report.json \\
        --output results/eval_gate_result.json

    # Subsequent runs — compare to baseline:
    uv run python scripts/eval_gate.py \\
        --report results/ch10_section_a_report.json \\
        --baseline results/eval_baseline.json \\
        --output results/eval_gate_result.json

    # Save current report as the new baseline (after a deliberate improvement):
    uv run python scripts/eval_gate.py \\
        --report results/ch10_section_a_report.json \\
        --baseline results/eval_baseline.json \\
        --update-baseline \\
        --output results/eval_gate_result.json

    # GitHub Actions: write step summary markdown:
    uv run python scripts/eval_gate.py \\
        --report results/ch10_section_a_report.json \\
        --baseline results/eval_baseline.json \\
        --github-summary \\
        --output results/eval_gate_result.json

Workflow in CI (ci.yml)
-----------------------
    1. Run the agent:   uv run python experiments/ch10_eval.py --section a ...
    2. Gate the result: uv run python scripts/eval_gate.py --report ... --baseline ...

The baseline file should be committed to the repository so every PR
compares against the same reference.  Use --update-baseline locally
after a deliberate improvement, then commit the updated baseline.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

RESULTS_DIR = Path(__file__).parent.parent / "results"


# ── Helpers ───────────────────────────────────────────────────────────────────


def _load_report(path: Path):
    """Load an EvalReport from JSON — returns a lightweight dict proxy."""
    from agentkit.eval.runner import EvalReport, EvalResult

    data = json.loads(path.read_text(encoding="utf-8"))
    results = [
        EvalResult(
            case_id=r["case_id"],
            category=r.get("category", "core"),
            output=r.get("output", ""),
            rubric_name=r["rubric_name"],
            verdict=r["verdict"],
            reasoning=r.get("reasoning", ""),
            from_cache=r.get("from_cache", False),
        )
        for r in data.get("results", [])
    ]
    return EvalReport(
        run_id=data.get("run_id", ""),
        timestamp=data.get("timestamp", ""),
        total_cases=data.get("total_cases", 0),
        total_rubrics=data.get("total_rubrics", 0),
        results=results,
        pass_rate_overall=data.get("pass_rate_overall", 0.0),
        pass_rate_by_rubric=data.get("pass_rate_by_rubric", {}),
        pass_rate_by_category=data.get("pass_rate_by_category", {}),
    )


def _write_github_summary(markdown: str) -> None:
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with Path(summary_path).open("a", encoding="utf-8") as fh:
            fh.write(markdown + "\n")
    else:
        print("\n--- GitHub Summary (GITHUB_STEP_SUMMARY not set) ---")
        print(markdown)


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Eval quality gate — exits 0 (pass) or 1 (fail)"
    )
    parser.add_argument(
        "--report", required=True, type=Path,
        help="Eval report JSON from the current run (ch10_section_a_report.json)",
    )
    parser.add_argument(
        "--baseline", type=Path, default=None,
        help="Baseline report JSON for comparison.  Omit for first run.",
    )
    parser.add_argument(
        "--output", type=Path, default=RESULTS_DIR / "eval_gate_result.json",
        help="Where to write the gate result JSON.",
    )
    parser.add_argument(
        "--update-baseline", action="store_true",
        help="Copy the current report to --baseline path after evaluation.",
    )
    parser.add_argument(
        "--github-summary", action="store_true",
        help="Write markdown to $GITHUB_STEP_SUMMARY for PR summary.",
    )
    parser.add_argument(
        "--must-pass", default="",
        help="Comma-separated case IDs that must PASS all rubrics.",
    )
    parser.add_argument(
        "--tolerance", type=float, default=0.10,
        help="Max allowed drop from baseline before blocking (default 0.10).",
    )
    args = parser.parse_args()

    # Load reports
    if not args.report.exists():
        print(f"Error: report not found: {args.report}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading report: {args.report}")
    current = _load_report(args.report)

    baseline = None
    if args.baseline and args.baseline.exists():
        print(f"Loading baseline: {args.baseline}")
        baseline = _load_report(args.baseline)
    elif args.baseline:
        print(f"Baseline not found at {args.baseline} — running with absolute thresholds only.")

    # Build config
    from agentkit.eval.gate import GateConfig, check_gate

    must_pass_ids = [x.strip() for x in args.must_pass.split(",") if x.strip()]
    config = GateConfig(tolerance=args.tolerance, must_pass_ids=must_pass_ids)

    # Run gate
    result = check_gate(current, baseline, config)

    # Print human-readable output
    print()
    print(result.summary_text())

    # Write gate result JSON
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nGate result saved to {args.output}")

    # GitHub Actions summary
    if args.github_summary:
        _write_github_summary(result.summary_markdown())

    # Update baseline if requested and gate passed
    if args.update_baseline:
        if args.baseline is None:
            print("Error: --update-baseline requires --baseline path.", file=sys.stderr)
            sys.exit(1)
        if result.passed:
            import shutil
            args.baseline.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(args.report, args.baseline)
            print(f"Baseline updated: {args.baseline}")
        else:
            print("Gate FAILED — baseline NOT updated.", file=sys.stderr)

    sys.exit(0 if result.passed else 1)


if __name__ == "__main__":
    main()
