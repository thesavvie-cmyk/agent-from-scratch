"""Run maintenance analysis (block 26).

Usage
-----
    uv run python scripts/maintenance_run.py

    # Custom paths:
    uv run python scripts/maintenance_run.py \\
        --traces traces/ \\
        --eval results/ch11_critic_gaia.jsonl \\
        --report results/maintenance_report.md \\
        --drafts results/maintenance_drafts.jsonl \\
        --hours 48

    # Cron / systemd example (run nightly at 02:00):
    #   0 2 * * * cd /opt/agentkit && uv run python scripts/maintenance_run.py
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "src"))


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run agentkit maintenance analysis")
    parser.add_argument(
        "--traces", type=Path, default=REPO / "traces",
        help="Directory with OTel JSONL trace files (default: traces/)",
    )
    parser.add_argument(
        "--eval", type=Path, default=REPO / "results" / "ch11_critic_gaia.jsonl",
        help="Eval results JSONL (default: results/ch11_critic_gaia.jsonl)",
    )
    parser.add_argument(
        "--report", type=Path, default=REPO / "results" / "maintenance_report.md",
        help="Output markdown report path",
    )
    parser.add_argument(
        "--drafts", type=Path, default=REPO / "results" / "maintenance_drafts.jsonl",
        help="Output case drafts JSONL path",
    )
    parser.add_argument(
        "--hours", type=int, default=24,
        help="Include spans from the past N hours (default: 24)",
    )
    args = parser.parse_args()

    from agentkit.maintenance import run_maintenance

    print(f"Trace dir  : {args.traces}")
    print(f"Eval file  : {args.eval}")
    print(f"Time window: past {args.hours}h")
    print()

    summary = await run_maintenance(
        trace_dir=args.traces,
        eval_results=args.eval,
        output_report=args.report,
        output_drafts=args.drafts,
        since_hours=args.hours,
    )

    print(f"Spans loaded   : {summary['span_count']}")
    print(f"FAIL cases     : {summary['fail_count']}")
    print(f"Case drafts    : {summary['draft_count']}")
    if summary["tool_errors"]:
        print("Tool errors    :", summary["tool_errors"])
    print(f"HitMax events  : {summary['hit_max']}")
    print()
    print(f"Report   → {summary['report_path']}")
    print(f"Drafts   → {summary['drafts_path']}")


if __name__ == "__main__":
    asyncio.run(main())
