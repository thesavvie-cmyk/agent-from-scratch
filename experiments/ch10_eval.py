"""Chapter 10 eval framework experiments (block 24).

Sections
--------
a) Re-judge GAIA results — load saved ch08 results, run LLM judge on answer_relevance,
   compare to exact-match labels, report how many verdicts changed.

b) Trajectory rubrics — run trajectory_soundness on ch04/ch08 traces, count how many
   cases with a correct exact-match answer had zero tool calls (= model guessed correctly).

c) Pairwise comparison — pick two ch08 configs (baseline vs +code), run judge_pairwise in
   both orderings for each case, measure positional bias.

d) Adversarial cases — run the agent on the 4 adversarial cases from make_custom_dataset(),
   judge with answer_relevance and source_credibility.
   (Requires ANTHROPIC_API_KEY)

Usage
-----
    uv run python experiments/ch10_eval.py --section all
    uv run python experiments/ch10_eval.py --section a
    uv run --group eval python experiments/ch10_eval.py --section a --results results/ch08_gaia_baseline.json

Notes
-----
- Sections a, b, c load saved result files — no API key needed for those.
- Section d requires ANTHROPIC_API_KEY.
- The judge uses SMART_MODEL (Sonnet) even when sections a–c run without the agent.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ── Helpers ────────────────────────────────────────────────────────────────────


def _find_results(pattern: str) -> list[Path]:
    """Find result files matching *pattern* under RESULTS_DIR."""
    return sorted(RESULTS_DIR.glob(pattern))


def _load_legacy(path: Path) -> list[dict[str, Any]]:
    from agentkit.eval.traces import load_legacy_results
    return load_legacy_results(path)


def _print_header(title: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def _make_judge_llm():
    from agentkit.config import SMART_MODEL
    from agentkit.llm import LlmClient
    return LlmClient(SMART_MODEL)


# ── Section A — re-judge GAIA results ─────────────────────────────────────────


async def section_a(results_path: Path) -> None:
    _print_header("Section A — LLM judge vs exact-match on GAIA results")

    legacy = _load_legacy(results_path)
    if not legacy:
        print(f"  No results found at {results_path}")
        return

    print(f"  Loaded {len(legacy)} results from {results_path.name}")

    judge_llm = _make_judge_llm()
    from agentkit.eval.rubrics import ANSWER_RELEVANCE
    from agentkit.eval.runner import judge_legacy_results, verdicts_changed

    cache_path = RESULTS_DIR / "verdict_cache.jsonl"
    report = await judge_legacy_results(
        legacy_results=legacy,
        judge_llm=judge_llm,
        rubrics=[ANSWER_RELEVANCE],
        cache_path=cache_path,
    )

    out_json = RESULTS_DIR / "ch10_section_a_report.json"
    report.save_json(out_json)
    report.save_markdown(RESULTS_DIR / "ch10_section_a_report.md")
    print(f"  Report saved to {out_json.name}")

    comparison = verdicts_changed(legacy, report, rubric_name="answer_relevance")
    total = comparison["same"] + comparison["changed"]
    print("\n  Exact-match vs LLM-judge comparison (answer_relevance):")
    print(f"    Same:    {comparison['same']}/{total}")
    print(f"    Changed: {comparison['changed']}/{total}")
    if comparison["changed_cases"]:
        print("\n  Changed cases:")
        for c in comparison["changed_cases"]:
            print(f"    {c['case_id']}: exact={c['exact']}  llm={c['llm']}")

    # Summary stats
    passed_exact = sum(1 for r in legacy if r.get("correct"))
    passed_llm = sum(
        1 for r in report.results
        if r.rubric_name == "answer_relevance" and r.verdict == "PASS"
    )
    print("\n  Overall pass counts:")
    print(f"    Exact match: {passed_exact}/{len(legacy)}")
    print(f"    LLM judge:   {passed_llm}/{len(legacy)}")


# ── Section B — trajectory rubrics ────────────────────────────────────────────


async def section_b(results_path: Path) -> None:
    _print_header("Section B — Trajectory soundness: correct answers with no tool calls")

    legacy = _load_legacy(results_path)
    if not legacy:
        print(f"  No results found at {results_path}")
        return

    print(f"  Loaded {len(legacy)} results from {results_path.name}")

    from agentkit.eval.traces import features_from_legacy

    # Find correct answers (exact match) with zero tool calls
    guessed_correct = []
    guessed_wrong = []
    tool_assisted = []

    for res in legacy:
        tf = features_from_legacy(res)
        is_correct = bool(res.get("correct", False))
        has_tools = tf.total_tool_calls > 0

        if is_correct and not has_tools:
            guessed_correct.append(res.get("task_id", "?"))
        elif not is_correct and not has_tools:
            guessed_wrong.append(res.get("task_id", "?"))
        else:
            tool_assisted.append(res.get("task_id", "?"))

    total = len(legacy)
    print("\n  Trajectory breakdown:")
    print(f"    Correct + no tools (guessed right): {len(guessed_correct)}/{total}")
    print(f"    Wrong  + no tools (guessed wrong):  {len(guessed_wrong)}/{total}")
    print(f"    Used tools:                         {len(tool_assisted)}/{total}")

    if guessed_correct:
        print("\n  Cases where answer was guessed (no tool calls, correct):")
        for tid in guessed_correct:
            match = next((r for r in legacy if r.get("task_id") == tid), {})
            q = match.get("question", "")[:80]
            pred = match.get("prediction", "")[:60]
            print(f"    {tid}: Q={q!r}  A={pred!r}")

    # Run trajectory_soundness judge on subset that used no tools
    no_tool_cases = [r for r in legacy if not features_from_legacy(r).total_tool_calls]
    if no_tool_cases:
        print(f"\n  Running trajectory_soundness judge on {len(no_tool_cases)} no-tool cases...")
        judge_llm = _make_judge_llm()
        from agentkit.eval.rubrics import TRAJECTORY_SOUNDNESS
        from agentkit.eval.runner import judge_legacy_results

        cache_path = RESULTS_DIR / "verdict_cache.jsonl"
        sub_report = await judge_legacy_results(
            legacy_results=no_tool_cases,
            judge_llm=judge_llm,
            rubrics=[TRAJECTORY_SOUNDNESS],
            cache_path=cache_path,
        )
        fail_count = sum(1 for r in sub_report.results if r.verdict == "FAIL")
        unclear_count = sum(1 for r in sub_report.results if r.verdict == "UNCLEAR")
        print(f"    trajectory_soundness FAIL: {fail_count}/{len(no_tool_cases)}")
        print(f"    trajectory_soundness UNCLEAR: {unclear_count}/{len(no_tool_cases)}")


# ── Section C — pairwise comparison ───────────────────────────────────────────


async def section_c(path_a: Path, path_b: Path) -> None:
    _print_header("Section C — Pairwise: baseline vs +code (positional bias check)")

    results_a = _load_legacy(path_a)
    results_b = _load_legacy(path_b)
    if not results_a or not results_b:
        print("  Need two result files for pairwise comparison.")
        return

    # Build output dicts keyed by task_id
    outputs_a = {r["task_id"]: r.get("prediction", "") for r in results_a}
    outputs_b = {r["task_id"]: r.get("prediction", "") for r in results_b}
    questions = {r["task_id"]: r.get("question", "") for r in results_a}
    common_ids = sorted(set(outputs_a) & set(outputs_b))

    if not common_ids:
        print("  No common task IDs between the two result files.")
        return

    print(f"  Common cases: {len(common_ids)}")
    print(f"  Config A: {path_a.name}")
    print(f"  Config B: {path_b.name}")

    judge_llm = _make_judge_llm()
    from agentkit.eval.dataset import EvalCase
    from agentkit.eval.judge import judge_pairwise
    from agentkit.eval.rubrics import ANSWER_RELEVANCE

    wins = {"A": 0, "B": 0, "TIE": 0, "UNCLEAR": 0}
    first_shown_counts: dict[str, dict[str, int]] = {"A": {}, "B": {}}
    changed_by_order: list[str] = []

    # Run each case twice (both orders) to detect positional bias
    for task_id in common_ids[:10]:  # cap at 10 to limit API cost
        case = EvalCase(id=task_id, input=questions.get(task_id, ""))
        oa = outputs_a[task_id]
        ob = outputs_b[task_id]

        # Run 1
        v1 = await judge_pairwise(judge_llm, case, oa, ob, ANSWER_RELEVANCE)
        # Run 2 (same order — randomization will shuffle internally)
        v2 = await judge_pairwise(judge_llm, case, oa, ob, ANSWER_RELEVANCE)

        wins[v1.winner] = wins.get(v1.winner, 0) + 1

        if v1.winner != v2.winner and v1.winner in ("A", "B") and v2.winner in ("A", "B"):
            changed_by_order.append(task_id)

        fs = v1.first_shown
        first_shown_counts[fs][v1.winner] = first_shown_counts[fs].get(v1.winner, 0) + 1

    print(f"\n  Win counts (answer_relevance, {len(common_ids[:10])} cases × 2 runs):")
    for label in ("A", "B", "TIE", "UNCLEAR"):
        print(f"    {label}: {wins.get(label, 0)}")

    print(f"\n  Cases where verdict flipped between runs: {len(changed_by_order)}")
    if changed_by_order:
        for tid in changed_by_order:
            print(f"    {tid}")

    print("\n  First-shown bias:")
    for shown in ("A", "B"):
        d = first_shown_counts.get(shown, {})
        if d:
            print(f"    First shown {shown}: {d}")


# ── Section D — adversarial cases ─────────────────────────────────────────────


async def section_d() -> None:
    _print_header("Section D — Adversarial cases (requires API key)")

    from agentkit.config import FAST_MODEL, SMART_MODEL
    from agentkit.eval.dataset import make_custom_dataset
    from agentkit.eval.rubrics import ANSWER_RELEVANCE, SOURCE_CREDIBILITY
    from agentkit.eval.runner import EvalRunner
    from agentkit.llm import LlmClient

    ds = make_custom_dataset().filter(category="adversarial")
    print(f"  Adversarial cases: {len(ds)}")
    for case in ds:
        print(f"    [{case.id}] {case.tags}  {case.input[:80]!r}")

    # Build a minimal agent (no search tools — we just test refusal behaviour)
    from agentkit.agent import Agent

    agent = Agent(
        model=LlmClient(FAST_MODEL),
        tools=[],
        max_steps=3,
    )
    judge_llm = LlmClient(SMART_MODEL)

    runner = EvalRunner(
        agent=agent,
        judge_llm=judge_llm,
        rubrics=[ANSWER_RELEVANCE, SOURCE_CREDIBILITY],
        cache_path=RESULTS_DIR / "verdict_cache.jsonl",
    )
    report = await runner.run_dataset(ds, run_id="adversarial")

    out_json = RESULTS_DIR / "ch10_section_d_report.json"
    report.save_json(out_json)
    report.save_markdown(RESULTS_DIR / "ch10_section_d_report.md")
    print(f"\n  Report saved to {out_json.name}")
    print(f"  Overall pass rate: {report.pass_rate_overall:.1%}")

    print("\n  Per-case results:")
    for r in report.results:
        icon = {"PASS": "PASS", "FAIL": "FAIL", "UNCLEAR": "????"}[r.verdict]
        print(f"    [{icon}] {r.case_id} / {r.rubric_name}")
        print(f"           Output: {r.output[:120]!r}")


# ── Entry point ────────────────────────────────────────────────────────────────


async def main(args: argparse.Namespace) -> None:
    section = args.section

    # Locate a default results file for sections a/b/c
    default_results: Path | None = None
    if args.results:
        default_results = Path(args.results)
    else:
        candidates = _find_results("ch08_gaia_baseline*.json") or _find_results("ch08*.json") or _find_results("*.json")
        if candidates:
            default_results = candidates[-1]

    if section in ("a", "all"):
        if default_results and default_results.exists():
            await section_a(default_results)
        else:
            print("Section A: no results file found. Use --results <path>")

    if section in ("b", "all"):
        if default_results and default_results.exists():
            await section_b(default_results)
        else:
            print("Section B: no results file found. Use --results <path>")

    if section in ("c", "all"):
        if args.results_b:
            path_b = Path(args.results_b)
        else:
            candidates = _find_results("ch08_gaia_*code*.json") or _find_results("ch08*.json")
            path_b = candidates[-1] if candidates else None  # type: ignore[assignment]
        if default_results and default_results.exists() and path_b and path_b.exists():
            await section_c(default_results, path_b)
        else:
            print("Section C: need two result files. Use --results A --results-b B")

    if section in ("d", "all"):
        await section_d()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Block 24 eval framework experiments")
    parser.add_argument(
        "--section",
        choices=["a", "b", "c", "d", "all"],
        default="all",
        help="Which section to run",
    )
    parser.add_argument(
        "--results",
        type=str,
        default=None,
        help="Path to legacy results JSON (for sections a/b/c)",
    )
    parser.add_argument(
        "--results-b",
        type=str,
        default=None,
        dest="results_b",
        help="Path to second config results JSON (for section c)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main(args))
