"""Human labelling — quick PASS/FAIL annotation for (case, output) pairs (block 24).

Usage
-----
    uv run python scripts/label.py \\
        --cases data/custom_dataset.jsonl \\
        --outputs results/agent_outputs.jsonl \\
        --output results/human_labels.jsonl \\
        --rubric answer_relevance

The script shows up to --limit (default 50) pairs that have not been labelled yet.
The annotator enters P (PASS), F (FAIL), S (skip), or Q (quit).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_jsonl(path: Path) -> list[dict]:
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


def _load_done(output_path: Path, rubric: str) -> set[str]:
    done: set[str] = set()
    if not output_path.exists():
        return done
    with output_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if entry.get("rubric") == rubric:
                    key = f"{entry.get('case_id')}||{entry.get('rubric')}"
                    done.add(key)
            except json.JSONDecodeError:
                pass
    return done


def _append(output_path: Path, entry: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Human PASS/FAIL labelling")
    parser.add_argument("--cases", required=True, type=Path, help="EvalCase JSONL")
    parser.add_argument("--outputs", required=True, type=Path, help="Agent outputs JSONL ({case_id, output})")
    parser.add_argument("--output", required=True, type=Path, help="Output JSONL for labels")
    parser.add_argument("--rubric", default="answer_relevance", help="Rubric name to annotate")
    parser.add_argument("--limit", type=int, default=50, help="Max pairs to label per session")
    args = parser.parse_args()

    cases = {c["id"]: c for c in _load_jsonl(args.cases)}
    outputs_raw = _load_jsonl(args.outputs)
    outputs = {r["case_id"]: r["output"] for r in outputs_raw if "case_id" in r and "output" in r}

    done = _load_done(args.output, args.rubric)

    pairs = []
    for case_id, case in cases.items():
        key = f"{case_id}||{args.rubric}"
        if key not in done and case_id in outputs:
            pairs.append((case_id, case, outputs[case_id]))

    pairs = pairs[: args.limit]

    if not pairs:
        print("Nothing left to label.")
        return

    print(f"Labelling {len(pairs)} pairs for rubric '{args.rubric}'")
    print("Commands: P=PASS  F=FAIL  S=skip  Q=quit\n")

    rubric_hints = {
        "answer_relevance": "Does the response directly address the question?",
        "source_credibility": "Are specific facts attributed to sources?",
        "format_compliance": "Does the response follow explicit format instructions?",
        "trajectory_soundness": "Did the agent use tools before stating specific facts?",
        "data_provenance": "Do the numbers match what the tools returned?",
    }
    hint = rubric_hints.get(args.rubric, "")
    if hint:
        print(f"Rubric hint: {hint}\n")

    for idx, (case_id, case, output) in enumerate(pairs, 1):
        print("─" * 70)
        print(f"[{idx}/{len(pairs)}] {case_id}  (category: {case.get('category', '?')})")
        print(f"\nQ: {case.get('input', '')[:300]}")
        if case.get("expected"):
            print(f"Expected: {case['expected'][:200]}")
        print(f"\nA: {output[:400]}")
        print()

        try:
            raw = input("Label [P/F/S/Q]: ").strip().upper()
        except (EOFError, KeyboardInterrupt):
            print("\nStopped.")
            break

        if raw == "Q":
            break
        if raw == "S":
            continue
        if raw not in ("P", "F"):
            print("  Unknown key, skipping.")
            continue

        verdict = "PASS" if raw == "P" else "FAIL"
        entry = {
            "case_id": case_id,
            "rubric": args.rubric,
            "verdict": verdict,
        }
        _append(args.output, entry)
        print(f"  → {verdict}\n")

    print(f"\nDone.  Labels saved to {args.output}")


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    main()
