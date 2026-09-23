"""Judge–human agreement — Cohen's kappa and disagreement analysis (block 24).

Usage
-----
    uv run python scripts/judge_agreement.py \\
        --human results/human_labels.jsonl \\
        --judge results/eval_report.json \\
        --rubric answer_relevance

Output
------
Prints:
  - Agreement % (raw)
  - Cohen's kappa (κ)
  - Disagreement table (case_id, human, judge)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ── Loaders ───────────────────────────────────────────────────────────────────


def _load_human(path: Path, rubric: str) -> dict[str, str]:
    """Return {case_id: verdict} from human label JSONL."""
    labels: dict[str, str] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if entry.get("rubric", "") == rubric:
                    cid = entry.get("case_id", "")
                    v = entry.get("verdict", "")
                    if cid and v in ("PASS", "FAIL"):
                        labels[cid] = v
            except json.JSONDecodeError:
                pass
    return labels


def _load_judge(path: Path, rubric: str) -> dict[str, str]:
    """Return {case_id: verdict} from eval report JSON."""
    data = json.loads(path.read_text(encoding="utf-8"))
    verdicts: dict[str, str] = {}
    for r in data.get("results", []):
        if r.get("rubric_name") == rubric and r.get("verdict") in ("PASS", "FAIL"):
            verdicts[r["case_id"]] = r["verdict"]
    return verdicts


# ── Cohen's kappa ─────────────────────────────────────────────────────────────


def cohen_kappa(labels_a: list[str], labels_b: list[str]) -> float:
    """Compute Cohen's kappa for two annotators with PASS/FAIL labels.

    κ = (P_o - P_e) / (1 - P_e)
    """
    n = len(labels_a)
    if n == 0:
        return 0.0

    agree = sum(a == b for a, b in zip(labels_a, labels_b))
    p_o = agree / n

    a_pass = labels_a.count("PASS") / n
    a_fail = labels_a.count("FAIL") / n
    b_pass = labels_b.count("PASS") / n
    b_fail = labels_b.count("FAIL") / n

    p_e = a_pass * b_pass + a_fail * b_fail

    if p_e >= 1.0:
        return 1.0
    return (p_o - p_e) / (1 - p_e)


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Judge–human agreement analysis")
    parser.add_argument("--human", required=True, type=Path, help="Human labels JSONL")
    parser.add_argument("--judge", required=True, type=Path, help="Eval report JSON")
    parser.add_argument("--rubric", default="answer_relevance", help="Rubric to compare")
    args = parser.parse_args()

    human = _load_human(args.human, args.rubric)
    judge = _load_judge(args.judge, args.rubric)

    # Only cases labelled by both
    common = sorted(set(human) & set(judge))
    if not common:
        print(f"No overlapping cases for rubric '{args.rubric}'.")
        return

    h_labels = [human[cid] for cid in common]
    j_labels = [judge[cid] for cid in common]

    agree_n = sum(h == j for h, j in zip(h_labels, j_labels))
    agree_pct = agree_n / len(common)
    kappa = cohen_kappa(h_labels, j_labels)

    print(f"\nRubric: {args.rubric}")
    print(f"Cases compared: {len(common)}")
    print(f"Agreement:      {agree_n}/{len(common)} = {agree_pct:.1%}")
    print(f"Cohen's kappa:  κ = {kappa:.3f}")

    # Kappa interpretation
    if kappa < 0.0:
        interp = "worse than chance"
    elif kappa < 0.20:
        interp = "slight"
    elif kappa < 0.40:
        interp = "fair"
    elif kappa < 0.60:
        interp = "moderate"
    elif kappa < 0.80:
        interp = "substantial"
    else:
        interp = "almost perfect"
    print(f"                ({interp} agreement)")

    # Disagreements
    disagreements = [
        (cid, human[cid], judge[cid])
        for cid in common
        if human[cid] != judge[cid]
    ]
    if not disagreements:
        print("\nNo disagreements.")
        return

    print(f"\nDisagreements ({len(disagreements)}):")
    print(f"  {'Case ID':<30}  {'Human':<6}  {'Judge':<6}")
    print("  " + "-" * 48)
    for cid, h, j in disagreements:
        print(f"  {cid:<30}  {h:<6}  {j:<6}")


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    main()
