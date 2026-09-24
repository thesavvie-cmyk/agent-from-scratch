# Quality Loop

The eval framework is only useful if it feeds back into development.
This document describes the process for turning production failures into
better agent behaviour and a stronger test suite.

## When to run the loop

Run after any of these events:
- A PR fails the eval gate
- A production deployment shows unexpected behaviour
- You update a rubric or add a new one
- Once a month as routine maintenance

## Step-by-step

### 1. Collect failures

```bash
# After a gate failure or scheduled run:
uv run python scripts/collect_failures.py \
    --report results/ch10_section_a_report.json \
    --output results/failures.jsonl
```

Prioritise by rubric: `factual_accuracy` failures are usually actionable
(wrong answer); `source_credibility` failures are often fixable by
prompting; `format_compliance` failures hint at prompt drift.

### 2. Open coding — understand the pattern

```bash
uv run python scripts/open_coding.py \
    --traces results/failures.jsonl \
    --output results/open_codes.jsonl
```

For each failure, assign a short label:
- `wrong_tool_choice` — agent searched when it should have computed (or vice versa)
- `hallucinated_fact` — stated something not in any search result
- `format_ignored` — added commentary when a bare answer was requested
- `search_gap` — correct approach, but search returned no useful results
- `max_steps` — ran out of budget without finding the answer
- `ambiguous_question` — question has multiple valid interpretations

Look for the dominant pattern. Three identical labels in five failures
is a signal; one each of five labels is noise.

### 3. Decide what to change

| Pattern | Action |
|---------|--------|
| `wrong_tool_choice` ×3+ | Update agent instructions or add a hint rubric |
| `hallucinated_fact` ×3+ | Strengthen `factual_accuracy` rubric; add search-required instruction |
| `format_ignored` ×3+ | Add a FORMAT instruction to the prompt; verify `format_compliance` rubric |
| `max_steps` ×5+ | Raise `max_steps` or split the task; add a step-budget rubric |
| `search_gap` ×3+ | Improve search query generation; add fallback search strategy |
| `ambiguous_question` | Fix ground truth label or mark as `edge` case |

### 4. Promote failures to the dataset

```bash
uv run python scripts/promote_to_dataset.py \
    --dataset data/custom_dataset.jsonl \
    --from-codes results/open_codes.jsonl \
    --source production
```

Only promote cases with a clear ground truth.  Ambiguous cases (`?`)
go into an `edge` category, not `core`, so they don't distort gate metrics.

### 5. Update the baseline after a deliberate fix

After merging a fix and verifying it improves quality:

```bash
uv run python scripts/eval_gate.py \
    --report results/ch10_section_a_report.json \
    --baseline results/eval_baseline.json \
    --update-baseline
git add results/eval_baseline.json
git commit -m "chore: update eval baseline after <description of fix>"
```

**Never update the baseline to hide a regression.**
If the gate blocks because quality genuinely dropped, fix the agent, not
the baseline.

## Cadence

- **Per PR**: gate runs automatically in CI on any change to `src/` or prompts.
- **Weekly**: run a full 20-task GAIA eval locally; review warnings.
- **Monthly**: open code 10–15 recent failures; update dataset and rubrics.
- **Before a release**: run the full eval + gate + adversarial suite; only
  deploy if gate passes.

## How many cases to look at

At n=20, each case is 5% of the score.  Looking at 5 failures gives you
a reasonable sample.  At n=50+, 10 failures is enough to see patterns.

Do not over-index on single-case differences.  A 5% gate warning on n=20
is likely noise.  A 15% block signal is real.

## Rubric update discipline

Before changing a rubric:
1. Collect 10+ cases where the current rubric gives you the "wrong" verdict.
2. Write a new draft criterion.
3. Apply it to those 10 cases manually.
4. Run the judge with the new rubric on the full dataset.
5. Compare pass rates — if they shift more than 10%, document why.
6. Update the baseline after merging.

Changing a rubric without re-baselining is the fastest way to lose trust
in your metrics.
