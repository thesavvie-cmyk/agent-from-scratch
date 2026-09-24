# agent-from-scratch

An AI agent framework built from scratch, chapter by chapter, following
"Build an AI Agent from Scratch" (Manning, 2025).

The repository contains a complete, production-ready agent framework with
web search, code execution, long-term memory, multi-agent orchestration,
OpenTelemetry tracing, and an LLM-as-judge eval pipeline with CI/CD gating.

## Quick start

```bash
# Copy and fill in API keys
cp .env.example .env

# Run the agent
uv run agent ask "What is the capital of France?"
uv run agent chat
uv run agent chat --session <session-id>   # resume a persistent session

# Health check
uv run agent doctor
```

### Required environment variables

```
ANTHROPIC_API_KEY=sk-ant-...
TAVILY_API_KEY=tvly-...        # web search
E2B_API_KEY=e2b_...            # code execution sandbox (optional)
```

## Running tests

```bash
uv run pytest                  # unit tests (no network)
uv run pytest -m live          # live tests (requires API keys)
uv run pytest tests/test_gate.py  # eval gate tests
```

## Eval pipeline

The project includes a full LLM-as-judge evaluation framework.

```bash
# Run GAIA benchmark (20 tasks)
uv run --group eval python experiments/ch08_gaia.py --tasks 20 --config baseline

# Judge results with Sonnet
uv run --group eval python experiments/ch10_eval.py --section a \
    --results results/ch08_gaia_baseline.json

# Run quality gate (compare to baseline)
uv run python scripts/eval_gate.py \
    --report results/ch10_section_a_report.json \
    --baseline results/eval_baseline.json

# Human labeling + Cohen's kappa
uv run python scripts/label.py --results results/ch08_gaia_baseline.json
uv run python scripts/judge_agreement.py \
    --human results/human_labels.jsonl \
    --judge results/ch10_section_a_report.json
```

**6 eval rubrics:** `answer_relevance`, `source_credibility`,
`format_compliance`, `trajectory_soundness`, `data_provenance`,
`factual_accuracy`.

**Key finding from block 24:** `exact_match=60%` overstated quality —
one case was a false positive because the gold string `"2"` appeared in
the body text while the agent answered `"3"`.
`factual_accuracy=55%` is the honest number.

## CI/CD

Two GitHub Actions workflows in `.github/workflows/`:

| Workflow | Trigger | What it does |
|----------|---------|--------------|
| `ci.yml` | PR / push | Ruff, unit tests, eval gate (15 tasks) |
| `deploy.yml` | push to main | Full eval (20 tasks), then SSH deploy |

The gate exits with code 1 if any rubric drops more than 10% from baseline
or any adversarial must-pass case fails.

```bash
# Update the baseline after a deliberate improvement
uv run python scripts/eval_gate.py \
    --report results/ch10_section_a_report.json \
    --baseline results/eval_baseline.json \
    --update-baseline
git add results/eval_baseline.json
git commit -m "chore: update eval baseline"
```

## Architecture

See [docs/architecture.md](docs/architecture.md) for a full component diagram.

High-level layers:
1. **LLM client** — LiteLLM wrapper for Anthropic models
2. **Agent** — ReAct loop with tool calling, reflection, planning, context compaction
3. **Tools** — web search (MCP/Tavily), code execution (e2b), file I/O, calculator
4. **Memory** — ChromaDB vector store for cross-session recall
5. **Multi-agent** — Sequential/Parallel/Loop workflows, AgentTool, A2A HTTP protocol
6. **Eval** — LLM-as-judge with 6 rubrics, VerdictCache, CI gate

## Session databases

Session history is stored in SQLite.  Default path: `~/.agentkit/sessions.db`.
Override with `SESSION_DB_PATH` in `.env`.

**The database contains conversation history — treat it like user data.**

- It is gitignored (`*.db`, `*.db-shm`, `*.db-wal`)
- It lives outside the repository by default (`~/.agentkit/`) and is never
  committed
- Exclude it explicitly when deploying:
  ```bash
  rsync -av --exclude='*.db*' --exclude='.env' --exclude='results/' \
      . user@host:path/
  ```
- Use `agent sessions --delete-user <id>` to remove all data for a user

## Quality loop

When the eval gate fails in CI, see [docs/quality_loop.md](docs/quality_loop.md)
for the process: collect failures → open code → fix → promote to dataset → update baseline.
