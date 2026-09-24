# Architecture

`agent-from-scratch` is a layered AI agent framework built incrementally
following "Build an AI Agent from Scratch" (Manning).

## Component map

```
┌─────────────────────────────────────────────────────────────┐
│                        CLI / API                            │
│  uv run agent ask / chat / doctor                           │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                      Agent (ReAct loop)                     │
│  src/agentkit/agent.py — Agent.run(question) → AgentResult  │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │  LlmClient   │  │  Tools       │  │  ExecutionContext  │  │
│  │  llm.py      │  │  tools/      │  │  context.py       │  │
│  │  (LiteLLM)   │  │  base.py     │  │  events, state,   │  │
│  └──────────────┘  │  mcp.py      │  │  current_step     │  │
│                    │  file.py     │  └───────────────────┘  │
│                    │  code.py     │                          │
│                    └──────────────┘                          │
└──────────────────────┬──────────────────────────────────────┘
                       │
       ┌───────────────┼───────────────────┐
       │               │                   │
┌──────▼──────┐  ┌─────▼──────┐  ┌────────▼────────┐
│   Memory    │  │  Planning  │  │  Observability  │
│  memory.py  │  │  planning  │  │  telemetry.py   │
│  (ChromaDB) │  │  .py       │  │  (OTel spans)   │
└─────────────┘  └────────────┘  └─────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                   Eval Framework                            │
│  src/agentkit/eval/                                         │
│  ┌──────────┐  ┌─────────┐  ┌──────────┐  ┌──────────────┐ │
│  │ dataset  │  │ rubrics │  │  judge   │  │   runner     │ │
│  │ .py      │  │ .py     │  │  .py     │  │   .py        │ │
│  │ EvalCase │  │ 6       │  │ Verdict  │  │ EvalRunner   │ │
│  │ GAIA     │  │ rubrics │  │ pairwise │  │ VerdictCache │ │
│  └──────────┘  └─────────┘  └──────────┘  └──────────────┘ │
│  ┌──────────┐  ┌──────────┐                                 │
│  │  traces  │  │  gate    │                                  │
│  │  .py     │  │  .py     │                                  │
│  │ OTel+    │  │ GateConfig│                                 │
│  │ legacy   │  │ check_gate│                                 │
│  └──────────┘  └──────────┘                                 │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                   Multi-Agent Layer                         │
│  ┌───────────────┐  ┌──────────────┐  ┌───────────────────┐ │
│  │  Workflows    │  │  AgentTool   │  │  A2A Protocol     │ │
│  │  workflow.py  │  │  tools/      │  │  a2a.py           │ │
│  │  Sequential   │  │  agent_tool  │  │  HTTP JSON-RPC    │ │
│  │  Parallel     │  │  .py         │  │  AgentCard        │ │
│  │  Loop         │  └──────────────┘  │  Bearer auth      │ │
│  └───────────────┘                    └───────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Key data flows

### Single question
```
user input
  → Agent.run(question)
    → LlmClient.generate() [LLM call 1: plan + tool selection]
    → tool.execute() [search / code / file]
    → LlmClient.generate() [LLM call 2: synthesise answer]
    → AgentResult(output, context)
```

### Eval pipeline
```
EvalCase (input + expected)
  → Agent.run()                  [or load pre-run results]
  → judge_single(rubric)         [Sonnet judge, reasoning-before-verdict]
  → VerdictCache.put()           [JSONL cache, keyed by SHA256]
  → EvalReport                  [JSON + markdown]
  → check_gate()                 [pass/fail + blockers + warnings]
  → exit 0/1                     [CI gate]
```

### A2A request
```
external agent
  → POST /tasks/send (JSON-RPC)
    → auth middleware (Bearer)
    → rate limiter (sliding window per IP)
    → BudgetGuard (token budget)
    → Agent.run(task.message)
    → A2ATask response
```

## Directory structure

```
agent-from-scratch/
├── src/agentkit/          # Core library
│   ├── agent.py           # Agent ReAct loop
│   ├── llm.py             # LiteLLM wrapper
│   ├── context.py         # ExecutionContext
│   ├── types.py           # Event, ToolCall, ToolResult, Message
│   ├── tools/             # Tool implementations
│   ├── eval/              # Eval framework
│   ├── memory.py          # ChromaDB long-term memory
│   ├── planning.py        # Think-first planning
│   ├── reflection.py      # Reflection instructions
│   ├── skills.py          # Agent skills (lazy loading)
│   ├── sessions.py        # SQLite session persistence
│   ├── telemetry.py       # OpenTelemetry tracing
│   ├── a2a.py             # A2A protocol (HTTP interop)
│   ├── budget.py          # Token budget guard
│   └── workflow.py        # Sequential/Parallel/Loop workflows
├── experiments/           # Chapter experiments (ch02–ch10)
├── scripts/               # Utility scripts (eval, labeling, CI)
├── tests/                 # Unit + integration tests
├── docs/                  # Architecture, tracing, quality loop
├── skills/                # Agent skill definitions (YAML + Python)
├── data/                  # Dataset files
└── results/               # Eval outputs (gitignored)
```

## Model strategy

| Role | Model | Reason |
|------|-------|--------|
| Agent (acting) | `claude-haiku-4-5` | Fast, cheap for ReAct loops |
| Judge (eval) | `claude-sonnet-4-6` | Better calibration for rubric evaluation |
| Embeddings | `voyage-3-lite` (or local) | Fast retrieval, no GPU required |

The Haiku/Sonnet split is critical: the same model cannot reliably judge
its own outputs.  Using a stronger model as judge avoids self-serving bias.

## Security model

- **Workspace isolation**: all file I/O goes through `Workspace.resolve()`;
  paths outside the workspace root raise `WorkspaceEscapeError` before I/O
- **Code sandboxing**: Python execution runs in e2b cloud sandboxes; no
  local code execution is permitted
- **A2A auth**: Bearer token required for all POST endpoints; configurable
  per-IP rate limiting; BudgetGuard caps token spend
- **Indirect injection**: `MockInjectionSearchTool` tests that agents ignore
  malicious instructions embedded in search results; adversarial rubric
  `adversarial_resistance` gates this in CI
- **Secrets**: all API keys via environment variables / GitHub Secrets;
  never committed; `.env` is gitignored
