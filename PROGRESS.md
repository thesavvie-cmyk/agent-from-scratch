# Progress

One line per block. Updated in the same commit as the code.
Full GAIA results are in `results/` (gitignored).

| Block | Topic | Status | Notes |
|-------|-------|--------|-------|
| 1-3   | LLM client, tool calling, manual ReAct | done | Foundation |
| 4     | Web search, schema gen, simple_agent_loop | done | GAIA baseline established |
| 5     | MCP client, McpToolset, async agent loop | done | |
| 6     | Types, ExecutionContext, transcript | done | |
| 7     | BaseTool, FunctionTool, @tool, LlmClient | done | |
| 8     | Agent ReAct loop, structured output, GAIA runner | done | haiku+tools: ~35% acc; 10 tasks hit max_steps=8 |
| 9     | CLI, diagnostics, budget guard, bench, deploy | done | |
| 10    | Vector search -- embeddings, chunking, retrieval | done | EMBEDDING_PROVIDER=local default |
| 11    | File tools, document analysis, callbacks | done | |
| 12    | Context compaction -- Truncate, Summarize, DropOldest, ContextBudget | done | GAIA threshold 12K |
| 13    | Sessions and state persistence between runs | done | |
| 14    | Long-term memory with ChromaDB | done | cross-session recall verified |
| 15    | Planning and think-first for agents | done | haiku+tools: 50% / 7 hit_max; haiku+tools+plan: 45% / 7 hit_max -- planning did not reduce looping |
| 16    | Reflection and replan integration | done | full GAIA 20 tasks: HitMax 6→2 (refl), 6→2 (plan+refl); acc 50% all configs; DupCalls=0; reflection used as ERROR ANALYSIS + SELF CHECK before marking unsolvable |
| 17    | Code execution via e2b sandbox | done | execute_python tool; sandbox lifecycle in Agent.run finally; sec(d): search-only=0/46 (hit max), search+code=46/46, code-only=43/46; sandbox creation ~600ms |
| 18    | Tool bridge + workspace tools  | done | env bridge (Tavily direct HTTPS, key as env var) + http bridge (local/Docker); sec(a): 10 searches in 1 execute_python=16.6s/1 round vs 10 agent rounds=20.2s/13 steps/30 calls; GAIA smoke(3): code solved Kipchoge calc (baseline missed); workspace tools: run_command, write/read/list_sandbox_file |
| 19    | Agent skills                   | done | SkillInfo, parse_frontmatter (PyYAML), discover_skills, format_skills_for_prompt, make_read_skill_tool; Agent(skills_dir); 3 real skills (web-research, gaia-file-analysis, data-extraction) with scripts; L1=157tok vs 10 tools=1302tok; at 100 tools L1+read saves 79%; sec(d): agent never reads skills for trivial tasks |
| ch8   | Full GAIA 20-task run (ch08_gaia.py) | done | baseline=60% HitMax=6, +refl=55% HitMax=8, +code=65% HitMax=5, +code+refl=not run (credits); all differences ≤2 tasks -- within noise at n=20; block-to-block comparison invalid (prompt changed); informative results came from point experiments with order-of-magnitude effects (46-city loop, 10-search batching), not from GAIA %diff |
| 20    | Multi-agent workflows              | done | SequentialWorkflow, ParallelWorkflow (return_exceptions), LoopWorkflow (max_iterations); WorkflowStep(share_context); WorkflowResult.all_events deduplicates by id; agents/specialists.py: make_researcher/coder/writer/reviewer; 22 unit tests green (mocked, no API key); experiments ch09_workflow.py a-e ready (needs API key to run) |
| 21    | Agent as Tool + Transfer           | done | AgentTool(BaseTool): child isolation, input_schema validation, child_traces in parent state, ChildAgentError→status=error, recursion+depth protection (_agent_call_stack propagated); TransferOrchestrator: shared context, transfer_to tool injected, ping-pong limit (max_transfers), transfer_log; 27 unit tests green (mocked); experiments ch09_orchestration.py a-e ready (needs API key) |
| 23    | OpenTelemetry observability        | done | telemetry.py: _NoOpTracer (zero-cost default), setup_tracing, FileSpanExporter→JSONL, add_span_event, reset_tracing; spans: agent.run/step, gen_ai.chat (GenAI conventions), tool.execute, workflow.sequential/parallel/loop.run, sandbox.create/destroy, context.compacted event; AgentTool same trace_id verified; asyncio.gather same trace_id verified; 13 unit tests (--group tracing) incl. CAPTURE_CONTENT=false security test; scripts/trace_tree.py; docs/tracing.md |

## Ждёт прогона (нужен API-ключ Anthropic + Tavily)

```
# Блок 20 — workflow experiments
uv run python experiments/ch09_workflow.py --section all

# Блок 21 — orchestration experiments
uv run python experiments/ch09_orchestration.py --section all

# Live тесты блоков 20–21
uv run pytest -m live tests/test_workflow.py tests/test_orchestration.py -v

# Блок 23 — live tracing (real agent + file exporter)
uv run --group tracing python experiments/ch10_tracing.py --section live
```
