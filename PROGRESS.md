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
| 22    | A2A protocol — agent interoperability over HTTP | done | AgentCard (/.well-known/agent.json), TaskStatus lifecycle, A2AServer (aiohttp), A2AClient, A2ATool; JSON-RPC 2.0 tasks/send + tasks/get; security: Bearer auth middleware, _RateLimiter (sliding window per IP), BudgetGuard integration; servers/a2a_agent.py standalone deployable server (search+files+code); agent doctor updated with A2A check; 27 unit tests green (--group a2a); section a re-run confirmed researcher used 2 real search calls (tools=2, was 0 with broken session); HTTP ~1.7ms overhead vs in-process; no API key for tests/experiments |
| ch8   | Full GAIA 20-task run (ch08_gaia.py) | done | baseline=60% HitMax=6, +refl=55% HitMax=8, +code=65% HitMax=5, +code+refl=not run (credits); all differences ≤2 tasks -- within noise at n=20; block-to-block comparison invalid (prompt changed); informative results came from point experiments with order-of-magnitude effects (46-city loop, 10-search batching), not from GAIA %diff |
| 20    | Multi-agent workflows              | done | SequentialWorkflow, ParallelWorkflow (return_exceptions), LoopWorkflow (max_iterations); WorkflowStep(share_context); WorkflowResult.all_events deduplicates by id; agents/specialists.py: make_researcher/coder/writer/reviewer; 22 unit tests green (mocked, no API key); experiments ch09_workflow.py a-e ready (needs API key to run) |
| 21    | Agent as Tool + Transfer           | done | AgentTool(BaseTool): child isolation, input_schema validation, child_traces in parent state, ChildAgentError→status=error, recursion+depth protection (_agent_call_stack propagated); TransferOrchestrator: shared context, transfer_to tool injected, ping-pong limit (max_transfers), transfer_log; 27 unit tests green (mocked); ping-pong transfer limit verified on mocks only — Haiku answers directly without calling transfer_to unconditionally; experiments ch09_orchestration.py a-e ready (needs API key) |
| 23    | OpenTelemetry observability        | done | telemetry.py: _NoOpTracer (zero-cost default), setup_tracing, FileSpanExporter→JSONL, add_span_event, reset_tracing; spans: agent.run/step, gen_ai.chat (GenAI conventions), tool.execute, workflow.sequential/parallel/loop.run, sandbox.create/destroy, context.compacted event; AgentTool same trace_id verified; asyncio.gather same trace_id verified; 13 unit tests (--group tracing) incl. CAPTURE_CONTENT=false security test; scripts/trace_tree.py; docs/tracing.md; bug fixed: ch09 experiments closed McpToolset before sections ran → 0.8ms span was session-already-closed exception, not real Tavily timing; fixed by moving section calls inside async with block; SequentialWorkflow.output_step added |
| 25    | CI/CD with eval gate (chapter 10.4) — FINAL BLOCK | done | eval/gate.py: GateConfig (tolerance=0.10 — 1 task noise passes, 2+ task regression blocks on n=15), check_gate (must_pass_ids, per-rubric drop vs baseline, absolute thresholds); scripts/eval_gate.py: CLI for CI (exit 0/1, --update-baseline, --github-summary → $GITHUB_STEP_SUMMARY); .github/workflows/ci.yml: ruff+pytest on every PR, eval gate on src/** changes only (path filter saves credits), uploads artifacts; .github/workflows/deploy.yml: full eval → gate → SSH rsync deploy → agent doctor healthcheck → rollback on failure; scripts/collect_failures.py, promote_to_dataset.py; docs/quality_loop.md, docs/architecture.md; README.md updated; 19 gate unit tests including deliberate-degradation test (80%→0% blocked exit=1, same report exit=0 verified); 589 total tests green, ruff clean |
| 24    | LLM-as-judge eval framework        | done | eval/dataset.py: EvalCase, EvalDataset (JSONL I/O, filter, split), gaia_to_eval_cases, make_custom_dataset (20 cases: 10 core/5 edge/4 adversarial+1 empty); eval/traces.py: TraceFeatures, extract_features, load/extract from legacy results and OTel JSONL; eval/rubrics.py: 6 rubrics (answer_relevance, source_credibility, format_compliance, trajectory_soundness, data_provenance, factual_accuracy) — last 3 require trace or gold; eval/judge.py: judge_single (reasoning-before-verdict), judge_pairwise (randomized order, first_shown recorded for positional-bias detection), VerdictCacheKey (SHA256[:16]); eval/runner.py: EvalRunner (async, per-rubric judge concurrency), VerdictCache (JSONL, persists across runs), EvalReport (JSON+markdown), _compute_metrics (overall/by-rubric/by-category pass rates), judge_legacy_results, verdicts_changed; scripts/open_coding.py, label.py, judge_agreement.py (Cohen's kappa); experiments/ch10_eval.py sections a-d; 73 unit tests green, ruff clean; sections a-c need saved results file, section d needs API key; manual labeling 20 GAIA cases (seed=42): κ=0.876 (almost perfect), 1 disagreement (7d4a7d1d: human labeling error corrected); exact_match=60% vs factual_accuracy=55% — 23dd907f is false PASS in exact_match (agent said 3, gold 2, "2" appeared in body text); 3 cases relevant-but-factually-wrong caught by factual_accuracy but not answer_relevance; ch08_gaia.py fixed: full prediction saved (was [:200]), events serialized (model_dump), OTel tracing by default, utf-8 encoding; indirect injection (custom-018) fixed: benign user input, malicious payload in tool result |

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
