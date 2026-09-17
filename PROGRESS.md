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
| 18    | Tool bridge + workspace tools  | done | SandboxBridge (HTTP, stub injection); workspace: run_command, write/read/list_sandbox_file; Agent(sandbox_tools, workspace); ch08_workspace.py (a-d), ch08_gaia.py (4 configs) |
