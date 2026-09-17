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
| 16    | Reflection and replan integration | done | @tool reflection writes need_replan to state; Agent injects replan instruction on next step; smoke GAIA (3 tasks): refl=67%/0 hit_max vs baseline 0%/1 -- reflection looks promising |
