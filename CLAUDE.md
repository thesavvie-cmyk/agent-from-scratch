# Agent from Scratch — Project Rules

This is a learning project. We build an agent framework from scratch following
"Build an AI Agent from Scratch" (Manning). Goal is understanding, not speed.

## Stack & Conventions

- Python 3.13, **uv only** (never pip), src-layout
- Package name: `agentkit` (lives in `src/agentkit/`)
- Type hints everywhere; Pydantic for data models
- LLM provider: **Anthropic only** (via SDK or LiteLLM)
- Code and comments: **English**; reports to user: **Russian**

## Dependencies

Do not add dependencies not mentioned in the current block's task.

## Testing

- Each block ends with tests: unit (no network) + live (`@pytest.mark.live`)
- Unit tests run by default: `uv run pytest`
- Live tests run explicitly: `uv run pytest -m live`

## Reference Code

Do **not** read `../reference` unless the user explicitly asks for a comparison.

## Block Completion

At the end of each block:
1. Commit: `git commit -m "block N: <brief description>"`
2. Report to the user in Russian:
   - What was done
   - Key decisions and reasoning
   - What remains unclear or worth revisiting
