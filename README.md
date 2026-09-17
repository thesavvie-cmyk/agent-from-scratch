# agent-from-scratch

Learning project: building an AI agent framework from scratch, following
"Build an AI Agent from Scratch" (Manning).

## Quick start

```bash
uv run agent ask "What is the capital of France?"
uv run agent chat
uv run agent chat --session <session-id>   # resume a persistent session
uv run agent doctor
```

## Running tests

```bash
uv run pytest               # unit tests (no network)
uv run pytest -m live       # live tests (requires ANTHROPIC_API_KEY)
```

## Session databases

Session history is stored in SQLite.  Default path: `~/.agentkit/sessions.db`.
Override with `SESSION_DB_PATH` in `.env`.

**The database contains conversation history — treat it like user data.**

- It is git-ignored (`*.db`, `*.db-shm`, `*.db-wal` in `.gitignore`)
- It lives outside the repository by default (`~/.agentkit/`) and is never
  committed
- If you deploy with rsync or scp, always exclude it explicitly:
  ```bash
  rsync -av --exclude='*.db*' --exclude='.env' src/ user@host:dst/
  ```
- File permissions are set to 600 (owner read/write only) on creation
- Use `agent sessions --delete-user <id>` to remove all data for a user
