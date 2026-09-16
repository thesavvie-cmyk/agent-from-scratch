"""Unit tests for block 11: Workspace, file tools, callbacks."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from agentkit.agent import Agent
from agentkit.callbacks import Callbacks, SkipTool
from agentkit.callbacks_builtin import make_approval_callback, make_search_compressor
from agentkit.context import ExecutionContext
from agentkit.tools.base import tool
from agentkit.tools.files import (
    Workspace,
    WorkspaceEscapeError,
    read_file,
    search_in_files,
)
from agentkit.types import ToolCall, ToolResult

# ── helpers ────────────────────────────────────────────────────────────────────


def _ctx(ws_path: Path) -> ExecutionContext:
    ctx = ExecutionContext()
    ctx.state["workspace"] = Workspace(ws_path)
    return ctx


def _fake_llm(text: str = "done") -> Any:
    """Return a mock LlmClient whose generate() returns a final text response."""
    from agentkit.llm import LlmResponse
    from agentkit.types import Message

    mock = MagicMock()
    mock.generate = AsyncMock(
        return_value=LlmResponse(
            content=[Message(role="assistant", content=text)]
        )
    )
    return mock


class _FakeEmbProvider:
    name = "fake"
    dimension = 4

    def embed(self, texts: list[str]) -> np.ndarray:
        result = np.zeros((len(texts), 4), dtype=np.float32)
        for i, t in enumerate(texts):
            seed = int(hashlib.md5(t.encode()).hexdigest()[:8], 16) % (2**31)
            rng = np.random.default_rng(seed)
            v = rng.random(4).astype(np.float32)
            result[i] = v / max(np.linalg.norm(v), 1e-10)
        return result


# ── Workspace ──────────────────────────────────────────────────────────────────


def test_workspace_resolve_inside(tmp_path: Path) -> None:
    ws = Workspace(tmp_path)
    sub = tmp_path / "sub"
    sub.mkdir()
    assert ws.resolve("sub") == sub


def test_workspace_resolve_dotdot_raises(tmp_path: Path) -> None:
    inner = tmp_path / "inner"
    inner.mkdir()
    ws = Workspace(inner)
    with pytest.raises(WorkspaceEscapeError):
        ws.resolve("../../etc/passwd")


def test_workspace_absolute_outside_raises(tmp_path: Path) -> None:
    ws = Workspace(tmp_path)
    with pytest.raises(WorkspaceEscapeError):
        ws.resolve("/etc/passwd")


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Creating symlinks requires Developer Mode on Windows",
)
def test_workspace_symlink_outside_raises(tmp_path: Path) -> None:
    ws = Workspace(tmp_path)
    inner = tmp_path / "inner"
    inner.mkdir()
    evil = inner / "link_out"
    evil.symlink_to(tmp_path.parent)  # points outside workspace
    with pytest.raises(WorkspaceEscapeError):
        ws.resolve("inner/link_out/secret")


def test_workspace_nested_path_ok(tmp_path: Path) -> None:
    ws = Workspace(tmp_path)
    deep = tmp_path / "a" / "b" / "c"
    deep.mkdir(parents=True)
    (deep / "file.txt").write_text("hi")
    resolved = ws.resolve("a/b/c/file.txt")
    assert resolved == deep / "file.txt"


# ── read_file ──────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_read_file_basic(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("alpha\nbeta\ngamma\n")
    result = await read_file.execute(_ctx(tmp_path), path="f.txt")
    assert "alpha" in result
    assert "gamma" in result


@pytest.mark.asyncio
async def test_read_file_line_range(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("\n".join(f"L{i}" for i in range(1, 21)) + "\n")
    result = await read_file.execute(_ctx(tmp_path), path="f.txt", start_line=5, end_line=8)
    assert "L5" in result
    assert "L8" in result
    assert "L4" not in result
    assert "L9" not in result


@pytest.mark.asyncio
async def test_read_file_nonexistent_returns_error(tmp_path: Path) -> None:
    result = await read_file.execute(_ctx(tmp_path), path="missing.txt")
    assert "Error" in result
    assert "not found" in result


@pytest.mark.asyncio
async def test_read_file_size_limit_truncates(tmp_path: Path) -> None:
    (tmp_path / "big.txt").write_bytes(b"x\n" * 30_000)  # ~60 KB
    result = await read_file.execute(_ctx(tmp_path), path="big.txt")
    assert "Truncated" in result


@pytest.mark.asyncio
async def test_read_file_escape_blocked(tmp_path: Path) -> None:
    ws_sub = tmp_path / "ws"
    ws_sub.mkdir()
    result = await read_file.execute(_ctx(ws_sub), path="../../secret")
    assert "Error" in result


# ── search_in_files ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_search_literal_hit(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("import sqlite3\nprint('hi')\n")
    (tmp_path / "b.py").write_text("import os\n")
    result = await search_in_files.execute(_ctx(tmp_path), pattern="sqlite3")
    assert "a.py" in result
    assert "sqlite3" in result
    assert "b.py" not in result


@pytest.mark.asyncio
async def test_search_regex_hit(tmp_path: Path) -> None:
    (tmp_path / "code.py").write_text("url = 'http://example.com'\n")
    result = await search_in_files.execute(
        _ctx(tmp_path), pattern=r"https?://\S+", is_regex=True
    )
    assert "http://example.com" in result


@pytest.mark.asyncio
async def test_search_no_matches(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("nothing here\n")
    result = await search_in_files.execute(_ctx(tmp_path), pattern="XYZNOTFOUND")
    assert "No matches" in result


@pytest.mark.asyncio
async def test_search_bad_regex_returns_error(tmp_path: Path) -> None:
    result = await search_in_files.execute(
        _ctx(tmp_path), pattern="[invalid(", is_regex=True
    )
    assert "Error" in result


# ── Callbacks via Agent.act() ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_before_tool_substitutes_args() -> None:
    received: dict[str, Any] = {}

    @tool
    def spy(x: str) -> str:
        """Spy."""
        received["x"] = x
        return f"got:{x}"

    def before(ctx: Any, name: str, args: dict[str, Any]) -> dict[str, Any] | None:
        if name == "spy":
            return {"x": "injected"}
        return None

    agent = Agent(
        model=_fake_llm(),
        tools=[spy],
        callbacks=Callbacks(before_tool=[before]),
    )
    ctx = ExecutionContext()
    calls = [ToolCall(tool_call_id="1", name="spy", arguments={"x": "original"})]
    results = await agent.act(ctx, calls)
    assert received["x"] == "injected"
    assert results[0].status == "success"


@pytest.mark.asyncio
async def test_before_tool_skip_tool_cancels_call() -> None:
    executed = {"called": False}

    @tool
    def dangerous() -> str:
        """Dangerous."""
        executed["called"] = True
        return "did it"

    def before(ctx: Any, name: str, args: dict[str, Any]) -> SkipTool:
        return SkipTool("not allowed")

    agent = Agent(
        model=_fake_llm(),
        tools=[dangerous],
        callbacks=Callbacks(before_tool=[before]),
    )
    ctx = ExecutionContext()
    calls = [ToolCall(tool_call_id="1", name="dangerous", arguments={})]
    results = await agent.act(ctx, calls)
    assert not executed["called"]
    assert results[0].status == "error"
    assert "not allowed" in results[0].content[0]


@pytest.mark.asyncio
async def test_after_tool_substitutes_result() -> None:
    @tool
    def greeter(name: str) -> str:
        """Greet."""
        return f"hello {name}"

    from agentkit.types import ToolResult as TR

    def after(ctx: Any, name: str, result: TR) -> TR:
        return TR(
            tool_call_id=result.tool_call_id,
            name=result.name,
            status="success",
            content=["OVERRIDDEN"],
        )

    agent = Agent(
        model=_fake_llm(),
        tools=[greeter],
        callbacks=Callbacks(after_tool=[after]),
    )
    ctx = ExecutionContext()
    calls = [ToolCall(tool_call_id="1", name="greeter", arguments={"name": "world"})]
    results = await agent.act(ctx, calls)
    assert results[0].content[0] == "OVERRIDDEN"


@pytest.mark.asyncio
async def test_callback_exception_does_not_crash_run() -> None:
    """A callback that raises must not crash agent.run()."""

    def exploding_before(ctx: Any, name: str, args: dict[str, Any]) -> None:
        raise RuntimeError("boom")

    agent = Agent(
        model=_fake_llm("answer"),
        callbacks=Callbacks(before_model=[exploding_before]),
    )
    result = await agent.run("hello")
    # Should still return a result despite callback crash
    assert result.output == "answer"
    assert result.error is None


# ── Approval callback ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_approval_allow() -> None:
    executed = {"called": False}

    @tool
    def risky() -> str:
        """Risky."""
        executed["called"] = True
        return "ok"

    cb = make_approval_callback({"risky"}, prompt_fn=lambda _: "y")
    agent = Agent(
        model=_fake_llm(),
        tools=[risky],
        callbacks=Callbacks(before_tool=[cb]),
    )
    ctx = ExecutionContext()
    await agent.act(ctx, [ToolCall(tool_call_id="1", name="risky", arguments={})])
    assert executed["called"]


@pytest.mark.asyncio
async def test_approval_deny() -> None:
    executed = {"called": False}

    @tool
    def risky() -> str:
        """Risky."""
        executed["called"] = True
        return "ok"

    cb = make_approval_callback({"risky"}, prompt_fn=lambda _: "n")
    agent = Agent(
        model=_fake_llm(),
        tools=[risky],
        callbacks=Callbacks(before_tool=[cb]),
    )
    ctx = ExecutionContext()
    results = await agent.act(ctx, [ToolCall(tool_call_id="1", name="risky", arguments={})])
    assert not executed["called"]
    assert results[0].status == "error"
    assert "Declined" in results[0].content[0]


# ── Search compressor callback ─────────────────────────────────────────────────


def _fake_index_factory() -> Any:
    from agentkit.retrieval import VectorIndex
    return VectorIndex(_FakeEmbProvider())


@pytest.mark.asyncio
async def test_compressor_long_result_compressed() -> None:
    """A result exceeding min_tokens is replaced with compressed chunks."""

    # Build a context that has a ToolCall with a 'query' argument
    ctx = ExecutionContext()
    call = ToolCall(tool_call_id="tc1", name="search_web", arguments={"query": "database"})
    ctx.add_tool_calls([call], author="agent")

    # Create a "long" result (repeat text to exceed min_tokens=5 char threshold)
    long_text = ("The database is PostgreSQL. " * 50) + ("It stores user records. " * 50)
    result = ToolResult(
        tool_call_id="tc1",
        name="search_web",
        status="success",
        content=[long_text],
    )

    cb = make_search_compressor(_fake_index_factory, top_k=2, min_tokens=5)
    replacement = cb(ctx, "search_web", result)

    assert replacement is not None
    assert isinstance(replacement, ToolResult)
    assert "Compressed" in replacement.content[0]


@pytest.mark.asyncio
async def test_compressor_short_result_unchanged() -> None:
    """A result below min_tokens passes through unchanged."""

    ctx = ExecutionContext()
    call = ToolCall(tool_call_id="tc2", name="search_web", arguments={"query": "x"})
    ctx.add_tool_calls([call], author="agent")

    short_result = ToolResult(
        tool_call_id="tc2",
        name="search_web",
        status="success",
        content=["short"],
    )
    cb = make_search_compressor(_fake_index_factory, top_k=2, min_tokens=9999)
    replacement = cb(ctx, "search_web", short_result)
    assert replacement is None
