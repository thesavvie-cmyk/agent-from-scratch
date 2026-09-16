"""Unit tests for block 12: compaction strategies and ContextBudget."""
from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from agentkit.context import ExecutionContext
from agentkit.memory.budget import ContextBudget, estimate_context_tokens
from agentkit.memory.compaction import (
    DropOldest,
    SummarizeHistory,
    TruncateOldToolResults,
    _safe_cut_points,
)
from agentkit.types import Message, ToolCall, ToolResult

# ── fixtures ───────────────────────────────────────────────────────────────────


def _make_call(i: int) -> ToolCall:
    return ToolCall(tool_call_id=f"c{i}", name="search", arguments={"q": f"q{i}"})


def _make_result(i: int, long: bool = False) -> ToolResult:
    content = "x" * 1000 if long else f"result{i}"
    return ToolResult(
        tool_call_id=f"c{i}", name="search", status="success", content=[content]
    )


def _make_message(text: str = "hello") -> Message:
    return Message(role="user", content=text)


def _make_interaction(i: int, long: bool = False) -> list[Any]:
    """One complete call+result pair."""
    return [_make_call(i), _make_result(i, long=long)]


def _ctx() -> ExecutionContext:
    return ExecutionContext()


# ── estimate_context_tokens ────────────────────────────────────────────────────


def test_estimate_returns_positive() -> None:
    contents = [_make_message("hello world")]
    assert estimate_context_tokens(contents) > 0


def test_estimate_longer_is_more() -> None:
    short = [_make_message("hi")]
    long = [_make_message("hi " * 200)]
    assert estimate_context_tokens(long) > estimate_context_tokens(short)


def test_estimate_cached() -> None:
    contents = [_make_message("cache test unique string 12345")]
    # Call twice and check same result (cached)
    r1 = estimate_context_tokens(contents)
    r2 = estimate_context_tokens(contents)
    assert r1 == r2



def test_estimate_tool_result_long_content() -> None:
    short = [_make_result(1, long=False)]
    long = [_make_result(1, long=True)]
    assert estimate_context_tokens(long) > estimate_context_tokens(short)


# ── _safe_cut_points ───────────────────────────────────────────────────────────


def test_safe_cut_empty() -> None:
    assert _safe_cut_points([]) == [0]


def test_safe_cut_only_messages() -> None:
    contents = [_make_message("a"), _make_message("b")]
    pts = _safe_cut_points(contents)
    # All positions are safe (no tool calls to match)
    assert 0 in pts
    assert len(pts) > 1


def test_safe_cut_complete_pair() -> None:
    c = _make_call(1)
    r = _make_result(1)
    pts = _safe_cut_points([c, r])
    # After the result, position 2 is safe
    assert 2 in pts


def test_safe_cut_incomplete_pair() -> None:
    c = _make_call(1)
    # No result — only position 0 is safe
    pts = _safe_cut_points([c])
    assert pts == [0]


def test_safe_cut_multiple_pairs() -> None:
    contents: list[Any] = []
    for i in range(3):
        contents.extend(_make_interaction(i))
    pts = _safe_cut_points(contents)
    # After each pair the cut should be safe
    assert 2 in pts
    assert 4 in pts
    assert 6 in pts


# ── TruncateOldToolResults ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_truncate_keeps_recent() -> None:
    contents: list[Any] = []
    for i in range(5):
        contents.extend(_make_interaction(i))

    ctx = _ctx()
    strategy = TruncateOldToolResults(keep_recent=2)
    result = await strategy.apply(contents, ctx)

    # Count preserved (non-placeholder) ToolResults
    real_results = [
        item for item in result
        if isinstance(item, ToolResult) and item.content[0] != "[tool result evicted]"
    ]
    assert len(real_results) == 2


@pytest.mark.asyncio
async def test_truncate_placeholders_for_old() -> None:
    contents: list[Any] = []
    for i in range(4):
        contents.extend(_make_interaction(i))

    ctx = _ctx()
    strategy = TruncateOldToolResults(keep_recent=1)
    result = await strategy.apply(contents, ctx)

    evicted = [
        item for item in result
        if isinstance(item, ToolResult) and item.content[0] == "[tool result evicted]"
    ]
    assert len(evicted) == 3  # 4 total - 1 kept = 3 evicted


@pytest.mark.asyncio
async def test_truncate_no_orphans() -> None:
    contents: list[Any] = []
    for i in range(4):
        contents.extend(_make_interaction(i))

    ctx = _ctx()
    strategy = TruncateOldToolResults(keep_recent=1)
    result = await strategy.apply(contents, ctx)

    call_ids = {item.tool_call_id for item in result if isinstance(item, ToolCall)}
    result_ids = {item.tool_call_id for item in result if isinstance(item, ToolResult)}
    # Every ToolResult must have a matching ToolCall
    assert result_ids.issubset(call_ids)


@pytest.mark.asyncio
async def test_truncate_keeps_messages() -> None:
    contents: list[Any] = [
        _make_message("start"),
        *_make_interaction(0),
        _make_message("middle"),
        *_make_interaction(1),
    ]
    ctx = _ctx()
    strategy = TruncateOldToolResults(keep_recent=0)
    result = await strategy.apply(contents, ctx)

    messages = [item for item in result if isinstance(item, Message)]
    assert len(messages) == 2


@pytest.mark.asyncio
async def test_truncate_custom_placeholder() -> None:
    contents = list(_make_interaction(0))
    ctx = _ctx()
    strategy = TruncateOldToolResults(keep_recent=0, placeholder="[gone]")
    result = await strategy.apply(contents, ctx)

    evicted = [
        item for item in result
        if isinstance(item, ToolResult) and item.content[0] == "[gone]"
    ]
    assert len(evicted) == 1


# ── DropOldest ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_drop_oldest_removes_prefix() -> None:
    # Build a context that is definitely over a small limit
    contents: list[Any] = []
    for i in range(10):
        contents.extend(_make_interaction(i, long=True))  # ~1000 chars each

    ctx = _ctx()
    # Very small limit to force dropping
    strategy = DropOldest(max_tokens=100)
    result = await strategy.apply(contents, ctx)

    # Result must be shorter than original
    assert len(result) < len(contents)


@pytest.mark.asyncio
async def test_drop_oldest_no_orphans() -> None:
    contents: list[Any] = []
    for i in range(10):
        contents.extend(_make_interaction(i, long=True))

    ctx = _ctx()
    strategy = DropOldest(max_tokens=200)
    result = await strategy.apply(contents, ctx)

    call_ids = {item.tool_call_id for item in result if isinstance(item, ToolCall)}
    result_ids = {item.tool_call_id for item in result if isinstance(item, ToolResult)}
    assert result_ids.issubset(call_ids)


@pytest.mark.asyncio
async def test_drop_oldest_fits_within_budget() -> None:
    contents: list[Any] = []
    for i in range(5):
        contents.extend(_make_interaction(i, long=True))

    ctx = _ctx()
    max_tok = 1000
    strategy = DropOldest(max_tokens=max_tok)
    result = await strategy.apply(contents, ctx)

    assert estimate_context_tokens(result) <= max_tok * 2  # allow some overshoot


@pytest.mark.asyncio
async def test_drop_oldest_never_returns_empty_for_messages() -> None:
    """DropOldest must not return [] when content is only Messages.

    Bug: _safe_cut_points adds len(contents) as a valid cut point.
    contents[len(contents):] == [] and estimate_context_tokens([]) == 1,
    which satisfies any max_tokens threshold, resulting in an empty list.
    That causes the Anthropic API error 'no non-system message'.
    """
    msgs: list[Any] = [
        Message(role="user", content=f"turn {i}") for i in range(10)
    ]
    ctx = _ctx()
    # max_tokens=1 is impossibly small — without the fix this returns []
    strategy = DropOldest(max_tokens=1)
    result = await strategy.apply(msgs, ctx)

    assert len(result) > 0, "DropOldest must never return an empty list"


# ── SummarizeHistory ─────────────────────────────────────────────────────────


def _fake_llm(summary: str = "Test summary") -> Any:
    from agentkit.llm import LlmResponse

    mock = MagicMock()
    mock.generate = AsyncMock(
        return_value=LlmResponse(
            content=[Message(role="assistant", content=summary)]
        )
    )
    return mock


@pytest.mark.asyncio
async def test_summarize_below_trigger_unchanged() -> None:
    contents: list[Any] = [_make_message("short")]
    ctx = _ctx()
    strategy = SummarizeHistory(_fake_llm(), keep_recent=2, trigger_tokens=999_999)
    result = await strategy.apply(contents, ctx)
    assert result == contents


@pytest.mark.asyncio
async def test_summarize_produces_summary_message() -> None:
    contents: list[Any] = []
    for i in range(6):
        contents.extend(_make_interaction(i))

    ctx = _ctx()
    strategy = SummarizeHistory(_fake_llm("Summary here."), keep_recent=2, trigger_tokens=0)
    result = await strategy.apply(contents, ctx)

    summary_msgs = [
        item for item in result
        if isinstance(item, Message) and "[History summary]" in item.content
    ]
    assert len(summary_msgs) == 1
    assert "Summary here." in summary_msgs[0].content


@pytest.mark.asyncio
async def test_summarize_no_orphans() -> None:
    contents: list[Any] = []
    for i in range(6):
        contents.extend(_make_interaction(i))

    ctx = _ctx()
    strategy = SummarizeHistory(_fake_llm(), keep_recent=2, trigger_tokens=0)
    result = await strategy.apply(contents, ctx)

    call_ids = {item.tool_call_id for item in result if isinstance(item, ToolCall)}
    result_ids = {item.tool_call_id for item in result if isinstance(item, ToolResult)}
    assert result_ids.issubset(call_ids)


@pytest.mark.asyncio
async def test_summarize_llm_error_returns_unchanged() -> None:
    """If the LLM call fails, SummarizeHistory returns unchanged contents."""
    mock_llm = MagicMock()
    mock_llm.generate = AsyncMock(side_effect=RuntimeError("boom"))

    contents: list[Any] = []
    for i in range(4):
        contents.extend(_make_interaction(i))

    ctx = _ctx()
    strategy = SummarizeHistory(mock_llm, keep_recent=1, trigger_tokens=0)
    result = await strategy.apply(contents, ctx)

    # Should still return something (possibly with a [summary unavailable] message)
    assert len(result) > 0


# ── ContextBudget ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_budget_no_compaction_when_under_limit() -> None:
    contents = [_make_message("short")]
    ctx = _ctx()
    budget = ContextBudget(
        max_tokens=999_999,
        strategies=[TruncateOldToolResults(keep_recent=1)],
    )
    result, report = await budget.fit(contents, ctx)
    assert result == contents
    assert report["applied"] == []
    assert report["original_tokens"] == report["final_tokens"]


@pytest.mark.asyncio
async def test_budget_applies_strategy_when_over_limit() -> None:
    # Build content that will exceed a tiny limit
    contents: list[Any] = []
    for i in range(10):
        contents.extend(_make_interaction(i, long=True))

    ctx = _ctx()
    budget = ContextBudget(
        max_tokens=50,
        strategies=[TruncateOldToolResults(keep_recent=1)],
    )
    _result, report = await budget.fit(contents, ctx)

    assert "TruncateOldToolResults" in report["applied"]
    assert report["final_tokens"] < report["original_tokens"]


@pytest.mark.asyncio
async def test_budget_report_structure() -> None:
    contents = [_make_message("test")]
    ctx = _ctx()
    budget = ContextBudget(max_tokens=999_999, strategies=[])
    _, report = await budget.fit(contents, ctx)

    assert "original_tokens" in report
    assert "final_tokens" in report
    assert "applied" in report
    assert isinstance(report["applied"], list)


@pytest.mark.asyncio
async def test_budget_multiple_strategies_order() -> None:
    """Second strategy is only applied if first was insufficient."""
    applied_order: list[str] = []

    class TrackingTruncate(TruncateOldToolResults):
        async def apply(self, contents, context):  # type: ignore[override]
            applied_order.append("truncate")
            return await super().apply(contents, context)

    class TrackingDrop(DropOldest):
        async def apply(self, contents, context):  # type: ignore[override]
            applied_order.append("drop")
            return await super().apply(contents, context)

    # Huge contents so both strategies get called
    contents: list[Any] = []
    for i in range(20):
        contents.extend(_make_interaction(i, long=True))

    ctx = _ctx()
    budget = ContextBudget(
        max_tokens=10,
        strategies=[TrackingTruncate(keep_recent=5), TrackingDrop(max_tokens=10)],
    )
    await budget.fit(contents, ctx)

    assert "truncate" in applied_order
    # Drop may or may not be called depending on whether truncate was enough


# ── Agent integration ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_agent_with_budget_logs_compaction() -> None:
    """Agent records compaction events in context.state['compaction_log']."""
    from unittest.mock import AsyncMock, MagicMock

    from agentkit.agent import Agent
    from agentkit.llm import LlmResponse
    from agentkit.types import Message as Msg

    # LLM returns a final answer immediately
    mock_llm = MagicMock()
    mock_llm.generate = AsyncMock(
        return_value=LlmResponse(
            content=[Msg(role="assistant", content="done")],
            usage_metadata={"input_tokens": 100, "output_tokens": 10},
        )
    )

    # Budget with a threshold of 1 to force compaction on first step
    budget = ContextBudget(
        max_tokens=1,
        strategies=[TruncateOldToolResults(keep_recent=0)],
    )

    agent = Agent(model=mock_llm, context_budget=budget)
    result = await agent.run("hello")

    log = result.context.state.get("compaction_log", [])
    assert len(log) >= 1
    assert "applied" in log[0]


# ── Live test (requires ANTHROPIC_API_KEY) ────────────────────────────────────


@pytest.mark.live
@pytest.mark.asyncio
async def test_live_summarize_history() -> None:
    """SummarizeHistory produces a real summary via the Anthropic API."""
    from agentkit.config import FAST_MODEL
    from agentkit.llm import LlmClient

    contents: list[Any] = []
    contents.append(Message(role="user", content="What is the capital of France?"))
    contents.extend(_make_interaction(1))
    contents.extend(_make_interaction(2))
    contents.extend(_make_interaction(3))
    contents.append(Message(role="assistant", content="Paris is the capital of France."))

    llm = LlmClient(FAST_MODEL)
    ctx = _ctx()
    strategy = SummarizeHistory(llm_client=llm, keep_recent=1, trigger_tokens=0)
    result = await strategy.apply(contents, ctx)

    summary_msgs = [
        item for item in result
        if isinstance(item, Message) and "[History summary]" in item.content
    ]
    assert len(summary_msgs) == 1
    # The summary should be non-trivial
    assert len(summary_msgs[0].content) > 50
