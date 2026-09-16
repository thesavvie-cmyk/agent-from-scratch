"""Unit tests for block 13: sessions and state persistence."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from agentkit.context import ExecutionContext
from agentkit.memory.session import InMemorySessionStore, Session, SqliteSessionStore
from agentkit.types import Event, Message, ToolCall, ToolResult


# ── fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture(params=["memory", "sqlite"])
def store(request: pytest.FixtureRequest, tmp_path: Path):
    """Parametrized fixture: both store implementations."""
    if request.param == "memory":
        return InMemorySessionStore()
    else:
        db = str(tmp_path / "test_sessions.db")
        return SqliteSessionStore(db_path=db)


def _make_event(role: str = "user", content: str = "hello") -> Event:
    return Event(
        execution_id="exec-1",
        author="test",
        content=[Message(role=role, content=content)],
    )


def _make_tool_event(i: int) -> Event:
    """Event with a ToolCall + ToolResult pair (two separate events in practice,
    but grouped here for brevity in round-trip tests)."""
    tc = ToolCall(tool_call_id=f"c{i}", name="search", arguments={"q": f"q{i}"})
    return Event(execution_id="exec-1", author="agent", content=[tc])


def _make_tool_result_event(i: int) -> Event:
    tr = ToolResult(
        tool_call_id=f"c{i}", name="search", status="success", content=[f"result{i}"]
    )
    return Event(execution_id="exec-1", author="agent", content=[tr])


# ── Session dataclass ─────────────────────────────────────────────────────────


def test_session_defaults() -> None:
    s = Session()
    assert s.user_id == "default"
    assert s.events == []
    assert s.state == {}
    assert s.metadata == {}
    assert len(s.session_id) == 36  # UUID


def test_session_unique_ids() -> None:
    assert Session().session_id != Session().session_id


# ── Store: create / get ────────────────────────────────────────────────────────


def test_create_returns_session(store: Any) -> None:
    s = store.create(user_id="alice")
    assert s.user_id == "alice"
    assert s.session_id


def test_get_returns_created_session(store: Any) -> None:
    s = store.create(user_id="alice")
    loaded = store.get(s.session_id)
    assert loaded is not None
    assert loaded.session_id == s.session_id
    assert loaded.user_id == "alice"


def test_get_missing_returns_none(store: Any) -> None:
    assert store.get("does-not-exist") is None


# ── append_events ─────────────────────────────────────────────────────────────


def test_append_events_single(store: Any) -> None:
    s = store.create()
    evt = _make_event()
    store.append_events(s.session_id, [evt])
    loaded = store.get(s.session_id)
    assert loaded is not None
    assert len(loaded.events) == 1


def test_append_events_accumulates(store: Any) -> None:
    s = store.create()
    store.append_events(s.session_id, [_make_event("user", "first")])
    store.append_events(s.session_id, [_make_event("assistant", "second")])
    loaded = store.get(s.session_id)
    assert loaded is not None
    assert len(loaded.events) == 2


def test_append_events_empty_is_noop(store: Any) -> None:
    s = store.create()
    store.append_events(s.session_id, [])
    loaded = store.get(s.session_id)
    assert loaded is not None
    assert len(loaded.events) == 0


def test_append_events_updates_updated_at(store: Any) -> None:
    s = store.create()
    before = s.updated_at
    time.sleep(0.01)
    store.append_events(s.session_id, [_make_event()])
    loaded = store.get(s.session_id)
    assert loaded is not None
    assert loaded.updated_at >= before


# ── Event round-trip: all ContentItem discriminators ─────────────────────────


def test_round_trip_message(store: Any) -> None:
    s = store.create()
    evt = Event(
        execution_id="x",
        author="test",
        content=[Message(role="user", content="hello world")],
    )
    store.append_events(s.session_id, [evt])
    loaded = store.get(s.session_id)
    assert loaded is not None
    msg = loaded.events[0].content[0]
    assert isinstance(msg, Message)
    assert msg.content == "hello world"
    assert msg.role == "user"


def test_round_trip_tool_call(store: Any) -> None:
    s = store.create()
    evt = _make_tool_event(1)
    store.append_events(s.session_id, [evt])
    loaded = store.get(s.session_id)
    assert loaded is not None
    tc = loaded.events[0].content[0]
    assert isinstance(tc, ToolCall)
    assert tc.name == "search"
    assert tc.arguments == {"q": "q1"}


def test_round_trip_tool_result(store: Any) -> None:
    s = store.create()
    evt = _make_tool_result_event(2)
    store.append_events(s.session_id, [evt])
    loaded = store.get(s.session_id)
    assert loaded is not None
    tr = loaded.events[0].content[0]
    assert isinstance(tr, ToolResult)
    assert tr.status == "success"
    assert tr.content == ["result2"]


# ── update_state ──────────────────────────────────────────────────────────────


def test_update_state_merges(store: Any) -> None:
    s = store.create()
    store.update_state(s.session_id, {"a": 1})
    store.update_state(s.session_id, {"b": 2})
    loaded = store.get(s.session_id)
    assert loaded is not None
    assert loaded.state == {"a": 1, "b": 2}


def test_update_state_overwrites_key(store: Any) -> None:
    s = store.create()
    store.update_state(s.session_id, {"key": "old"})
    store.update_state(s.session_id, {"key": "new"})
    loaded = store.get(s.session_id)
    assert loaded is not None
    assert loaded.state["key"] == "new"


# ── list_sessions ─────────────────────────────────────────────────────────────


def test_list_sessions_filters_by_user(store: Any) -> None:
    store.create(user_id="alice")
    store.create(user_id="alice")
    store.create(user_id="bob")
    alice = store.list_sessions("alice")
    bob = store.list_sessions("bob")
    assert len(alice) == 2
    assert len(bob) == 1


def test_list_sessions_respects_limit(store: Any) -> None:
    for _ in range(5):
        store.create(user_id="carol")
    listed = store.list_sessions("carol", limit=3)
    assert len(listed) == 3


def test_list_sessions_empty_user(store: Any) -> None:
    assert store.list_sessions("nobody") == []


# ── delete ────────────────────────────────────────────────────────────────────


def test_delete_existing_session(store: Any) -> None:
    s = store.create()
    assert store.delete(s.session_id) is True
    assert store.get(s.session_id) is None


def test_delete_missing_session(store: Any) -> None:
    assert store.delete("nonexistent") is False


# ── delete_user ───────────────────────────────────────────────────────────────


def test_delete_user_removes_all_sessions(store: Any) -> None:
    store.create(user_id="dave")
    store.create(user_id="dave")
    store.create(user_id="eve")
    n = store.delete_user("dave")
    assert n == 2
    assert store.list_sessions("dave") == []
    assert len(store.list_sessions("eve")) == 1


def test_delete_user_unknown(store: Any) -> None:
    assert store.delete_user("nobody") == 0


# ── User isolation ────────────────────────────────────────────────────────────


def test_isolation_between_users(store: Any) -> None:
    sa = store.create(user_id="alice")
    sb = store.create(user_id="bob")

    store.append_events(sa.session_id, [_make_event("user", "alice message")])
    store.append_events(sb.session_id, [_make_event("user", "bob message")])

    loaded_a = store.get(sa.session_id)
    loaded_b = store.get(sb.session_id)
    assert loaded_a is not None and loaded_b is not None

    a_text = loaded_a.events[0].content[0].content  # type: ignore[union-attr]
    b_text = loaded_b.events[0].content[0].content  # type: ignore[union-attr]

    assert "alice" in a_text
    assert "alice" not in b_text
    assert "bob" in b_text
    assert "bob" not in a_text


# ── Agent integration ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_agent_persists_events(store: Any) -> None:
    """Agent.run with a session persists new events to the store."""
    from agentkit.agent import Agent
    from agentkit.llm import LlmClient, LlmRequest, LlmResponse

    session = store.create(user_id="test")

    # Mock LlmClient so no real API call
    mock_client = AsyncMock(spec=LlmClient)
    mock_client.generate.return_value = LlmResponse(
        content=[Message(role="assistant", content="42")],
        usage_metadata={},
    )

    agent = Agent(
        model=mock_client,
        tools=[],
        instructions="",
        max_steps=2,
        session_store=store,
    )
    result = await agent.run("What is the answer?", session=session)

    assert result.error is None
    assert "42" in str(result.output)

    # Events should be in the store now
    loaded = store.get(session.session_id)
    assert loaded is not None
    assert len(loaded.events) > 0


@pytest.mark.asyncio
async def test_agent_loads_past_events(store: Any) -> None:
    """Past session events are loaded into the context before the new turn."""
    from agentkit.agent import Agent
    from agentkit.llm import LlmClient, LlmResponse

    # Pre-populate session with a past event
    session = store.create(user_id="test")
    past_evt = _make_event("assistant", "The sky is blue.")
    store.append_events(session.session_id, [past_evt])

    # Reload so session.events is populated
    session = store.get(session.session_id) or session
    assert len(session.events) == 1

    captured_contents: list[Any] = []

    async def _capture(req: Any) -> LlmResponse:
        captured_contents.extend(req.contents)
        return LlmResponse(
            content=[Message(role="assistant", content="Yes, I remember.")],
            usage_metadata={},
        )

    mock_client = AsyncMock(spec=LlmClient)
    mock_client.generate.side_effect = _capture

    agent = Agent(
        model=mock_client,
        tools=[],
        instructions="",
        max_steps=2,
        session_store=store,
    )
    await agent.run("What did we say before?", session=session)

    # The past event content should appear in the request
    contents_text = " ".join(
        getattr(c, "content", "") for c in captured_contents if hasattr(c, "content")
    )
    assert "blue" in contents_text


@pytest.mark.asyncio
async def test_agent_session_state_shared(store: Any) -> None:
    """context.state['session_state'] references session.state — writes survive."""
    from agentkit.agent import Agent
    from agentkit.llm import LlmClient, LlmResponse
    from agentkit.tools.base import tool

    session = store.create(user_id="test")

    @tool
    def write_state(context: Any, value: str) -> str:
        """Write a value to session state."""
        context.state.get("session_state", context.state)["saved"] = value
        return "done"

    responses = [
        # First call: trigger tool
        LlmResponse(
            content=[ToolCall(tool_call_id="c1", name="write_state", arguments={"value": "hello"})],
            usage_metadata={},
        ),
        # Second call: final answer
        LlmResponse(
            content=[Message(role="assistant", content="Saved!")],
            usage_metadata={},
        ),
    ]

    mock_client = AsyncMock(spec=LlmClient)
    mock_client.generate.side_effect = responses

    agent = Agent(
        model=mock_client,
        tools=[write_state],
        instructions="",
        max_steps=3,
        session_store=store,
    )
    await agent.run("save hello", session=session)

    loaded = store.get(session.session_id)
    assert loaded is not None
    assert loaded.state.get("saved") == "hello"


# ── SqliteSessionStore specific ───────────────────────────────────────────────


def test_sqlite_creates_file(tmp_path: Path) -> None:
    db = str(tmp_path / "new.db")
    SqliteSessionStore(db_path=db)
    assert Path(db).exists()


def test_sqlite_cascade_delete_removes_events(tmp_path: Path) -> None:
    """Deleting a session via ON DELETE CASCADE removes its events."""
    import sqlite3

    db = str(tmp_path / "cascade.db")
    s_store = SqliteSessionStore(db_path=db)
    s = s_store.create(user_id="test")
    s_store.append_events(s.session_id, [_make_event()])

    # Confirm events exist
    con = sqlite3.connect(db)
    count_before = con.execute(
        "SELECT COUNT(*) FROM events WHERE session_id = ?", (s.session_id,)
    ).fetchone()[0]
    con.close()
    assert count_before == 1

    s_store.delete(s.session_id)

    con = sqlite3.connect(db)
    count_after = con.execute(
        "SELECT COUNT(*) FROM events WHERE session_id = ?", (s.session_id,)
    ).fetchone()[0]
    con.close()
    assert count_after == 0


def test_sqlite_schema_version(tmp_path: Path) -> None:
    db = str(tmp_path / "schema.db")
    SqliteSessionStore(db_path=db)
    import sqlite3

    con = sqlite3.connect(db)
    version = con.execute("SELECT version FROM schema_version").fetchone()[0]
    con.close()
    assert version == 1


# ── Live tests ─────────────────────────────────────────────────────────────────


@pytest.mark.live
@pytest.mark.asyncio
async def test_live_session_restart(tmp_path: Path) -> None:
    """Integration test: persist a fact, reload session, recall the fact."""
    from agentkit.agent import Agent
    from agentkit.llm import LlmClient

    db = str(tmp_path / "live.db")
    store1 = SqliteSessionStore(db_path=db)
    agent1 = Agent(
        model=LlmClient("anthropic/claude-haiku-4-5"),
        tools=[],
        instructions="You are a helpful assistant.",
        max_steps=3,
        session_store=store1,
    )
    session = store1.create(user_id="live_test")
    await agent1.run("My lucky number is 7. Remember it.", session=session)

    # Reload from disk
    store2 = SqliteSessionStore(db_path=db)
    agent2 = Agent(
        model=LlmClient("anthropic/claude-haiku-4-5"),
        tools=[],
        instructions="You are a helpful assistant.",
        max_steps=3,
        session_store=store2,
    )
    session2 = store2.get(session.session_id)
    assert session2 is not None
    result = await agent2.run("What is my lucky number?", session=session2)
    assert "7" in str(result.output)
