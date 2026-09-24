"""Unit tests for A2A server, client, and tool (block 22).

All tests use an in-process aiohttp TestServer — no real TCP port, no API key.

Run with:
    uv run --group dev --group a2a pytest tests/test_a2a.py -v
"""
from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("aiohttp")
from aiohttp.test_utils import TestClient, TestServer

from agentkit.a2a import (
    A2AClient,
    A2AServer,
    A2ATask,
    A2ATool,
    TaskStatus,
    _parse_task_result,
)

# ── helpers ───────────────────────────────────────────────────────────────────


def _mock_agent(answer: str = "42", name: str = "agent", error: str | None = None) -> Any:
    from agentkit.agent import AgentResult
    from agentkit.context import ExecutionContext

    result = AgentResult(output=answer, context=ExecutionContext(), error=error)
    agent = MagicMock()
    agent.name = name
    agent.run = AsyncMock(return_value=result)
    return agent


async def _client_for(agent: Any) -> tuple[A2AServer, TestClient]:
    """Create an A2AServer and wire up an aiohttp TestClient."""
    server = A2AServer(agent, name=agent.name, description="test", port=0)
    app = server._make_app()
    return server, TestClient(TestServer(app))


# ── AgentCard endpoint ────────────────────────────────────────────────────────


async def test_agent_card_returns_metadata() -> None:
    agent = _mock_agent(name="researcher")
    _server, tc = await _client_for(agent)
    async with tc:
        resp = await tc.get("/.well-known/agent.json")
        assert resp.status == 200
        data = await resp.json()
    assert data["name"] == "researcher"
    assert "tasks/send" in data["capabilities"]
    assert "tasks/get" in data["capabilities"]
    assert "url" in data


# ── tasks/send ────────────────────────────────────────────────────────────────


async def test_send_returns_completed() -> None:
    agent = _mock_agent("The answer is 42.")
    _server, tc = await _client_for(agent)
    async with tc:
        resp = await tc.post("/", json={
            "jsonrpc": "2.0", "id": 1, "method": "tasks/send",
            "params": {
                "id": "t-001",
                "message": {"role": "user", "parts": [{"type": "text", "text": "What is 6×7?"}]},
            },
        })
        data = await resp.json()

    assert data["result"]["id"] == "t-001"
    assert data["result"]["status"]["state"] == "completed"
    text = data["result"]["artifacts"][0]["parts"][0]["text"]
    assert "42" in text
    agent.run.assert_awaited_once_with("What is 6×7?")


async def test_send_auto_generates_id() -> None:
    agent = _mock_agent()
    _server, tc = await _client_for(agent)
    async with tc:
        resp = await tc.post("/", json={
            "jsonrpc": "2.0", "id": 1, "method": "tasks/send",
            "params": {"message": {"role": "user", "parts": [{"type": "text", "text": "hi"}]}},
        })
        data = await resp.json()

    assert isinstance(data["result"]["id"], str)
    assert data["result"]["id"]  # non-empty


async def test_send_agent_error_returns_failed() -> None:
    agent = _mock_agent(answer="", error="search unavailable")
    _server, tc = await _client_for(agent)
    async with tc:
        resp = await tc.post("/", json={
            "jsonrpc": "2.0", "id": 1, "method": "tasks/send",
            "params": {
                "id": "t-fail",
                "message": {"role": "user", "parts": [{"type": "text", "text": "fail"}]},
            },
        })
        data = await resp.json()

    assert data["result"]["status"]["state"] == "failed"
    assert "artifacts" not in data["result"]
    assert data["result"]["error"] == "search unavailable"


async def test_send_agent_exception_returns_failed() -> None:

    agent = MagicMock()
    agent.name = "bad"
    agent.run = AsyncMock(side_effect=RuntimeError("boom"))

    _server, tc = await _client_for(agent)
    async with tc:
        resp = await tc.post("/", json={
            "jsonrpc": "2.0", "id": 1, "method": "tasks/send",
            "params": {"id": "t-exc", "message": {"role": "user", "parts": [{"type": "text", "text": "x"}]}},
        })
        data = await resp.json()

    assert data["result"]["status"]["state"] == "failed"
    assert "boom" in data["result"]["error"]


# ── tasks/get ────────────────────────────────────────────────────────────────


async def test_get_after_send() -> None:
    agent = _mock_agent("pong")
    _server, tc = await _client_for(agent)
    async with tc:
        await tc.post("/", json={
            "jsonrpc": "2.0", "id": 1, "method": "tasks/send",
            "params": {"id": "t-get", "message": {"role": "user", "parts": [{"type": "text", "text": "ping"}]}},
        })
        resp = await tc.post("/", json={
            "jsonrpc": "2.0", "id": 2, "method": "tasks/get",
            "params": {"id": "t-get"},
        })
        data = await resp.json()

    assert data["result"]["id"] == "t-get"
    assert data["result"]["status"]["state"] == "completed"


async def test_get_unknown_id_returns_rpc_error() -> None:
    agent = _mock_agent()
    _server, tc = await _client_for(agent)
    async with tc:
        resp = await tc.post("/", json={
            "jsonrpc": "2.0", "id": 3, "method": "tasks/get",
            "params": {"id": "does-not-exist"},
        })
        data = await resp.json()

    assert "error" in data
    assert data["error"]["code"] == -32602


# ── unknown method ────────────────────────────────────────────────────────────


async def test_unknown_method_returns_rpc_error() -> None:
    agent = _mock_agent()
    _server, tc = await _client_for(agent)
    async with tc:
        resp = await tc.post("/", json={
            "jsonrpc": "2.0", "id": 4, "method": "tasks/cancel", "params": {},
        })
        data = await resp.json()

    assert "error" in data
    assert data["error"]["code"] == -32601


# ── parse helpers ─────────────────────────────────────────────────────────────


def test_parse_task_result_completed() -> None:
    task = _parse_task_result({
        "id": "t1",
        "status": {"state": "completed"},
        "artifacts": [{"parts": [{"type": "text", "text": "hello"}]}],
    })
    assert task.id == "t1"
    assert task.status == TaskStatus.completed
    assert task.output_text == "hello"


def test_parse_task_result_failed() -> None:
    task = _parse_task_result({"id": "t2", "status": {"state": "failed"}, "error": "oops"})
    assert task.status == TaskStatus.failed
    assert task.error == "oops"
    assert task.output_text is None


def test_parse_task_result_unknown_state_defaults_to_completed() -> None:
    task = _parse_task_result({"id": "t3", "status": {"state": "bogus"}})
    assert task.status == TaskStatus.completed


# ── A2AClient against TestServer ──────────────────────────────────────────────


async def test_client_get_card() -> None:
    agent = _mock_agent(name="writer")
    _server, tc = await _client_for(agent)
    async with tc:
        url = str(tc.make_url(""))
        client = A2AClient(url)
        card = await client.get_card()
    assert card.name == "writer"
    assert isinstance(card.capabilities, list)
    assert "tasks/send" in card.capabilities


async def test_client_send_task() -> None:
    agent = _mock_agent("The answer is 42.")
    _server, tc = await _client_for(agent)
    async with tc:
        url = str(tc.make_url(""))
        task = await A2AClient(url).send_task("What is 6×7?")
    assert task.status == TaskStatus.completed
    assert task.output_text is not None
    assert "42" in task.output_text


async def test_client_send_task_explicit_id() -> None:
    agent = _mock_agent("pong")
    _server, tc = await _client_for(agent)
    async with tc:
        url = str(tc.make_url(""))
        task = await A2AClient(url).send_task("ping", task_id="my-id")
    assert task.id == "my-id"


async def test_client_get_task() -> None:
    agent = _mock_agent("done")
    _server, tc = await _client_for(agent)
    async with tc:
        url = str(tc.make_url(""))
        client = A2AClient(url)
        await client.send_task("go", task_id="fetch-me")
        fetched = await client.get_task("fetch-me")
    assert fetched.id == "fetch-me"
    assert fetched.status == TaskStatus.completed


# ── A2ATool ───────────────────────────────────────────────────────────────────


async def test_a2a_tool_execute_returns_text() -> None:
    from agentkit.context import ExecutionContext

    agent = _mock_agent("The answer is 42.")
    _server, tc = await _client_for(agent)
    async with tc:
        url = str(tc.make_url(""))
        tool = A2ATool(A2AClient(url), name="research", description="Research facts")
        result = await tool.execute(ExecutionContext(), task="What is 6×7?")
    assert "42" in result


def test_a2a_tool_schema() -> None:
    tool = A2ATool(A2AClient("http://localhost:9999"), name="researcher", description="Desc")
    defn = tool.tool_definition
    assert defn["function"]["name"] == "researcher"
    params = defn["function"]["parameters"]
    assert "task" in params["properties"]
    assert "task" in params["required"]


async def test_a2a_tool_propagates_remote_error() -> None:
    from agentkit.context import ExecutionContext

    client = A2AClient("http://localhost:9999")
    client.send_task = AsyncMock(  # type: ignore[method-assign]
        return_value=A2ATask(id="x", status=TaskStatus.failed, error="timeout")
    )
    tool = A2ATool(client, name="r", description="d")
    result = await tool.execute(ExecutionContext(), task="anything")
    assert "Error" in result
    assert "timeout" in result


# ── Auth ──────────────────────────────────────────────────────────────────────


async def test_auth_missing_token_returns_401() -> None:
    agent = _mock_agent()
    server = A2AServer(agent, port=0, auth_token="secret")
    app = server._make_app()
    async with TestClient(TestServer(app)) as tc:
        resp = await tc.post("/", json={
            "jsonrpc": "2.0", "id": 1, "method": "tasks/send",
            "params": {"message": {"role": "user", "parts": [{"type": "text", "text": "hi"}]}},
        })
        assert resp.status == 401
        data = await resp.json()
        assert data["error"]["code"] == -32600


async def test_auth_wrong_token_returns_401() -> None:
    agent = _mock_agent()
    server = A2AServer(agent, port=0, auth_token="secret")
    app = server._make_app()
    async with TestClient(TestServer(app)) as tc:
        resp = await tc.post("/", json={
            "jsonrpc": "2.0", "id": 1, "method": "tasks/send",
            "params": {"message": {"role": "user", "parts": [{"type": "text", "text": "hi"}]}},
        }, headers={"Authorization": "Bearer wrong"})
        assert resp.status == 401


async def test_auth_correct_token_passes() -> None:
    agent = _mock_agent("pong")
    server = A2AServer(agent, port=0, auth_token="secret")
    app = server._make_app()
    async with TestClient(TestServer(app)) as tc:
        resp = await tc.post("/", json={
            "jsonrpc": "2.0", "id": 1, "method": "tasks/send",
            "params": {"id": "auth-ok", "message": {"role": "user", "parts": [{"type": "text", "text": "ping"}]}},
        }, headers={"Authorization": "Bearer secret"})
        assert resp.status == 200
        data = await resp.json()
        assert data["result"]["status"]["state"] == "completed"


async def test_auth_card_always_public() -> None:
    """GET /.well-known/agent.json must be accessible without a token."""
    agent = _mock_agent(name="locked")
    server = A2AServer(agent, port=0, auth_token="secret")
    app = server._make_app()
    async with TestClient(TestServer(app)) as tc:
        resp = await tc.get("/.well-known/agent.json")
        assert resp.status == 200
        data = await resp.json()
        assert data["name"] == "locked"


# ── Rate limiter ──────────────────────────────────────────────────────────────


def test_rate_limiter_allows_under_limit() -> None:
    from agentkit.a2a import _RateLimiter
    rl = _RateLimiter(max_per_minute=5)
    for _ in range(5):
        assert rl.allow("1.2.3.4")


def test_rate_limiter_blocks_over_limit() -> None:
    from agentkit.a2a import _RateLimiter
    rl = _RateLimiter(max_per_minute=3)
    for _ in range(3):
        rl.allow("1.2.3.4")
    assert not rl.allow("1.2.3.4")


def test_rate_limiter_per_ip() -> None:
    from agentkit.a2a import _RateLimiter
    rl = _RateLimiter(max_per_minute=2)
    assert rl.allow("1.1.1.1")
    assert rl.allow("1.1.1.1")
    assert not rl.allow("1.1.1.1")
    # different IP is not affected
    assert rl.allow("2.2.2.2")


async def test_rate_limit_via_server() -> None:
    agent = _mock_agent()
    server = A2AServer(agent, port=0, rate_limit=2)
    app = server._make_app()
    async with TestClient(TestServer(app)) as tc:
        payload = {
            "jsonrpc": "2.0", "id": 1, "method": "tasks/send",
            "params": {"message": {"role": "user", "parts": [{"type": "text", "text": "q"}]}},
        }
        r1 = await tc.post("/", json=payload)
        r2 = await tc.post("/", json=payload)
        r3 = await tc.post("/", json=payload)
        assert r1.status == 200
        assert r2.status == 200
        d3 = await r3.json()
        # third request is rate-limited
        assert "error" in d3
        assert d3["error"]["code"] == -32600


# ── Budget guard ──────────────────────────────────────────────────────────────


async def test_budget_guard_exceeded_returns_failed() -> None:
    from unittest.mock import MagicMock

    from agentkit.budget import BudgetExceededError, BudgetGuard

    guard = MagicMock(spec=BudgetGuard)
    guard.check.side_effect = BudgetExceededError("token budget exceeded")

    agent = _mock_agent("should not reach")
    server = A2AServer(agent, port=0, budget_guard=guard)
    app = server._make_app()
    async with TestClient(TestServer(app)) as tc:
        resp = await tc.post("/", json={
            "jsonrpc": "2.0", "id": 1, "method": "tasks/send",
            "params": {"id": "b1", "message": {"role": "user", "parts": [{"type": "text", "text": "hi"}]}},
        })
        data = await resp.json()

    assert data["result"]["status"]["state"] == "failed"
    assert "Budget exceeded" in data["result"]["error"]
    # agent.run should NOT have been called
    agent.run.assert_not_awaited()
