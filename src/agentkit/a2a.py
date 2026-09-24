"""A2A (Agent-to-Agent) protocol — server, client, and tool (block 22).

The A2A protocol is a Google-led open standard for agent interoperability.
Agents expose an HTTP server; other agents discover and call them over HTTP.

Key concepts
------------
AgentCard   — JSON metadata at GET /.well-known/agent.json
              (name, description, url, capabilities)
A2ATask     — unit of work: submitted → working → completed/failed/canceled
A2AServer   — wraps an Agent as an A2A-compliant aiohttp HTTP server
A2AClient   — sends tasks to a remote A2A server, returns A2ATask
A2ATool     — BaseTool wrapping A2AClient so orchestrators treat remote
              agents identically to local in-process tools

JSON-RPC 2.0 methods
--------------------
tasks/send  — submit a task and block until the agent finishes
tasks/get   — retrieve a previously submitted task by id

Wire format (tasks/send request)
---------------------------------
    POST /
    {
      "jsonrpc": "2.0", "id": 1, "method": "tasks/send",
      "params": {
        "id": "<task-uuid>",
        "message": {
          "role": "user",
          "parts": [{"type": "text", "text": "What is 2+2?"}]
        }
      }
    }

Wire format (tasks/send response)
----------------------------------
    {
      "jsonrpc": "2.0", "id": 1,
      "result": {
        "id": "<task-uuid>",
        "status": {"state": "completed"},
        "artifacts": [{"parts": [{"type": "text", "text": "4"}]}]
      }
    }
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from collections import defaultdict
from enum import Enum
from typing import TYPE_CHECKING, Any, Self

from pydantic import BaseModel

from agentkit.schema import build_tool_definition
from agentkit.tools.base import BaseTool

if TYPE_CHECKING:
    from agentkit.agent import Agent
    from agentkit.context import ExecutionContext

logger = logging.getLogger(__name__)


# ── Rate limiter ──────────────────────────────────────────────────────────────


class _RateLimiter:
    """Sliding-window rate limiter: up to *max_per_minute* calls per IP."""

    def __init__(self, max_per_minute: int) -> None:
        self._max = max_per_minute
        self._timestamps: dict[str, list[float]] = defaultdict(list)

    def allow(self, ip: str) -> bool:
        now = time.monotonic()
        window_start = now - 60.0
        ts = self._timestamps[ip]
        # drop expired entries
        while ts and ts[0] < window_start:
            ts.pop(0)
        if len(ts) >= self._max:
            return False
        ts.append(now)
        return True


# ── Data types ────────────────────────────────────────────────────────────────


class AgentCard(BaseModel):
    """A2A agent metadata, served at GET /.well-known/agent.json."""

    name: str
    description: str
    url: str
    version: str = "0.1"
    capabilities: list[str] = []


class TaskStatus(str, Enum):
    submitted = "submitted"
    working = "working"
    completed = "completed"
    failed = "failed"
    canceled = "canceled"


class A2ATask(BaseModel):
    """A submitted task and its outcome."""

    id: str
    status: TaskStatus = TaskStatus.submitted
    input_text: str = ""
    output_text: str | None = None
    error: str | None = None

    def to_rpc_result(self) -> dict[str, Any]:
        """Serialise to A2A JSON-RPC ``result`` dict."""
        result: dict[str, Any] = {
            "id": self.id,
            "status": {"state": self.status.value},
        }
        if self.output_text is not None:
            result["artifacts"] = [
                {"parts": [{"type": "text", "text": self.output_text}]}
            ]
        if self.error is not None:
            result["error"] = self.error
        return result


def _parse_task_result(result: dict[str, Any]) -> A2ATask:
    """Convert a JSON-RPC ``result`` dict back to an A2ATask."""
    task_id = result.get("id", "")
    state_str = result.get("status", {}).get("state", "completed")
    try:
        status = TaskStatus(state_str)
    except ValueError:
        status = TaskStatus.completed
    artifacts = result.get("artifacts", [])
    text: str | None = None
    if artifacts:
        parts = artifacts[0].get("parts", [])
        texts = [p.get("text", "") for p in parts if p.get("type") == "text"]
        text = "\n".join(texts) or None
    error = result.get("error")
    return A2ATask(id=task_id, status=status, output_text=text, error=error)


# ── Server ────────────────────────────────────────────────────────────────────


class A2AServer:
    """Expose an Agent as an A2A-compliant HTTP server.

    Implements GET ``/.well-known/agent.json`` (AgentCard) and
    POST ``/`` (JSON-RPC 2.0 dispatcher for ``tasks/send`` and ``tasks/get``).

    The server runs inside the current asyncio event loop using aiohttp.

    Parameters
    ----------
    agent:
        The Agent whose ``run()`` is called for each incoming task.
    name:
        Human-readable name for the AgentCard.  Defaults to ``agent.name``.
    description:
        Short description for the AgentCard.
    host:
        Bind address.  Defaults to ``"127.0.0.1"``.
    port:
        TCP port.  Use ``0`` for an OS-assigned ephemeral port (useful in
        tests); call ``actual_port`` after ``start()`` to find out which.
    auth_token:
        If set, every POST ``/`` request must carry the header
        ``Authorization: Bearer <auth_token>``.
        GET ``/.well-known/agent.json`` is always public.
    rate_limit:
        Maximum POST requests per IP per minute.  ``None`` = unlimited.
    budget_guard:
        Optional ``BudgetGuard`` from ``agentkit.budget``.  Its ``check()``
        is called before each task; ``BudgetExceededError`` is returned as a
        ``failed`` task rather than crashing the server.
    """

    def __init__(
        self,
        agent: Agent,
        name: str | None = None,
        description: str | None = None,
        host: str = "127.0.0.1",
        port: int = 8090,
        auth_token: str | None = None,
        rate_limit: int | None = None,
        budget_guard: Any | None = None,
    ) -> None:
        self._agent = agent
        self._host = host
        self._port = port
        self._name = name or agent.name
        self._description = description or ""
        self._auth_token = auth_token
        self._rate_limiter = _RateLimiter(rate_limit) if rate_limit is not None else None
        self._budget_guard = budget_guard
        self._tasks: dict[str, A2ATask] = {}
        self._runner: Any = None
        self._site: Any = None

    @property
    def url(self) -> str:
        """Base URL of this server (uses configured port, not actual_port)."""
        return f"http://{self._host}:{self._port}"

    @property
    def actual_port(self) -> int:
        """OS-assigned port after ``start()``.  Same as *port* if port != 0."""
        if self._site is not None:
            name = self._site.name  # "http://host:port/"
            return int(name.rstrip("/").rsplit(":", 1)[-1])
        return self._port

    def agent_card(self) -> AgentCard:
        return AgentCard(
            name=self._name,
            description=self._description,
            url=f"http://{self._host}:{self.actual_port}",
            capabilities=["tasks/send", "tasks/get"],
        )

    # ── app factory (separated so tests can use TestServer) ───────────────────

    def _make_app(self) -> Any:
        try:
            from aiohttp import web
        except ImportError as exc:
            raise ImportError(
                "aiohttp is required for A2AServer. "
                "Install it with:  uv add --group a2a aiohttp"
            ) from exc

        middlewares: list[Any] = []

        if self._auth_token is not None:
            _token = self._auth_token  # capture for closure

            @web.middleware
            async def _auth_middleware(request: Any, handler: Any) -> Any:
                if request.path == "/.well-known/agent.json":
                    return await handler(request)
                auth = request.headers.get("Authorization", "")
                if not auth.startswith("Bearer ") or auth[7:] != _token:
                    return web.Response(
                        status=401,
                        content_type="application/json",
                        text=json.dumps({
                            "jsonrpc": "2.0",
                            "id": None,
                            "error": {"code": -32600, "message": "Unauthorized"},
                        }),
                    )
                return await handler(request)

            middlewares.append(_auth_middleware)

        app = web.Application(middlewares=middlewares)
        app.router.add_get("/.well-known/agent.json", self._handle_card)
        app.router.add_post("/", self._handle_rpc)
        return app

    # ── request handlers ──────────────────────────────────────────────────────

    async def _handle_card(self, request: Any) -> Any:
        from aiohttp import web

        return web.Response(
            content_type="application/json",
            text=self.agent_card().model_dump_json(),
        )

    async def _handle_rpc(self, request: Any) -> Any:
        if self._rate_limiter is not None:
            ip = request.remote or "unknown"
            if not self._rate_limiter.allow(ip):
                return self._json_error(None, -32600, "Rate limit exceeded")

        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            return self._json_error(None, -32700, "Parse error")

        rpc_id = body.get("id")
        method = body.get("method", "")
        params = body.get("params", {})

        if method == "tasks/send":
            return await self._rpc_send(rpc_id, params)
        if method == "tasks/get":
            return self._rpc_get(rpc_id, params)
        return self._json_error(rpc_id, -32601, f"Method not found: {method!r}")

    async def _rpc_send(self, rpc_id: Any, params: dict[str, Any]) -> Any:
        task_id = params.get("id") or str(uuid.uuid4())
        message = params.get("message", {})
        parts = message.get("parts", [])
        input_text = " ".join(
            p.get("text", "") for p in parts if p.get("type") == "text"
        ).strip()

        task = A2ATask(id=task_id, status=TaskStatus.working, input_text=input_text)
        self._tasks[task_id] = task

        if self._budget_guard is not None:
            try:
                self._budget_guard.check()
            except Exception as exc:  # noqa: BLE001  (BudgetExceededError)
                task.status = TaskStatus.failed
                task.error = f"Budget exceeded: {exc}"
                return self._json_ok(rpc_id, task.to_rpc_result())

        try:
            result = await self._agent.run(input_text)
            if result.error:
                task.status = TaskStatus.failed
                task.error = result.error
            else:
                task.status = TaskStatus.completed
                task.output_text = str(result.output)
        except Exception as exc:  # noqa: BLE001
            logger.warning("A2AServer task %s raised: %s", task_id, exc)
            task.status = TaskStatus.failed
            task.error = str(exc)

        return self._json_ok(rpc_id, task.to_rpc_result())

    def _rpc_get(self, rpc_id: Any, params: dict[str, Any]) -> Any:
        task_id = params.get("id", "")
        task = self._tasks.get(task_id)
        if task is None:
            return self._json_error(rpc_id, -32602, f"Unknown task id: {task_id!r}")
        return self._json_ok(rpc_id, task.to_rpc_result())

    # ── response helpers ──────────────────────────────────────────────────────

    def _json_ok(self, rpc_id: Any, result: Any) -> Any:
        from aiohttp import web

        return web.Response(
            content_type="application/json",
            text=json.dumps({"jsonrpc": "2.0", "id": rpc_id, "result": result}),
        )

    def _json_error(self, rpc_id: Any, code: int, message: str) -> Any:
        from aiohttp import web

        return web.Response(
            content_type="application/json",
            text=json.dumps({
                "jsonrpc": "2.0",
                "id": rpc_id,
                "error": {"code": code, "message": message},
            }),
        )

    # ── lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the aiohttp server in the background."""
        from aiohttp import web

        self._runner = web.AppRunner(self._make_app())
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, self._host, self._port)
        await self._site.start()
        logger.info("A2AServer '%s' listening at %s", self._name, self._site.name)

    async def stop(self) -> None:
        """Shut down the server."""
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None
            self._site = None

    async def __aenter__(self) -> Self:
        await self.start()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.stop()


# ── Client ────────────────────────────────────────────────────────────────────


class A2AClient:
    """Send tasks to a remote A2A server over HTTP.

    Parameters
    ----------
    url:
        Base URL of the remote server (e.g. ``"http://localhost:8090"``).
    """

    def __init__(self, url: str) -> None:
        self._url = url.rstrip("/")

    async def get_card(self) -> AgentCard:
        """Fetch the server's AgentCard from ``/.well-known/agent.json``."""
        import aiohttp

        async with aiohttp.ClientSession() as session, session.get(
            f"{self._url}/.well-known/agent.json"
        ) as resp:
            data = await resp.json(content_type=None)
        return AgentCard(**data)

    async def send_task(
        self,
        message: str,
        task_id: str | None = None,
    ) -> A2ATask:
        """Submit *message* as a task and wait for the result.

        Returns an A2ATask with status ``completed`` or ``failed``.
        """
        import aiohttp

        tid = task_id or str(uuid.uuid4())
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tasks/send",
            "params": {
                "id": tid,
                "message": {
                    "role": "user",
                    "parts": [{"type": "text", "text": message}],
                },
            },
        }
        async with aiohttp.ClientSession() as session, session.post(
            self._url, json=payload
        ) as resp:
            data = await resp.json(content_type=None)

        if "error" in data:
            return A2ATask(
                id=tid,
                status=TaskStatus.failed,
                error=data["error"].get("message", "unknown error"),
            )
        return _parse_task_result(data.get("result", {}))

    async def get_task(self, task_id: str) -> A2ATask:
        """Retrieve a previously submitted task by id."""
        import aiohttp

        payload = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tasks/get",
            "params": {"id": task_id},
        }
        async with aiohttp.ClientSession() as session, session.post(
            self._url, json=payload
        ) as resp:
            data = await resp.json(content_type=None)

        if "error" in data:
            return A2ATask(
                id=task_id,
                status=TaskStatus.failed,
                error=data["error"].get("message", "unknown error"),
            )
        return _parse_task_result(data.get("result", {}))


# ── Tool ─────────────────────────────────────────────────────────────────────


class A2ATool(BaseTool):
    """Call a remote A2A agent as a local tool.

    The LLM sees one parameter — ``task`` (string) — and calls this tool
    exactly like any other.  The request is forwarded over HTTP; the result
    is returned as a plain string.

    Parameters
    ----------
    client:
        An A2AClient pointed at the remote agent's URL.
    name:
        Tool name exposed to the LLM (e.g. ``"research"``).
    description:
        Tool description shown to the LLM.
    """

    def __init__(self, client: A2AClient, name: str, description: str) -> None:
        schema = build_tool_definition(
            name,
            description,
            {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "The task or question to send to the remote agent.",
                    }
                },
                "required": ["task"],
            },
        )
        super().__init__(name=name, description=description, tool_definition=schema)
        self._client = client

    async def execute(self, context: ExecutionContext, **kwargs: Any) -> str:
        task_text = kwargs.get("task", "")
        result = await self._client.send_task(task_text)
        if result.status == TaskStatus.failed:
            return f"Error from remote agent: {result.error or 'unknown failure'}"
        return result.output_text or ""
