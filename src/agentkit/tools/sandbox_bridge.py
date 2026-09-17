"""Tool bridge: exposes host tools to Python code inside the sandbox (block 18).

Two backends — chosen per tool automatically
--------------------------------------------

env backend (cloud e2b, default for known tools)
    The sandbox has full internet access, so it can call external APIs
    directly.  The bridge injects the required API key as an env var and
    generates a stub that calls the API over HTTPS — no host process
    involved at execution time.

    Registered for: search_web (Tavily REST API)

    Security notes
    --------------
    • Use a separate Tavily key with a small quota (sandbox-only).
    • Never inject ANTHROPIC_API_KEY — model-generated code could then call
      the LLM and make the bill unpredictable.
    • Pass only the keys needed for the bridged tools, never the full env.

http backend (local / Docker sandboxes)
    Starts a background-thread HTTP server in the host process.  Stubs use
    urllib.request to POST kwargs and receive the result.  Works when the
    sandbox can reach 127.0.0.1 (local Docker template, same-host dev env).
    Not suitable for remote e2b cloud sandboxes.

Usage
-----
    # Cloud e2b — env backend picks up automatically for search_web
    bridge = SandboxBridge(
        tools=[search_web_tool],
        context=ctx,
        loop=loop,
        envs={"TAVILY_API_KEY": os.environ["TAVILY_API_KEY"]},
    )
    host, port = bridge.start()                 # no-op for env-only tools
    await sandbox.run_code(bridge.env_setup_code())   # inject key
    await sandbox.run_code(bridge.stub_code(host, port))  # inject functions

    # Local / Docker — http backend
    bridge = SandboxBridge(tools=[my_local_tool], context=ctx, loop=loop)
    host, port = bridge.start("127.0.0.1")
    await sandbox.run_code(bridge.stub_code(host, port))

    bridge.stop()   # called automatically in Agent._kill_sandbox
"""
from __future__ import annotations

import asyncio
import json
import socket
import threading
from concurrent.futures import Future
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agentkit.context import ExecutionContext
    from agentkit.tools.base import BaseTool


# ── Direct (env-backend) stub registry ────────────────────────────────────────
# Maps tool_name → callable(envs: dict[str, str]) → Python source string.
# Add entries here for any tool whose cloud API is directly reachable from
# the sandbox.  The generated function must match the host tool's return format.


def _tavily_search_stub(envs: dict[str, str]) -> str:
    """Python source for search_web that calls Tavily REST from inside sandbox."""
    return '''\
def search_web(query, max_results=5, topic="general", time_range=None):
    """Search the web using Tavily (direct HTTPS from sandbox)."""
    import json as _j, os as _o, urllib.request as _r
    key = _o.environ.get("TAVILY_API_KEY", "")
    body = {"api_key": key, "query": query,
            "max_results": max_results, "topic": topic}
    if time_range:
        body["time_range"] = time_range
    data = _j.dumps(body).encode()
    req = _r.Request(
        "https://api.tavily.com/search", data=data,
        headers={"Content-Type": "application/json"},
    )
    with _r.urlopen(req, timeout=30) as resp:
        payload = _j.loads(resp.read())
    results = payload.get("results", [])
    if not results:
        return "(no results)"
    parts = []
    for i, r in enumerate(results, 1):
        parts.append(f"[{i}] {r.get('title', '')}\\n"
                     f"{r.get('url', '')}\\n"
                     f"{r.get('content', '')}")
    return "\\n\\n".join(parts)
'''


_DIRECT_STUBS: dict[str, Any] = {
    "search_web": _tavily_search_stub,
}


# ── Helpers ────────────────────────────────────────────────────────────────────


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ── SandboxBridge ──────────────────────────────────────────────────────────────


class SandboxBridge:
    """Route each tool to the right backend (env or http) automatically.

    Parameters
    ----------
    tools:
        Host-side tools to expose inside the sandbox.
    context:
        ExecutionContext for the current run (passed to http-backend tools).
    loop:
        Main asyncio event loop (for run_coroutine_threadsafe in http mode).
    envs:
        Environment variables to inject into the sandbox.  Required for
        tools that use the env backend (e.g. ``{"TAVILY_API_KEY": "..."}``).
    """

    def __init__(
        self,
        tools: list[BaseTool],
        context: ExecutionContext,
        loop: asyncio.AbstractEventLoop,
        envs: dict[str, str] | None = None,
    ) -> None:
        self._all_tools: dict[str, BaseTool] = {t.name: t for t in tools}
        self._context = context
        self._loop = loop
        self._envs: dict[str, str] = dict(envs or {})

        # Partition tools into env-backend (direct API) vs http-backend
        self._env_tools: dict[str, BaseTool] = {}
        self._http_tools: dict[str, BaseTool] = {}
        for name, tool in self._all_tools.items():
            if name in _DIRECT_STUBS:
                self._env_tools[name] = tool
            else:
                self._http_tools[name] = tool

        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._host: str = "127.0.0.1"
        self._port: int = 0

    # ── lifecycle ──────────────────────────────────────────────────────────────

    def start(self, host: str = "127.0.0.1", port: int = 0) -> tuple[str, int]:
        """Start the HTTP server (only if http-backend tools exist).

        Returns (host, port).  For env-only setups port=0 (no server started).
        """
        if not self._http_tools:
            self._host = host
            return host, 0

        if port == 0:
            port = _free_port()

        bridge = self

        class _Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                parts = self.path.strip("/").split("/")
                if len(parts) != 2 or parts[0] != "tool":
                    self._respond(404, {"error": "Not found"})
                    return
                tool_name = parts[1]
                tool = bridge._http_tools.get(tool_name)
                if tool is None:
                    self._respond(404, {"error": f"Unknown tool: {tool_name!r}"})
                    return
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length)
                try:
                    kwargs: dict[str, Any] = json.loads(body) if body else {}
                except json.JSONDecodeError as exc:
                    self._respond(400, {"error": f"Bad JSON: {exc}"})
                    return
                fut: Future[Any] = asyncio.run_coroutine_threadsafe(
                    tool.execute(bridge._context, **kwargs),
                    bridge._loop,
                )
                try:
                    result = fut.result(timeout=60)
                    self._respond(200, {"result": result})
                except Exception as exc:  # noqa: BLE001
                    self._respond(500, {"error": str(exc)})

            def _respond(self, code: int, payload: dict[str, Any]) -> None:
                body = json.dumps(payload, ensure_ascii=False).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, fmt: str, *args: Any) -> None:
                pass

        self._server = HTTPServer((host, port), _Handler)
        self._host = host
        self._port = port
        self._thread = threading.Thread(
            target=self._server.serve_forever, daemon=True, name="SandboxBridge"
        )
        self._thread.start()
        return host, port

    def stop(self) -> None:
        """Shut down the HTTP server.  Safe to call if not started."""
        if self._server is not None:
            self._server.shutdown()
            self._server = None

    # ── code generation ────────────────────────────────────────────────────────

    def env_setup_code(self) -> str:
        """Return Python source that sets env vars in the sandbox process.

        Inject this via ``sandbox.run_code(bridge.env_setup_code())`` BEFORE
        ``stub_code()``, so the stubs can read the env vars on first call.
        """
        if not self._envs:
            return ""
        lines = ["import os as _os"]
        for k, v in self._envs.items():
            lines.append(f"_os.environ[{k!r}] = {v!r}")
        return "\n".join(lines)

    def stub_code(self, host: str = "", port: int = 0) -> str:
        """Return Python source that defines one callable per registered tool.

        Env-backend tools: stubs call cloud APIs directly (HTTPS, env var key).
        HTTP-backend tools: stubs POST to the bridge server at host:port.
        """
        parts: list[str] = []

        # env-backend stubs (direct cloud API calls)
        for name in self._env_tools:
            stub_fn = _DIRECT_STUBS[name]
            parts.append(stub_fn(self._envs))

        # http-backend stubs
        if self._http_tools:
            http_header = [
                "import json as _json, urllib.request as _req",
                f"_BRIDGE_HOST = {host!r}",
                f"_BRIDGE_PORT = {port!r}",
                "",
                "def _call_bridge(tool_name, **kwargs):",
                "    url = f'http://{_BRIDGE_HOST}:{_BRIDGE_PORT}/tool/{tool_name}'",
                "    data = _json.dumps(kwargs).encode()",
                "    req = _req.Request(url, data=data,",
                "                       headers={'Content-Type': 'application/json'})",
                "    with _req.urlopen(req, timeout=60) as resp:",
                "        payload = _json.loads(resp.read())",
                "    if 'error' in payload:",
                "        raise RuntimeError(",
                "            f'Bridge error ({tool_name}): {payload[\"error\"]}')",
                "    return payload['result']",
                "",
            ]
            parts.append("\n".join(http_header))
            for name, tool in self._http_tools.items():
                doc = (tool.description or "").replace('"', '\\"').replace("\n", "\\n")[:200]
                parts.append(
                    f"def {name}(**kwargs):\n"
                    f'    """{doc}"""\n'
                    f"    return _call_bridge({name!r}, **kwargs)\n"
                )

        return "\n".join(parts)

    def instructions_hint(self) -> str:
        """One-line hint listing bridged tools for inclusion in agent instructions."""
        names = ", ".join(f"{n}(…)" for n in self._all_tools)
        backend_note = (
            "via direct HTTPS" if not self._http_tools else
            "via bridge" if not self._env_tools else
            "via direct HTTPS or bridge"
        )
        return (
            f"Inside execute_python the following host tools are available as "
            f"Python functions ({backend_note}): {names}. "
            "Call them directly — they return results as strings."
        )
