"""HTTP bridge: exposes host tools to Python code running inside the sandbox (block 18).

Architecture
------------
1.  SandboxBridge starts a lightweight HTTP server on the HOST process.
2.  Each registered tool is reachable via POST /tool/<name> with a JSON body
    of kwargs.  The handler dispatches to ``tool.execute()`` on the host and
    returns JSON ``{"result": "..."}`` or ``{"error": "..."}``.
3.  ``stub_code(host, port)`` returns Python source that can be injected into
    the sandbox (via ``sandbox.run_code(stubs)``) once at startup.  Every
    registered tool becomes a callable Python function inside the sandbox.
4.  Async tools are called from the background HTTP-server thread via
    ``asyncio.run_coroutine_threadsafe(coro, main_loop)``.

Remote sandboxes (e2b cloud)
----------------------------
e2b sandboxes run in the cloud and can make outbound HTTP requests to the
internet.  For local development on a machine accessible from the sandbox
(e.g. a VPS, cloud VM, or with an inbound tunnel such as ngrok), set
``host`` to the publicly reachable address of the host machine.  For unit
tests and local experimentation a loopback address (127.0.0.1) is sufficient.

Usage
-----
    loop = asyncio.get_running_loop()
    bridge = SandboxBridge(tools=[search_web_tool], context=ctx, loop=loop)
    host, port = bridge.start()                     # background thread
    await sandbox.run_code(bridge.stub_code(host, port))  # inject once
    # From now on, sandbox Python code can call: result = search_web(query="…")
    bridge.stop()                                   # called in Agent._kill_sandbox
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


def _free_port() -> int:
    """Return an available TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class SandboxBridge:
    """HTTP bridge that forwards sandbox→host tool calls via a background thread.

    Parameters
    ----------
    tools:
        Host-side tools that sandbox Python code should be able to call.
    context:
        The ExecutionContext for the current agent run (passed to tool.execute).
    loop:
        The main asyncio event loop (needed to dispatch async tools from the
        synchronous HTTP server thread).
    """

    def __init__(
        self,
        tools: list[BaseTool],
        context: ExecutionContext,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self._tools: dict[str, BaseTool] = {t.name: t for t in tools}
        self._context = context
        self._loop = loop
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._host: str = "127.0.0.1"
        self._port: int = 0

    # ── lifecycle ──────────────────────────────────────────────────────────────

    def start(self, host: str = "127.0.0.1", port: int = 0) -> tuple[str, int]:
        """Start the bridge HTTP server in a daemon thread.

        Returns
        -------
        (host, port) — pass both to ``stub_code()``.
        """
        if port == 0:
            port = _free_port()

        bridge = self

        class _Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                # Route: /tool/<tool_name>
                parts = self.path.strip("/").split("/")
                if len(parts) != 2 or parts[0] != "tool":
                    self._respond(404, {"error": "Not found"})
                    return

                tool_name = parts[1]
                tool = bridge._tools.get(tool_name)
                if tool is None:
                    self._respond(404, {"error": f"Unknown tool: {tool_name!r}"})
                    return

                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length)
                try:
                    kwargs: dict[str, Any] = json.loads(body) if body else {}
                except json.JSONDecodeError as exc:
                    self._respond(400, {"error": f"Bad JSON body: {exc}"})
                    return

                # Dispatch the async tool from this sync thread.
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
                pass  # suppress access logs

        self._server = HTTPServer((host, port), _Handler)
        self._host = host
        self._port = port
        self._thread = threading.Thread(
            target=self._server.serve_forever, daemon=True, name="SandboxBridge"
        )
        self._thread.start()
        return host, port

    def stop(self) -> None:
        """Shut down the HTTP server.  Safe to call even if not started."""
        if self._server is not None:
            self._server.shutdown()
            self._server = None

    # ── stub generation ────────────────────────────────────────────────────────

    def stub_code(self, host: str, port: int) -> str:
        """Return Python source code to inject into the sandbox.

        The generated code defines one Python function per registered tool.
        Each function serialises its keyword arguments as JSON, POSTs them to
        the bridge, and returns the result string.

        Parameters
        ----------
        host:
            Hostname or IP reachable FROM the sandbox (e.g. an ngrok public
            host, or "127.0.0.1" for local/test environments).
        port:
            Port the bridge is listening on (returned by ``start()``).
        """
        lines: list[str] = [
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
            "        raise RuntimeError(f'Bridge error ({tool_name}): {payload[\"error\"]}')",
            "    return payload['result']",
            "",
        ]
        for name, tool in self._tools.items():
            doc = (tool.description or "").replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")[:200]
            lines += [
                f"def {name}(**kwargs):",
                f'    """{doc}"""',
                f"    return _call_bridge({name!r}, **kwargs)",
                "",
            ]
        return "\n".join(lines)

    def instructions_hint(self) -> str:
        """One-line hint to add to agent instructions listing bridged tools.

        Example: "Inside execute_python you can call: search_web(query=…)"
        """
        names = ", ".join(f"{n}(…)" for n in self._tools)
        return (
            f"Inside execute_python the following host tools are available as "
            f"Python functions: {names}. "
            "Call them directly — they contact the host and return results."
        )
