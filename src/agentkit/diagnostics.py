"""System diagnostics — `agent doctor` command (block 9).

Checks platform, memory, disk, node/npx, API keys, Anthropic API
connectivity, and MCP server cold-start time.

Exit code 0 = all critical checks passed.
Exit code 1 = at least one critical check failed.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from typing import Any

from agentkit.config import env_status

# ── Individual checks ─────────────────────────────────────────────────────────


def _check_platform() -> dict[str, Any]:
    return {
        "platform": sys.platform,
        "python": sys.version.split()[0],
        "arch": _uname_machine(),
    }


def _uname_machine() -> str:
    try:
        import platform
        return platform.machine()
    except Exception:  # noqa: BLE001
        return "unknown"


def _check_memory() -> dict[str, Any]:
    try:
        import psutil  # optional dep

        mem = psutil.virtual_memory()
        return {
            "ram_total_gb": round(mem.total / 1e9, 2),
            "ram_free_gb": round(mem.available / 1e9, 2),
        }
    except ImportError:
        return {"ram_total_gb": "N/A", "ram_free_gb": "N/A"}


def _check_disk() -> dict[str, Any]:
    usage = shutil.disk_usage(".")
    return {
        "disk_total_gb": round(usage.total / 1e9, 2),
        "disk_free_gb": round(usage.free / 1e9, 2),
    }


def _run_version(cmd: list[str]) -> str | None:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=5)
        return out.decode().strip().splitlines()[0]
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def _check_node() -> dict[str, Any]:
    node = _run_version(["node", "--version"])
    npx = _run_version(["npx", "--version"])
    return {
        "node": node or "NOT FOUND",
        "npx": npx or "NOT FOUND",
        "node_ok": node is not None,
        "npx_ok": npx is not None,
    }


def _check_env_keys() -> dict[str, Any]:
    return env_status()


def _check_anthropic_api() -> dict[str, Any]:
    import os

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return {"anthropic_ok": False, "anthropic_error": "ANTHROPIC_API_KEY not set"}
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        # A lightweight call: list models
        client.models.list()
        return {"anthropic_ok": True}
    except Exception as exc:  # noqa: BLE001
        return {"anthropic_ok": False, "anthropic_error": str(exc)[:120]}


async def _check_mcp_cold_start() -> dict[str, Any]:
    from agentkit.config import find_uv
    from agentkit.mcp_client import McpToolset

    cmd = find_uv()
    args = ["run", "python", "-m", "agentkit.servers.tavily_server"]
    t0 = time.perf_counter()
    try:
        async with McpToolset(cmd, args):
            elapsed = time.perf_counter() - t0
            return {"mcp_ok": True, "mcp_cold_start_s": round(elapsed, 2)}
    except Exception as exc:  # noqa: BLE001
        elapsed = time.perf_counter() - t0
        return {
            "mcp_ok": False,
            "mcp_cold_start_s": round(elapsed, 2),
            "mcp_error": str(exc)[:120],
        }


# ── Main entry point ──────────────────────────────────────────────────────────


async def run_doctor(json_output: bool = False) -> int:
    """Run all checks, print results, return exit code (0 = OK, 1 = critical failure)."""
    result: dict[str, Any] = {}

    result["system"] = {**_check_platform(), **_check_memory(), **_check_disk()}
    result["node"] = _check_node()
    result["env_keys"] = _check_env_keys()

    print("Checking Anthropic API...", flush=True)
    result["anthropic"] = _check_anthropic_api()

    print("Measuring MCP cold start...", flush=True)
    result["mcp"] = await _check_mcp_cold_start()

    if json_output:
        print(json.dumps(result, indent=2))
    else:
        _print_human(result)

    critical_ok = (
        result["anthropic"].get("anthropic_ok", False)
        and result["env_keys"].get("ANTHROPIC_API_KEY", False)
    )
    return 0 if critical_ok else 1


def _print_human(r: dict[str, Any]) -> None:
    s = r["system"]
    n = r["node"]
    keys = r["env_keys"]
    ant = r["anthropic"]
    mcp = r["mcp"]

    print("\n=== Agent Doctor ===\n")
    print(f"  Platform  : {s['platform']} {s['arch']}  Python {s['python']}")
    print(f"  RAM       : {s['ram_free_gb']} GB free / {s['ram_total_gb']} GB total")
    print(f"  Disk      : {s['disk_free_gb']} GB free / {s['disk_total_gb']} GB total")
    print()

    node_sym = "✓" if n["node_ok"] else "✗"
    npx_sym = "✓" if n["npx_ok"] else "✗"
    print(f"  Node      : [{node_sym}] {n['node']}")
    print(f"  npx       : [{npx_sym}] {n['npx']}")
    print()

    for k, v in keys.items():
        sym = "✓" if v else "✗"
        val = "set" if v else "MISSING"
        print(f"  {k:<24}: [{sym}] {val}")
    print()

    ant_sym = "✓" if ant.get("anthropic_ok") else "✗"
    ant_msg = "OK" if ant.get("anthropic_ok") else ant.get("anthropic_error", "failed")
    print(f"  Anthropic API : [{ant_sym}] {ant_msg}")

    mcp_sym = "✓" if mcp.get("mcp_ok") else "✗"
    mcp_msg = f"{mcp['mcp_cold_start_s']}s cold start"
    if not mcp.get("mcp_ok"):
        mcp_msg += f"  — {mcp.get('mcp_error', 'failed')}"
    print(f"  MCP server    : [{mcp_sym}] {mcp_msg}")
    print()
