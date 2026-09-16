"""Chapter 3 MCP demo (block 5): McpToolset with official + custom servers.

Sections:
  A -- official tavily-mcp: list tools, compare descriptions
  B -- direct .call() on official server
  C -- custom server (tavily_server.py): list tools
  D -- Kipchoge task with official server (tavily_search + tavily_extract)
  E -- Kipchoge task with custom server (search_web only)
"""
from __future__ import annotations

import asyncio
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

from agentkit.config import SMART_MODEL
from agentkit.loop import async_simple_agent_loop
from agentkit.mcp_client import McpToolset


def sep(title: str) -> None:
    print(f"\n{'=' * 65}")
    print(f"  {title}")
    print("=" * 65)


SYSTEM = (
    "You are a helpful research assistant. "
    "When looking up the Moon's perigee distance, retrieve the exact value "
    "from the Wikipedia article about the Moon (not from memory). "
    "Use the minimum perigee (closest approach) value."
)

KIPCHOGE_Q = (
    "If Eliud Kipchoge could maintain his record-making marathon pace indefinitely, "
    "how many thousand hours would it take him to run the distance between the Earth "
    "and the Moon at its closest approach? "
    "Use the minimum perigee value from the Wikipedia page for the Moon. "
    "Round to the nearest 1000 hours."
)

OFFICIAL_CMD = ("npx", ["-y", "tavily-mcp@latest"])
CUSTOM_CMD = ("uv", ["run", "python", "-m", "agentkit.servers.tavily_server"])


async def main() -> None:
    # ── A. Official server: list tools ────────────────────────────────────────
    sep("A. Official tavily-mcp: tool list and descriptions")

    try:
        async with McpToolset(*OFFICIAL_CMD) as ts:
            defns = ts.tool_definitions()
            print(f"\nOfficial server: {len(defns)} tool(s)")
            for d in defns:
                fn = d["function"]
                desc_preview = fn["description"][:120].replace("\n", " ")
                params = list(fn.get("parameters", {}).get("properties", {}).keys())
                print(f"\n  [{fn['name']}]")
                print(f"  desc  : {desc_preview}...")
                print(f"  params: {params}")

            print("\n[Comparison with function_to_tool_definition (block 4)]")
            print("  Official: full per-param descriptions, enum constraints, rich schema")
            print("  Generated: first-paragraph docstring only, no per-param docs, no enums")
            print("  -> Handwritten / official schemas are significantly more informative")

            # ── B. Direct .call() ──────────────────────────────────────────────
            sep("B. Direct .call() on official server")
            result = await ts.call("tavily_search", {"query": "Eliud Kipchoge marathon world record 2022"})
            preview = result[:400].replace("\n", " ")
            print(f"\nResult preview: {preview!r}")

    except RuntimeError as exc:
        print(f"\n[SKIP] Official tavily-mcp unavailable: {exc}")
        print("       Install Node.js and npx to enable this section.")
        official_available = False
    else:
        official_available = True

    # ── C. Custom server: list tools ──────────────────────────────────────────
    sep("C. Custom server (tavily_server.py): tool list")

    async with McpToolset(*CUSTOM_CMD) as ts:
        defns = ts.tool_definitions()
        print(f"\nCustom server: {len(defns)} tool(s)")
        for d in defns:
            fn = d["function"]
            desc_preview = fn["description"][:120].replace("\n", " ")
            params = list(fn.get("parameters", {}).get("properties", {}).keys())
            print(f"\n  [{fn['name']}]")
            print(f"  desc  : {desc_preview}...")
            print(f"  params: {params}")

    # ── D. Kipchoge with official server ──────────────────────────────────────
    sep("D. Kipchoge task — official server, expected: 17 (thousand hours)")

    if not official_available:
        print("\n[SKIP] Official server unavailable.")
    else:
        async with McpToolset(*OFFICIAL_CMD) as ts:
            answer_d, history_d = await async_simple_agent_loop(
                SYSTEM, KIPCHOGE_Q, [], model=SMART_MODEL, max_rounds=14, mcp_toolsets=[ts]
            )

        print(f"\nAnswer: {answer_d[:300]!r}")
        print("\nCall chain:")
        for m in history_d:
            if m["role"] == "assistant" and m.get("tool_calls"):
                for tc in m["tool_calls"]:
                    args_preview = tc["function"]["arguments"][:80].replace("\n", " ")
                    print(f"  -> {tc['function']['name']}({args_preview})")
            elif m["role"] == "tool":
                preview = m["content"][:120].replace("\n", " ")
                print(f"  <- {preview!r}")

    # ── E. Kipchoge with custom server ────────────────────────────────────────
    sep("E. Kipchoge task — custom server (search_web only), expected: 17")

    async with McpToolset(*CUSTOM_CMD) as ts:
        answer_e, history_e = await async_simple_agent_loop(
            SYSTEM, KIPCHOGE_Q, [], model=SMART_MODEL, max_rounds=12, mcp_toolsets=[ts]
        )

    print(f"\nAnswer: {answer_e[:300]!r}")
    print("\nCall chain:")
    for m in history_e:
        if m["role"] == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                args_preview = tc["function"]["arguments"][:80].replace("\n", " ")
                print(f"  -> {tc['function']['name']}({args_preview})")
        elif m["role"] == "tool":
            preview = m["content"][:120].replace("\n", " ")
            print(f"  <- {preview!r}")

    # Summary
    sep("Summary")
    print("\n  D (official): used tavily_extract for Wikipedia?")
    if official_available:
        extract_used = any(
            m.get("tool_calls") and any(
                tc["function"]["name"] == "tavily_extract"
                for tc in m["tool_calls"]
            )
            for m in history_d
            if m["role"] == "assistant"
        )
        print(f"    tavily_extract called: {extract_used}")
    else:
        print("    (skipped)")
    print("  E (custom): search_web only (no extract tool available)")
    search_calls_e = sum(
        1 for m in history_e
        if m["role"] == "assistant" and m.get("tool_calls")
        for tc in m["tool_calls"]
        if tc["function"]["name"] == "search_web"
    )
    print(f"    search_web calls: {search_calls_e}")


if __name__ == "__main__":
    asyncio.run(main())
