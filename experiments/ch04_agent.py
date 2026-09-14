"""Chapter 4 agent demo (block 8): Agent class, structured output.

Sections:
  A -- calculator agent: 1234 * 5678
  B -- Kipchoge question with MCP tools; full display_trace
  C -- structured output: SentimentAnalysis without tools
  D -- contrast: same Kipchoge question without tools
"""
from __future__ import annotations

import asyncio
import sys
from typing import Literal

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

from pydantic import BaseModel

from agentkit.agent import Agent
from agentkit.config import FAST_MODEL
from agentkit.llm import LlmClient
from agentkit.mcp_client import McpToolset
from agentkit.tools.base import FunctionTool
from agentkit.tools.mcp import load_mcp_tools
from agentkit.tools_manual import calculator
from agentkit.utils import display_trace

MCP_CMD = ("uv", ["run", "python", "-m", "agentkit.servers.tavily_server"])
KIPCHOGE_Q = (
    "How many hours of training did marathon world record holder "
    "Eliud Kipchoge reportedly do per week to prepare for his record run? "
    "Give just the number."
)


def sep(title: str) -> None:
    print(f"\n{'=' * 65}")
    print(f"  {title}")
    print("=" * 65)


async def main() -> None:
    client = LlmClient(FAST_MODEL)

    # ── A. Calculator agent ────────────────────────────────────────────────────
    sep("A. Calculator agent: 1234 * 5678")

    calc_tool = FunctionTool(calculator)
    agent_calc = Agent(
        model=client,
        tools=[calc_tool],
        instructions="Use the calculator tool for arithmetic. Show your work.",
        max_steps=5,
        name="calc_agent",
    )
    result_a = await agent_calc.run("What is 1234 multiplied by 5678?")
    print(f"\nOutput : {result_a.output}")
    print(f"Steps  : {result_a.context.current_step}")
    usage_a = result_a.context.state.get("token_usage", {})
    print(f"Tokens : {usage_a}")

    # ── B. Kipchoge with MCP tools + display_trace ─────────────────────────────
    sep("B. Kipchoge with MCP tools + display_trace")

    async with McpToolset(*MCP_CMD) as ts:
        mcp_tools = load_mcp_tools(ts)
        agent_mcp = Agent(
            model=client,
            tools=mcp_tools,
            instructions="Use the search tools to look up information. Be concise.",
            max_steps=8,
            name="mcp_agent",
        )
        result_b = await agent_mcp.run(KIPCHOGE_Q)

    print(f"\nOutput : {result_b.output}")
    print("\n--- Trace ---")
    print(display_trace(result_b.context))

    # ── C. Structured output: SentimentAnalysis ────────────────────────────────
    sep("C. Structured output: SentimentAnalysis")

    class SentimentAnalysis(BaseModel):
        sentiment: Literal["positive", "negative", "neutral"]
        confidence: float
        key_phrases: list[str]

    agent_struct = Agent(
        model=client,
        output_type=SentimentAnalysis,
        instructions="Analyze the sentiment of the user's text.",
        max_steps=3,
        name="sentiment_agent",
    )
    text = "I absolutely love this product! It completely changed my life for the better!"
    result_c = await agent_struct.run(text)

    print(f"\nInput    : {text[:60]!r}")
    print(f"Type     : {type(result_c.output).__name__}")
    print(f"Output   : {result_c.output}")
    print(f"Is model : {isinstance(result_c.output, SentimentAnalysis)}")

    # ── D. No tools — contrast ─────────────────────────────────────────────────
    sep("D. Same Kipchoge question — no tools (contrast)")

    agent_notools = Agent(
        model=client,
        instructions="You are a helpful assistant. Answer from your training knowledge.",
        max_steps=3,
        name="notools_agent",
    )
    result_d = await agent_notools.run(KIPCHOGE_Q)

    print(f"\nOutput : {result_d.output}")
    print(f"Steps  : {result_d.context.current_step}")
    print("\n  -> Without tools the model falls back to training knowledge,")
    print("     which may be incomplete or uncertain about specific numbers.")


if __name__ == "__main__":
    asyncio.run(main())
