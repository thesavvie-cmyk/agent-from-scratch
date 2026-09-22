"""Specialist agent factories (block 20).

Each factory returns a pre-configured Agent with a focused system prompt and
a narrow tool set.  The prompts are intentionally short (≤10 lines each) so
token cost stays low when the agent is one step in a larger workflow.

Usage
-----
    from agentkit.agents.specialists import make_researcher, make_writer
    from agentkit.llm import LlmClient

    model = LlmClient("anthropic/claude-haiku-4-5")
    researcher = make_researcher(model, tools=search_tools)
    writer = make_writer(model)
"""
from __future__ import annotations

from agentkit.agent import Agent
from agentkit.llm import LlmClient
from agentkit.tools.base import BaseTool


def make_researcher(
    model: LlmClient,
    tools: list[BaseTool] | None = None,
    max_steps: int = 8,
) -> Agent:
    """Return a research specialist agent.

    Searches for accurate, current information and produces a structured
    summary of key facts, statistics, and context.
    """
    return Agent(
        model=model,
        tools=tools or [],
        name="researcher",
        instructions=(
            "You are a research specialist.\n"
            "Search for accurate, current information on the given topic.\n"
            "Output a structured summary: key facts, statistics, and context.\n"
            "Be thorough but concise — focus on what matters for the next step."
        ),
        max_steps=max_steps,
    )


def make_coder(
    model: LlmClient,
    tools: list[BaseTool] | None = None,
    max_steps: int = 6,
) -> Agent:
    """Return a coding specialist agent.

    Writes clean, working code from the provided requirements.
    """
    return Agent(
        model=model,
        tools=tools or [],
        name="coder",
        instructions=(
            "You are a coding specialist.\n"
            "Write clean, working code based on the provided requirements.\n"
            "Use standard libraries. Add brief inline comments for non-obvious logic.\n"
            "Output the code followed by a one-sentence explanation of what it does."
        ),
        max_steps=max_steps,
    )


def make_writer(
    model: LlmClient,
    tools: list[BaseTool] | None = None,
    max_steps: int = 4,
) -> Agent:
    """Return a writing specialist agent.

    Transforms research notes into clear, engaging prose.
    """
    return Agent(
        model=model,
        tools=tools or [],
        name="writer",
        instructions=(
            "You are a writing specialist.\n"
            "Transform research and notes into clear, engaging prose.\n"
            "Write in an informative, professional tone. Use concrete examples.\n"
            "Produce a complete, well-structured article or section."
        ),
        max_steps=max_steps,
    )


def make_reviewer(
    model: LlmClient,
    tools: list[BaseTool] | None = None,
    max_steps: int = 4,
) -> Agent:
    """Return an editorial reviewer agent.

    Evaluates text for accuracy, clarity, and structure.
    Replies with APPROVED or REVISION NEEDED plus brief feedback.
    """
    return Agent(
        model=model,
        tools=tools or [],
        name="reviewer",
        instructions=(
            "You are an editorial reviewer.\n"
            "Evaluate the provided text for accuracy, clarity, and structure.\n"
            "If acceptable: reply APPROVED: <one-line reason>.\n"
            "If it needs work: reply REVISION NEEDED: <specific, actionable feedback>.\n"
            "Be concise."
        ),
        max_steps=max_steps,
    )
