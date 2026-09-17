"""Reflection tool — self-monitoring for agent loops (block 16).

Design
------
The reflection tool gives the agent a structured pause to reason about its
own progress. It writes a machine-readable flag (need_replan) to
context.state so the Agent loop — not just the model — can observe it and
inject an explicit replan instruction on the next step.

The entire usage contract lives in the docstring of the tool function below.
The LLM reads it as the tool description, so the docstring IS the prompt.
"""
from __future__ import annotations

from agentkit.context import ExecutionContext
from agentkit.tools.base import tool

REFLECTION_INSTRUCTIONS = (
    "After every 2-3 tool calls, or before giving your final answer, "
    "call reflection() to assess your progress. "
    "Set need_replan=True if you are repeating the same searches, "
    "hitting repeated errors, or have drifted from the original goal."
)

_REPLAN_INSTRUCTION = (
    "REPLAN REQUIRED: Your reflection flagged need_replan=True. "
    "Your current approach is not working. Stop and reconsider your strategy: "
    "try different search terms, a different tool, or break the task into "
    "smaller steps before proceeding."
)


@tool
def reflection(
    context: ExecutionContext,
    analysis: str,
    need_replan: bool = False,
) -> str:
    """Pause and assess your current progress before continuing.

    Call this tool to reflect on what you have done so far, whether it is
    working, and whether you should change strategy. The analysis you provide
    becomes part of the trace and informs your next steps.

    Parameters
    ----------
    analysis : str
        Your honest assessment of the current situation. Be specific:
        what did you find, what failed, what is missing.
    need_replan : bool, default False
        Set to True when your current approach is clearly not working and
        you need to change strategy. The agent loop will inject an explicit
        replan instruction into your next step.

    When to call
    ------------
    PROGRESS REVIEW -- after 2-3 tool calls, take stock of what you have:
        analysis="I searched for X and found Y. I still need Z to answer
        the question. Next: search for Z."
        need_replan=False

    ERROR ANALYSIS -- when the same tool keeps failing:
        analysis="get_wikipedia_page raised RuntimeError twice. Wikipedia
        is unavailable. I should switch to search_web instead."
        need_replan=True

    RESULT SYNTHESIS -- when combining information from multiple sources:
        analysis="Source A says 384,400 km (mean distance). Source B says
        362,600 km (perigee). The question asks for average distance, so
        I should use 384,400 km and note the distinction."
        need_replan=False

    SELF CHECK -- before giving final_answer, verify completeness:
        analysis="The question asks for 3 laureates. I found Hinton and
        Hopfield (2024) but have not confirmed the 2023 winner yet.
        I need one more search before I can answer."
        need_replan=False

    When NOT to call
    ----------------
    - Simple single-step tasks ("What is 2+2?", "Translate this word") --
      the overhead is not justified
    - When you already have all the information you need -- call final_answer
      directly instead
    - After every single tool call -- reflection is a strategic pause, not
      a mechanical step between every action
    - As a substitute for actually doing the work -- reflect, then act
    """
    context.state["need_replan"] = need_replan
    flag = " [need_replan=True flagged]" if need_replan else ""
    return f"Reflection recorded.{flag}\n{analysis}"
