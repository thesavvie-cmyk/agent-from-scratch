"""Plan critique loop (block 26).

Design
------
After the agent calls create_plan(), the critique loop runs N rounds of:
  1. Critic LLM call (clean context, no tools): reviews the plan and lists
     issues (missing steps, wrong order, vague steps, irrelevant steps).
  2. Revisor LLM call (clean context, no tools): rewrites the plan as a
     JSON array of step strings, addressing the critique.

Both calls share the same LlmClient but use a completely clean context
(no conversation history) so the main agent's tool calls and intermediate
observations do not bias the critique.

If the critic says "no issues" the loop exits early.

Integration
-----------
Agent(planning="critiqued", critique_rounds=N):
  - After create_plan executes, Agent._critique_plan(ctx, task) is awaited.
  - context.state["plan"] is replaced with the revised Plan.
  - The main ReAct loop then proceeds with the improved plan.
"""
from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING

from agentkit.llm import LlmRequest
from agentkit.planning import Plan, Task
from agentkit.types import Message

if TYPE_CHECKING:
    from agentkit.llm import LlmClient

logger = logging.getLogger(__name__)

# ── Prompts ───────────────────────────────────────────────────────────────────

_CRITIC_SYSTEM = (
    "You are a plan critic. Your job is to review a step-by-step plan and "
    "identify weaknesses: missing steps, wrong order, overly vague steps, "
    "redundant steps, or steps that do not contribute to the stated goal.\n"
    "Be concise. List 1–5 specific issues as bullet points. "
    "If the plan looks complete and well-ordered, write exactly: No issues."
    "\nDo NOT rewrite the plan — only list problems."
)

_REVISOR_SYSTEM = (
    "You are a plan editor. You receive a task description, the current plan, "
    "and a list of critique points. Rewrite the plan to address the critique. "
    "Keep every step that is already correct. Add, remove, reorder, or "
    "rephrase steps only where the critique demands it.\n"
    "Output ONLY a JSON array of step-description strings, with no markdown "
    "fences and no extra text. Example:\n"
    '[\"Search for X\", \"Calculate Y from X\", \"Report the result\"]'
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _plan_to_numbered_list(plan: Plan) -> str:
    return "\n".join(f"{i + 1}. {t.description}" for i, t in enumerate(plan.tasks))


def _parse_revised_steps(text: str) -> list[str] | None:
    """Extract a JSON array of strings from LLM text.  Returns None on failure."""
    # Strip markdown fences if present
    text = re.sub(r"```[a-z]*\n?", "", text).strip()
    try:
        data = json.loads(text)
        if isinstance(data, list) and all(isinstance(s, str) for s in data):
            return [s.strip() for s in data if s.strip()]
    except json.JSONDecodeError:
        pass
    # Fallback: try to find first [...] block
    m = re.search(r"\[.*?\]", text, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group())
            if isinstance(data, list) and all(isinstance(s, str) for s in data):
                return [s.strip() for s in data if s.strip()]
        except json.JSONDecodeError:
            pass
    return None


# ── Core functions ────────────────────────────────────────────────────────────


async def _get_critique(model: LlmClient, task: str, plan: Plan) -> str:
    """Ask the critic to review the plan.  Returns critique text."""
    user_msg = (
        f"Task: {task}\n\n"
        f"Plan:\n{_plan_to_numbered_list(plan)}\n\n"
        "List any issues with this plan."
    )
    request = LlmRequest(
        instructions=[_CRITIC_SYSTEM],
        contents=[Message(role="user", content=user_msg)],
        tools=[],
        tool_choice=None,
    )
    response = await model.generate(request)
    if response.error_message:
        logger.warning("Critic LLM error: %s", response.error_message)
        return "No issues."
    for item in response.content:
        if isinstance(item, Message):
            return item.content
    return "No issues."


async def _get_revised_plan(
    model: LlmClient, task: str, plan: Plan, critique: str
) -> Plan:
    """Ask the revisor to rewrite the plan based on the critique.

    Returns a new Plan with the same id scheme (t1, t2, ...).
    Falls back to the original plan if parsing fails.
    """
    user_msg = (
        f"Task: {task}\n\n"
        f"Current plan:\n{_plan_to_numbered_list(plan)}\n\n"
        f"Critique:\n{critique}\n\n"
        "Rewrite the plan addressing the critique."
    )
    request = LlmRequest(
        instructions=[_REVISOR_SYSTEM],
        contents=[Message(role="user", content=user_msg)],
        tools=[],
        tool_choice=None,
    )
    response = await model.generate(request)
    if response.error_message:
        logger.warning("Revisor LLM error: %s", response.error_message)
        return plan

    text = ""
    for item in response.content:
        if isinstance(item, Message):
            text = item.content
            break

    steps = _parse_revised_steps(text)
    if not steps:
        logger.warning(
            "Revisor returned unparseable output — keeping original plan. "
            "Raw: %.120s", text
        )
        return plan

    new_plan = Plan(
        tasks=[Task(id=f"t{i + 1}", description=d) for i, d in enumerate(steps)]
    )
    logger.debug(
        "Plan revised: %d → %d tasks", len(plan.tasks), len(new_plan.tasks)
    )
    return new_plan


# ── Public API ────────────────────────────────────────────────────────────────


async def run_critique_rounds(
    model: LlmClient,
    task: str,
    plan: Plan,
    rounds: int = 2,
) -> Plan:
    """Run up to *rounds* critique+revise cycles; return the improved Plan.

    Parameters
    ----------
    model:
        The LlmClient to use for both critic and revisor calls.
        Typically the smart model (SMART_MODEL) for better critique quality.
    task:
        One-sentence description of what the agent is trying to accomplish.
        Taken from the user's original question.
    plan:
        The initial Plan produced by the agent's create_plan() call.
    rounds:
        Maximum number of critique+revise cycles.  Exits early if the
        critic reports "No issues".  Default: 2.
    """
    current = plan
    for i in range(rounds):
        critique = await _get_critique(model, task, current)
        logger.debug("Critique round %d: %s", i + 1, critique[:120])
        if "no issues" in critique.lower():
            logger.debug("Critic satisfied after %d round(s) — stopping early.", i + 1)
            break
        current = await _get_revised_plan(model, task, current, critique)
    return current
