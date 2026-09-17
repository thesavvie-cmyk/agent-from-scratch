"""Planning primitives and tools (block 15).

Design
------
A Plan is a list of Tasks stored as a plain dict in context.state["plan"].
This makes it persist across agent steps, survive session serialisation, and
appear in traces automatically (context.state is included in every AgentResult).

Plan status in instructions
----------------------------
When planning=True, Agent._prepare_llm_request appends plan.progress() to the
instruction list on *every* step.  This is the "goal retention" mechanism: the
model always sees which tasks are pending, in-progress, done, or failed, no
matter how much tool-call content has accumulated in the context window.

Think-first step
----------------
When think_first=True, Agent makes one no-tools LLM call before the main loop.
The response is stored as a regular Event so the reasoning appears in the trace
and informs all subsequent steps.  The step counter is NOT incremented (the
think call does not count against max_steps).
"""
from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel

from agentkit.context import ExecutionContext
from agentkit.tools.base import tool

# ── Status ─────────────────────────────────────────────────────────────────────


class TaskStatus(str, Enum):
    pending = "pending"
    in_progress = "in_progress"
    done = "done"
    failed = "failed"


_STATUS_SYMBOL: dict[TaskStatus, str] = {
    TaskStatus.pending: "[ ]",
    TaskStatus.in_progress: "[>]",
    TaskStatus.done: "[x]",
    TaskStatus.failed: "[!]",
}

# ── Data model ─────────────────────────────────────────────────────────────────


class Task(BaseModel):
    """One step in a plan."""

    id: str
    description: str
    status: TaskStatus = TaskStatus.pending
    result: str | None = None
    notes: str | None = None


class Plan(BaseModel):
    """Ordered list of tasks that comprise the agent's current goal."""

    tasks: list[Task]

    def next_pending(self) -> Task | None:
        """Return the first task that is still pending, or None."""
        return next((t for t in self.tasks if t.status == TaskStatus.pending), None)

    def mark(
        self,
        task_id: str,
        status: TaskStatus,
        result: str | None = None,
    ) -> None:
        """Update the status of a task by id.

        Raises ValueError with a clear message if *task_id* is not found — the
        agent gets this as a tool-result error and can self-correct.
        """
        for t in self.tasks:
            if t.id == task_id:
                t.status = status
                if result:
                    t.result = result
                return
        ids = ", ".join(t.id for t in self.tasks)
        raise ValueError(
            f"Task {task_id!r} not found in plan. Available ids: {ids}"
        )

    def is_complete(self) -> bool:
        """True when every task is done or failed (none pending or in-progress)."""
        return bool(self.tasks) and all(
            t.status in (TaskStatus.done, TaskStatus.failed) for t in self.tasks
        )

    def progress(self) -> str:
        """Compact one-line status for context injection.

        Example: "[x] search | [>] calculate | [ ] report"
        """
        return " | ".join(
            f"{_STATUS_SYMBOL[t.status]} {t.description}" for t in self.tasks
        )


# ── Agent instruction constants ────────────────────────────────────────────────

PLANNING_INSTRUCTIONS = (
    "You have planning tools. For any multi-step task:\n"
    "1. Begin by calling create_plan() with each step as a separate item.\n"
    "2. Work through steps in order: call update_task() to mark a step\n"
    "   in_progress when you start it, and done (or failed) when finished.\n"
    "3. Provide your final answer only when all tasks are done or failed.\n"
    "The current plan status is shown at the start of every message."
)

THINK_INSTRUCTIONS = (
    "Before using any tools, reason through this task step by step: "
    "what information do you need, what steps are required, and what order "
    "should they go in? Write out your analysis. "
    "Do not call any tools in this response."
)

# ── Planning tools ─────────────────────────────────────────────────────────────

_PLAN_KEY = "plan"


@tool
def create_plan(context: ExecutionContext, tasks: list[str]) -> str:
    """Create a step-by-step plan for the current task.

    Pass a list of step descriptions. If a plan already exists it is replaced
    (noted in the result). Steps are assigned ids t1, t2, t3, ...
    """
    replaced = _PLAN_KEY in context.state
    plan = Plan(
        tasks=[Task(id=f"t{i + 1}", description=d) for i, d in enumerate(tasks)]
    )
    context.state[_PLAN_KEY] = plan.model_dump()
    note = " (replaced previous plan)" if replaced else ""
    steps = "\n".join(f"  {t.id}: {t.description}" for t in plan.tasks)
    return f"Plan created{note}:\n{steps}"


@tool
def update_task(
    context: ExecutionContext,
    task_id: str,
    status: Literal["pending", "in_progress", "done", "failed"],
    result: str = "",
) -> str:
    """Update the status of a task in the current plan.

    task_id: task identifier (t1, t2, ...)
    status: new status — pending, in_progress, done, or failed
    result: optional result text or note to attach to the task
    """
    plan_data = context.state.get(_PLAN_KEY)
    if plan_data is None:
        return "No plan exists. Call create_plan() first."
    plan = Plan.model_validate(plan_data)
    try:
        plan.mark(task_id, TaskStatus(status), result=result or None)
    except ValueError as exc:
        return str(exc)
    context.state[_PLAN_KEY] = plan.model_dump()
    return f"Task {task_id} marked {status}. Progress: {plan.progress()}"


@tool
def get_plan(context: ExecutionContext) -> str:
    """Return the current plan with status of all tasks."""
    plan_data = context.state.get(_PLAN_KEY)
    if plan_data is None:
        return "No plan exists. Call create_plan() first."
    plan = Plan.model_validate(plan_data)
    lines = []
    for t in plan.tasks:
        sym = _STATUS_SYMBOL[t.status]
        line = f"  {sym} {t.id}: {t.description}"
        if t.result:
            line += f"\n       Result: {t.result}"
        lines.append(line)
    header = "Complete." if plan.is_complete() else "In progress."
    return f"Plan ({header})\n" + "\n".join(lines)
