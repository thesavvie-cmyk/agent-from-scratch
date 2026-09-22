"""Multi-agent workflow primitives (block 20).

Three workflow types
--------------------
SequentialWorkflow  — steps run in order; each step receives the previous
                      step's output as its input.
ParallelWorkflow    — all steps receive the same input and run concurrently;
                      results are merged by a caller-supplied function.
                      A single step failure never aborts the others
                      (asyncio.gather with return_exceptions=True).
LoopWorkflow        — one step is repeated until *condition* returns False
                      or *max_iterations* is reached.

WorkflowStep
------------
Wraps an Agent with a *share_context* flag.

  share_context=False (default)
      Each step gets a fresh ExecutionContext.  The agent sees only its own
      conversation; prior steps are invisible.  Token cost per step is minimal.

  share_context=True
      The running context is passed to the next step so the agent sees
      the full conversation history of all previous steps.  Token cost grows
      with each step; the benefit is that the agent can reference earlier work.

WorkflowResult
--------------
Aggregates all per-step AgentResults.  ``all_events`` collects events from
every step, deduplicating by event-id so shared contexts don't double-count.
``total_tokens`` sums token usage across all steps.
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from agentkit.agent import Agent, AgentResult
from agentkit.context import ExecutionContext
from agentkit.types import Event

logger = logging.getLogger(__name__)


# ── Data types ─────────────────────────────────────────────────────────────────


@dataclass
class WorkflowStep:
    """An agent together with its context-sharing policy."""

    agent: Agent
    share_context: bool = False


@dataclass
class WorkflowResult:
    """Aggregated result from a workflow run.

    Attributes
    ----------
    output:
        Final output (output of the last step, or merged output for parallel).
    step_results:
        One AgentResult per step (or per iteration for LoopWorkflow).
    error:
        Set when any step returned an error.  For ParallelWorkflow this is a
        semicolon-joined string of all per-step errors.
    """

    output: Any  # str | BaseModel
    step_results: list[AgentResult] = field(default_factory=list)
    error: str | None = None

    @property
    def all_events(self) -> list[Event]:
        """Events from every step in chronological order, deduplicated by id."""
        seen: set[str] = set()
        result: list[Event] = []
        for sr in self.step_results:
            for ev in sr.context.events:
                if ev.id not in seen:
                    seen.add(ev.id)
                    result.append(ev)
        return result

    def total_tokens(self) -> dict[str, int]:
        """Sum token usage across all steps."""
        inp = out = 0
        for sr in self.step_results:
            u = sr.context.state.get("token_usage", {})
            inp += u.get("input_tokens", 0)
            out += u.get("output_tokens", 0)
        return {"input_tokens": inp, "output_tokens": out}

    def tokens_per_step(self) -> list[dict[str, int]]:
        """Per-step token usage list, aligned with step_results."""
        result = []
        for sr in self.step_results:
            u = sr.context.state.get("token_usage", {})
            result.append(
                {
                    "input_tokens": u.get("input_tokens", 0),
                    "output_tokens": u.get("output_tokens", 0),
                }
            )
        return result


# ── Sequential ─────────────────────────────────────────────────────────────────


class SequentialWorkflow:
    """Run steps one after another; each step's output feeds the next step.

    Parameters
    ----------
    steps:
        Ordered list of WorkflowSteps.  Must be non-empty.
    """

    def __init__(self, steps: list[WorkflowStep]) -> None:
        if not steps:
            raise ValueError("SequentialWorkflow requires at least one step")
        self.steps = steps

    async def run(self, user_input: str) -> WorkflowResult:
        """Execute all steps and return a WorkflowResult."""
        from agentkit.telemetry import get_tracer

        step_results: list[AgentResult] = []
        current_input = user_input
        shared_ctx: ExecutionContext | None = None

        _tracer = get_tracer()
        with _tracer.start_as_current_span("workflow.sequential.run") as _span:
            _span.set_attribute("workflow.num_steps", len(self.steps))
            for step in self.steps:
                ctx = shared_ctx if step.share_context else None
                result = await step.agent.run(current_input, context=ctx)
                step_results.append(result)

                if step.share_context:
                    shared_ctx = result.context

                if result.error:
                    return WorkflowResult(
                        output=result.output,
                        step_results=step_results,
                        error=result.error,
                    )

                current_input = str(result.output)

        return WorkflowResult(
            output=step_results[-1].output,
            step_results=step_results,
        )


# ── Parallel ───────────────────────────────────────────────────────────────────


def _default_merge(outputs: list[str]) -> str:
    parts = [f"[{i + 1}] {o}" for i, o in enumerate(outputs) if o.strip()]
    return "\n\n---\n\n".join(parts)


class ParallelWorkflow:
    """Run all steps concurrently; merge their outputs.

    One step's exception does not abort the others: failures are collected
    and reported in ``WorkflowResult.error`` while the successful outputs
    are still merged.

    Parameters
    ----------
    steps:
        Steps to run in parallel.  All receive the same *user_input*.
    merge:
        Function ``(list[str]) -> str`` to combine per-step outputs.
        Defaults to numbered sections separated by ``---``.
    """

    def __init__(
        self,
        steps: list[WorkflowStep],
        merge: Callable[[list[str]], str] | None = None,
    ) -> None:
        if not steps:
            raise ValueError("ParallelWorkflow requires at least one step")
        self.steps = steps
        self.merge = merge or _default_merge

    async def run(self, user_input: str) -> WorkflowResult:
        """Run all steps concurrently; merge results."""
        from agentkit.telemetry import get_tracer

        _tracer = get_tracer()
        with _tracer.start_as_current_span("workflow.parallel.run") as _span:
            _span.set_attribute("workflow.num_steps", len(self.steps))
            tasks = [step.agent.run(user_input) for step in self.steps]
            raw = await asyncio.gather(*tasks, return_exceptions=True)

        step_results: list[AgentResult] = []
        outputs: list[str] = []
        errors: list[str] = []

        for i, res in enumerate(raw):
            if isinstance(res, Exception):
                logger.warning("Parallel step %d raised: %s", i, res)
                step_results.append(
                    AgentResult(
                        output="",
                        context=ExecutionContext(),
                        error=str(res),
                    )
                )
                outputs.append("")
                errors.append(f"step {i}: {res}")
            else:
                step_results.append(res)
                outputs.append(str(res.output))
                if res.error:
                    errors.append(f"step {i}: {res.error}")

        merged = self.merge(outputs)
        return WorkflowResult(
            output=merged,
            step_results=step_results,
            error="; ".join(errors) if errors else None,
        )


# ── Loop ───────────────────────────────────────────────────────────────────────


class LoopWorkflow:
    """Repeat a single step until *condition* returns False or the iteration
    cap is hit.

    The condition is evaluated *after* each run.  Returning ``False`` stops the
    loop; ``True`` means "keep going, run again".  The loop always stops after
    *max_iterations* regardless of the condition.

    The current step's output becomes the input for the next iteration.

    Parameters
    ----------
    step:
        The WorkflowStep to repeat.
    condition:
        ``(AgentResult) -> bool``.  ``True`` = keep looping.
    max_iterations:
        Hard cap.  Must be >= 1.
    """

    def __init__(
        self,
        step: WorkflowStep,
        condition: Callable[[AgentResult], bool],
        max_iterations: int = 3,
    ) -> None:
        if max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        self.step = step
        self.condition = condition
        self.max_iterations = max_iterations

    async def run(self, user_input: str) -> WorkflowResult:
        """Execute the loop and return a WorkflowResult with one entry per
        iteration."""
        step_results: list[AgentResult] = []
        current_input = user_input
        shared_ctx: ExecutionContext | None = None

        for _i in range(self.max_iterations):
            ctx = shared_ctx if self.step.share_context else None
            result = await self.step.agent.run(current_input, context=ctx)
            step_results.append(result)

            if self.step.share_context:
                shared_ctx = result.context

            if result.error:
                return WorkflowResult(
                    output=result.output,
                    step_results=step_results,
                    error=result.error,
                )

            if not self.condition(result):
                break

            current_input = str(result.output)

        return WorkflowResult(
            output=step_results[-1].output,
            step_results=step_results,
        )
