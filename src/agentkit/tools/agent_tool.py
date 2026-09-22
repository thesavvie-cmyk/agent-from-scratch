"""AgentTool — wrap an Agent as a callable tool for orchestrators (block 21).

The child agent runs in a fresh ExecutionContext; the parent's context and
history are never passed to it.  The *request* string (or validated Pydantic
model JSON when *input_schema* is given) is the only information the child
receives.

Child traces are preserved in the parent context under::

    context.state["child_traces"][<8-char hex id>]

Recursion protection
--------------------
``_agent_call_stack`` is propagated through child contexts so that chains
like A → B → A are detected, not just direct self-calls.
A hard ``max_depth`` cap provides a second layer of defence.

Error handling
--------------
Any failure (child error, validation failure, exception, recursion) raises
``ChildAgentError``.  Agent.act() catches this and produces a
``ToolResult(status="error")``, so the orchestrator continues running.
"""
from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel

from agentkit.agent import (  # no circular dep: agent.py never imports agent_tool
    Agent,
    AgentResult,
)
from agentkit.context import ExecutionContext
from agentkit.schema import build_tool_definition
from agentkit.tools.base import BaseTool


class ChildAgentError(RuntimeError):
    """Raised when a child agent fails or is blocked by recursion protection."""


class AgentTool(BaseTool):
    """Wraps an Agent so orchestrators can call it as a tool.

    Default schema: a single ``request: str`` parameter.
    Pass *input_schema* (a Pydantic BaseModel subclass) for richer schemas
    with automatic validation.

    Parameters
    ----------
    agent:
        The agent to wrap.
    name:
        Tool name seen by the LLM.  Defaults to ``agent.name``.
    description:
        Tool description.  Defaults to the first 200 chars of the agent's
        instructions.
    input_schema:
        Optional Pydantic model.  When set, the tool's JSON schema mirrors
        the model and input is validated before the child runs.
    max_depth:
        Maximum nesting depth.  Prevents runaway chains even when direct
        recursion is not involved.
    """

    def __init__(
        self,
        agent: Agent,
        name: str | None = None,
        description: str | None = None,
        input_schema: type[BaseModel] | None = None,
        max_depth: int = 3,
    ) -> None:
        _name = name or agent.name
        _desc = description or (
            agent.instructions[:200].strip()
            if agent.instructions
            else f"Call the {agent.name} agent."
        )
        super().__init__(name=_name, description=_desc)
        self._agent = agent
        self._input_schema = input_schema
        self._max_depth = max_depth

    def _generate_definition(self) -> dict[str, Any]:
        if self._input_schema is not None:
            schema = self._input_schema.model_json_schema()
            # Drop keys OpenAI doesn't accept at the parameters level
            params = {k: v for k, v in schema.items() if k not in ("title", "$defs")}
        else:
            params = {
                "type": "object",
                "properties": {
                    "request": {
                        "type": "string",
                        "description": "The request or question to send to the agent.",
                    }
                },
                "required": ["request"],
            }
        return build_tool_definition(self.name, self.description, params)

    async def execute(self, context: ExecutionContext, **kwargs: Any) -> str:
        """Run the child agent and return its output as a string.

        Raises ChildAgentError on any failure so Agent.act() records
        ``status="error"`` in the parent's trace.
        """
        # ── Recursion / depth protection ─────────────────────────────────────
        call_stack: list[str] = list(context.state.get("_agent_call_stack", []))
        if self._agent.name in call_stack:
            raise ChildAgentError(
                f"Recursive call detected: '{self._agent.name}' is already "
                f"in the call stack {call_stack!r}"
            )
        if len(call_stack) >= self._max_depth:
            raise ChildAgentError(
                f"Maximum agent nesting depth ({self._max_depth}) exceeded"
            )

        # ── Input extraction / validation ─────────────────────────────────────
        if self._input_schema is not None:
            try:
                validated = self._input_schema.model_validate(kwargs)
                request = validated.model_dump_json()
            except Exception as exc:
                raise ChildAgentError(
                    f"Invalid input for '{self._agent.name}': {exc}"
                ) from exc
        else:
            request = str(kwargs.get("request", ""))

        # ── Isolated child context with propagated call stack ─────────────────
        child_ctx = ExecutionContext()
        child_ctx.state["_agent_call_stack"] = call_stack + [self._agent.name]

        # ── Run child ──────────────────────────────────────────────────────────
        trace_id = uuid.uuid4().hex[:8]
        try:
            result: AgentResult = await self._agent.run(request, context=child_ctx)
        except Exception as exc:
            context.state.setdefault("child_traces", {})[trace_id] = {
                "agent": self._agent.name,
                "error": str(exc),
                "steps": 0,
                "events": 0,
            }
            raise ChildAgentError(
                f"Agent '{self._agent.name}' raised an exception: {exc}"
            ) from exc

        # ── Save child trace in parent state ──────────────────────────────────
        context.state.setdefault("child_traces", {})[trace_id] = {
            "agent": self._agent.name,
            "steps": result.context.current_step,
            "events": len(result.context.events),
            "error": result.error,
        }

        if result.error:
            raise ChildAgentError(
                f"Agent '{self._agent.name}' failed: {result.error}"
            )

        return str(result.output)
