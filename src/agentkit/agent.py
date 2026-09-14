"""Agent class — ReAct loop with structured output support (block 8)."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from agentkit.context import ExecutionContext
from agentkit.llm import LlmClient, LlmRequest, LlmResponse
from agentkit.schema import build_tool_definition
from agentkit.tools.base import BaseTool, FunctionTool
from agentkit.types import Event, Message, ToolCall, ToolResult

logger = logging.getLogger(__name__)


class _LlmTransientError(RuntimeError):
    """Raised by _step when LlmClient returns error_message (transient fault).

    Config errors (bad API key, unknown model) are NOT wrapped here — they
    propagate as the original litellm exceptions so the caller can distinguish
    them from transient failures.
    """


@dataclass
class AgentResult:
    """Output from a single agent invocation."""

    output: str | BaseModel
    context: ExecutionContext
    error: str | None = None


class Agent:
    """ReAct agent: think → act → observe loop.

    When *output_type* is a Pydantic BaseModel subclass the agent
    automatically adds a ``final_answer`` tool whose schema is derived from
    ``output_type.model_json_schema()``.  Every step is forced (tool_choice
    = "required") so the model ultimately calls ``final_answer`` to commit
    its structured answer.

    Error handling for LlmResponse.error_message
    --------------------------------------------
    LiteLLM already retries internally on transient errors (num_retries).
    Re-trying at the agent level would double the wait and still fail on
    persistent errors.  When error_message is set, _step() raises
    _LlmTransientError; run() catches it and returns AgentResult(error=...).
    Configuration errors (bad key, unknown model) propagate as the original
    litellm exceptions — they bypass the _LlmTransientError catch so the
    caller can distinguish them from transient network faults.
    """

    def __init__(
        self,
        model: LlmClient,
        tools: list[BaseTool] | None = None,
        instructions: str = "",
        max_steps: int = 10,
        name: str = "agent",
        output_type: type[BaseModel] | None = None,
    ) -> None:
        self.model = model
        self.instructions = instructions
        self.max_steps = max_steps
        self.name = name
        self.output_type = output_type
        self._setup_tools(tools)

    # ── Setup ─────────────────────────────────────────────────────────────────

    def _setup_tools(self, tools: list[BaseTool] | None) -> None:
        """Build tool registry; add final_answer tool when output_type is set."""
        self._toolbox: dict[str, BaseTool] = {t.name: t for t in (tools or [])}
        self.output_tool_name: str | None = None

        if self.output_type is not None:
            output_cls = self.output_type

            def _final_answer(**kwargs: Any) -> BaseModel:
                """Call this tool to submit your final structured answer."""
                return output_cls.model_validate(kwargs)

            ft = FunctionTool(_final_answer)
            # Override schema: use output_cls.model_json_schema() as parameters.
            # FunctionTool's auto-schema can't inspect **kwargs usefully.
            ft._tool_definition = build_tool_definition(
                "final_answer",
                "Call this tool to submit your final structured answer.",
                output_cls.model_json_schema(),
            )
            self._toolbox["final_answer"] = ft
            self.output_tool_name = "final_answer"

    # ── Public API ─────────────────────────────────────────────────────────────

    async def run(
        self,
        user_input: str,
        context: ExecutionContext | None = None,
    ) -> AgentResult:
        """Run the agent on *user_input* and return the final result."""
        ctx = context or ExecutionContext()
        ctx.add_message("user", user_input, author=self.name)

        final_result: str | BaseModel | None = None
        run_error: str | None = None
        try:
            while final_result is None and ctx.current_step < self.max_steps:
                final_result = await self._step(ctx)
        except _LlmTransientError as exc:
            run_error = str(exc)

        if run_error is not None:
            ctx.final_result = ""
            return AgentResult(output="", error=run_error, context=ctx)

        output: str | BaseModel = (
            final_result
            if final_result is not None
            else f"[max_steps={self.max_steps} reached without final answer]"
        )
        ctx.final_result = output
        return AgentResult(output=output, context=ctx)

    # ── Loop internals ─────────────────────────────────────────────────────────

    async def _step(self, context: ExecutionContext) -> str | BaseModel | None:
        """One think→act cycle.  Returns the final result or None to continue."""
        request = self._prepare_llm_request(context)
        response = await self.think(request)

        # Accumulate token usage in context state for downstream reporting
        _usage = context.state.setdefault(
            "token_usage", {"input_tokens": 0, "output_tokens": 0}
        )
        _usage["input_tokens"] += response.usage_metadata.get("input_tokens", 0)
        _usage["output_tokens"] += response.usage_metadata.get("output_tokens", 0)

        if response.error_message:
            logger.warning(
                "LLM transient error at step %d: %s",
                context.current_step,
                response.error_message,
            )
            raise _LlmTransientError(response.error_message)

        # Record think event (may contain Message + ToolCall items)
        think_event = Event(
            execution_id=context.execution_id,
            author=self.name,
            content=response.content,  # type: ignore[arg-type]
        )
        context.add_event(think_event)

        if self._is_final_response(think_event):
            context.increment_step()
            return self._extract_final_result(think_event)

        # Execute tool calls; record results
        tool_calls = [i for i in response.content if isinstance(i, ToolCall)]
        if tool_calls:
            results = await self.act(context, tool_calls)
            context.add_tool_results(results, author=self.name)

        context.increment_step()
        return None

    def _prepare_llm_request(self, context: ExecutionContext) -> LlmRequest:
        instructions = [self.instructions] if self.instructions else []
        contents = list(context.iter_content())

        if self.output_tool_name is not None:
            tool_choice: str | None = "required"
        elif self._toolbox:
            tool_choice = "auto"
        else:
            tool_choice = None

        return LlmRequest(
            instructions=instructions,
            contents=contents,
            tools=list(self._toolbox.values()),
            tool_choice=tool_choice,
        )

    async def think(self, request: LlmRequest) -> LlmResponse:
        """Delegate to LlmClient.generate."""
        return await self.model.generate(request)

    async def act(
        self,
        context: ExecutionContext,
        tool_calls: list[ToolCall],
    ) -> list[ToolResult]:
        """Execute tool calls; unknown tools and exceptions → status='error'."""
        results: list[ToolResult] = []
        for call in tool_calls:
            tool = self._toolbox.get(call.name)
            if tool is None:
                results.append(
                    ToolResult(
                        tool_call_id=call.tool_call_id,
                        name=call.name,
                        status="error",
                        content=[f"Unknown tool: {call.name!r}"],
                    )
                )
                continue
            try:
                result = await tool.execute(context, **call.arguments)
                results.append(
                    ToolResult(
                        tool_call_id=call.tool_call_id,
                        name=call.name,
                        status="success",
                        content=[result],
                    )
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Tool %r raised: %s", call.name, exc)
                results.append(
                    ToolResult(
                        tool_call_id=call.tool_call_id,
                        name=call.name,
                        status="error",
                        content=[f"Error: {exc}"],
                    )
                )
        return results

    def _is_final_response(self, event: Event) -> bool:
        """True when the think event signals a terminal response."""
        if self.output_tool_name is None:
            # Text mode: a Message with no tool calls in the same event
            has_message = any(isinstance(i, Message) for i in event.content)
            has_tool_call = any(isinstance(i, ToolCall) for i in event.content)
            return has_message and not has_tool_call
        # Structured mode: the output tool was called
        return any(
            isinstance(i, ToolCall) and i.name == self.output_tool_name
            for i in event.content
        )

    def _extract_final_result(self, event: Event) -> str | BaseModel:
        """Extract the final answer from a terminal think event."""
        if self.output_tool_name is None:
            for item in event.content:
                if isinstance(item, Message):
                    return item.content
            return ""
        # Structured mode: validate tool call arguments against output_type
        for item in event.content:
            if isinstance(item, ToolCall) and item.name == self.output_tool_name:
                return self.output_type.model_validate(item.arguments)  # type: ignore[union-attr]
        return ""
