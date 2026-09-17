"""Agent class — ReAct loop with structured output support (block 8)."""
from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from agentkit.context import ExecutionContext
from agentkit.llm import LlmClient, LlmRequest, LlmResponse
from agentkit.schema import build_tool_definition
from agentkit.tools.base import BaseTool, FunctionTool
from agentkit.types import Event, Message, ToolCall, ToolResult

if TYPE_CHECKING:
    from agentkit.callbacks import Callbacks
    from agentkit.memory.budget import ContextBudget
    from agentkit.memory.longterm import LongTermMemory
    from agentkit.memory.session import Session, SessionStore

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
        callbacks: Callbacks | None = None,
        context_budget: ContextBudget | None = None,
        session_store: SessionStore | None = None,
        memory: LongTermMemory | None = None,
        user_id: str = "default",
    ) -> None:
        self.model = model
        self.instructions = instructions
        self.max_steps = max_steps
        self.name = name
        self.output_type = output_type
        self._callbacks = callbacks
        self._context_budget = context_budget
        self._session_store = session_store
        self._memory = memory
        self._user_id = user_id
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
        session: Session | None = None,
    ) -> AgentResult:
        """Run the agent on *user_input* and return the final result.

        When *session* is provided the previous event history is loaded into
        the context before the new turn, and new events are persisted back to
        the store at the end of the run.
        """
        if session is not None and context is None:
            ctx = self._load_session_contents(session)
        else:
            ctx = context or ExecutionContext()

        if session is not None:
            # Let tools write to session.state directly through context.state
            ctx.state["session_state"] = session.state

        # Inject relevant long-term memories into context BEFORE first LLM call.
        # Stored in context.state so _prepare_llm_request picks it up once.
        if self._memory is not None:
            await self._inject_memories(ctx, user_input)

        # Remember the event count before this turn so we can capture new events
        events_before = len(ctx.events)

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
            result = AgentResult(output="", error=run_error, context=ctx)
        else:
            output: str | BaseModel = (
                final_result
                if final_result is not None
                else f"[max_steps={self.max_steps} reached without final answer]"
            )
            ctx.final_result = output
            result = AgentResult(output=output, context=ctx)

        # Fire-and-forget: extract memories from this run and persist them.
        # We return result first, then the extraction task runs when the event
        # loop is next idle.  asyncio.create_task() is used (not ensure_future)
        # so the task is associated with the running loop and gets a proper
        # exception traceback if it fails.  In CLI usage the loop stays alive
        # for the next user prompt, giving the task time to complete.
        if self._memory is not None and not result.error:
            import asyncio
            new_events = ctx.events[events_before:]
            asyncio.create_task(  # noqa: RUF006
                self._save_memories(new_events)
            )

        # Persist new events and state to the session store
        if session is not None and self._session_store is not None:
            new_events = ctx.events[events_before:]
            if new_events:
                try:
                    self._session_store.append_events(session.session_id, new_events)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Failed to persist session events: %s", exc)
            state_patch = dict(session.state)
            if state_patch:
                try:
                    self._session_store.update_state(session.session_id, state_patch)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Failed to persist session state: %s", exc)

        return result

    def _load_session_contents(self, session: Session) -> ExecutionContext:
        """Build an ExecutionContext pre-loaded with this session's event history."""
        ctx = ExecutionContext()
        for evt in session.events:
            ctx.add_event(evt)
        ctx.state.update(session.state)
        return ctx

    # ── Loop internals ─────────────────────────────────────────────────────────

    async def _step(self, context: ExecutionContext) -> str | BaseModel | None:
        """One think→act cycle.  Returns the final result or None to continue."""
        request = await self._prepare_llm_request(context)

        if self._callbacks:
            for cb in self._callbacks.before_model:
                await self._invoke_callback(cb, context, request)

        response = await self.think(request)

        if self._callbacks:
            for cb in self._callbacks.after_model:
                await self._invoke_callback(cb, context, response)

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

    async def _inject_memories(self, context: ExecutionContext, query: str) -> None:
        """Search long-term memory and store results in context.state."""
        assert self._memory is not None
        try:
            memories = self._memory.search(
                user_id=self._user_id,
                query=query,
                top_k=5,
                min_score=0.5,
            )
            if memories:
                block = "What is known about the user:\n" + "\n".join(
                    f"- {m.text}" for m in memories
                )
                context.state["memory_context"] = block
                logger.debug(
                    "Injected %d memories for user %s", len(memories), self._user_id
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Memory injection failed: %s", exc)

    async def _save_memories(self, events: list[Event]) -> None:
        """Extract facts from *events* and persist to long-term memory."""
        assert self._memory is not None
        try:
            from agentkit.memory.extraction import extract_memories

            existing = [m.text for m in self._memory.list_all(self._user_id)[:50]]
            facts = await extract_memories(self.model, events, existing=existing)
            for fact in facts:
                self._memory.add(
                    user_id=self._user_id,
                    text=fact,
                    source_session_id=events[0].execution_id if events else None,
                )
            if facts:
                logger.debug(
                    "Saved %d memory facts for user %s", len(facts), self._user_id
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Memory save failed: %s", exc)

    async def _prepare_llm_request(self, context: ExecutionContext) -> LlmRequest:
        instructions = [self.instructions] if self.instructions else []

        # Append memory context block if injected (added once at run() start)
        memory_ctx = context.state.get("memory_context")
        if memory_ctx:
            instructions = list(instructions) + [memory_ctx]

        contents = list(context.iter_content())

        if self._context_budget is not None:
            contents, report = await self._context_budget.fit(contents, context)
            if report["applied"]:
                log = context.state.setdefault("compaction_log", [])
                log.append(
                    {
                        "step": context.current_step,
                        "original_tokens": report["original_tokens"],
                        "final_tokens": report["final_tokens"],
                        "applied": report["applied"],
                    }
                )

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
        from agentkit.callbacks import SkipTool

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

            # ── before_tool callbacks ──────────────────────────────────────────
            args: dict[str, Any] = dict(call.arguments)
            skip: SkipTool | None = None
            if self._callbacks:
                for cb in self._callbacks.before_tool:
                    rv = await self._invoke_callback(cb, context, call.name, args)
                    if isinstance(rv, SkipTool):
                        skip = rv
                        break
                    if isinstance(rv, dict):
                        args = rv

            if skip is not None:
                results.append(
                    ToolResult(
                        tool_call_id=call.tool_call_id,
                        name=call.name,
                        status="error",
                        content=[f"Tool skipped: {skip.reason}"],
                    )
                )
                continue

            # ── execute ────────────────────────────────────────────────────────
            try:
                raw = await tool.execute(context, **args)
                tool_result = ToolResult(
                    tool_call_id=call.tool_call_id,
                    name=call.name,
                    status="success",
                    content=[raw],
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Tool %r raised: %s", call.name, exc)
                tool_result = ToolResult(
                    tool_call_id=call.tool_call_id,
                    name=call.name,
                    status="error",
                    content=[f"Error: {exc}"],
                )

            # ── after_tool callbacks ───────────────────────────────────────────
            if self._callbacks:
                for cb in self._callbacks.after_tool:
                    rv = await self._invoke_callback(cb, context, call.name, tool_result)
                    if isinstance(rv, ToolResult):
                        tool_result = rv

            results.append(tool_result)
        return results

    async def _invoke_callback(self, cb: Any, *args: Any) -> Any:
        """Call a sync or async callback; log and return None on exception."""
        try:
            rv = cb(*args)
            if inspect.isawaitable(rv):
                rv = await rv
            return rv
        except Exception as exc:  # noqa: BLE001
            name = getattr(cb, "__name__", repr(cb))
            logger.warning("Callback %r raised: %s", name, exc)

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
