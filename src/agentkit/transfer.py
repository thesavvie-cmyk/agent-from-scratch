"""Agent-to-agent transfer with shared context (block 21).

``TransferOrchestrator`` manages a registry of agents and routes a
conversation from one to another when ``transfer_to`` is called.

How it works
------------
1. The orchestrator injects a ``transfer_to`` tool into every registered
   agent's toolbox at construction time.
2. When an agent calls ``transfer_to(agent_name, reason)``, the tool writes
   a ``_transfer_request`` marker to the shared ExecutionContext.
3. After each agent.run() the orchestrator checks the marker:
   - if present → switch to the target agent, pass a continuation message,
     repeat;
   - if absent  → done.
4. The context is shared throughout: every agent sees the full conversation
   history, including all previous agents' messages.

Protection against ping-pong
-----------------------------
``max_transfers`` caps the total number of hand-offs.  When the limit is
reached the run stops and ``TransferResult.error`` is set.

Trace visibility
----------------
``TransferResult.transfer_log`` records every hand-off as
``{from, to, reason}``.  The shared context's events already contain the
full turn-by-turn trace with correct ``author`` fields.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from agentkit.agent import Agent, AgentResult
from agentkit.context import ExecutionContext
from agentkit.tools.base import BaseTool, FunctionTool

logger = logging.getLogger(__name__)


@dataclass
class TransferResult:
    """Result of a TransferOrchestrator run.

    Attributes
    ----------
    output:
        Final output produced by the last active agent.
    context:
        The shared ExecutionContext containing all agents' events.
    transfer_log:
        Ordered list of ``{from, to, reason}`` dicts, one per hand-off.
    error:
        Set when a transfer failed (unknown target, limit exceeded, etc.).
    """

    output: Any  # str | BaseModel
    context: ExecutionContext
    transfer_log: list[dict[str, str]] = field(default_factory=list)
    error: str | None = None


class TransferOrchestrator:
    """Manages a registry of agents and routes conversations between them.

    Parameters
    ----------
    agents:
        Mapping of agent name → Agent.  Agent names must match ``agent.name``.
    entry_agent:
        Name of the first agent to handle ``run()``.
    max_transfers:
        Maximum number of hand-offs allowed per run.  Prevents ping-pong
        loops.  Defaults to 5.
    """

    def __init__(
        self,
        agents: dict[str, Agent],
        entry_agent: str,
        max_transfers: int = 5,
    ) -> None:
        if entry_agent not in agents:
            raise ValueError(
                f"Entry agent '{entry_agent}' not found in registry. "
                f"Available: {sorted(agents)}"
            )
        self._agents = agents
        self._entry = entry_agent
        self._max_transfers = max_transfers

        # Build one shared transfer tool and inject it into every agent
        self._transfer_tool: BaseTool = self._build_transfer_tool()
        for agent in agents.values():
            agent._toolbox[self._transfer_tool.name] = self._transfer_tool

    # ── Internal ───────────────────────────────────────────────────────────────

    def _build_transfer_tool(self) -> BaseTool:
        """Return a ``transfer_to`` FunctionTool closed over this registry."""
        agent_names: set[str] = set(self._agents.keys())

        async def transfer_to(
            context: ExecutionContext,
            agent_name: str,
            reason: str,
        ) -> str:
            """Transfer the conversation to a specialist agent.

            Use when the user's request is better handled by a different agent.
            The full conversation history is preserved for the target agent.

            Parameters
            ----------
            agent_name : str
                Name of the target agent as listed in the registry.
            reason : str
                Why you are transferring — recorded in the trace.
            """
            if agent_name not in agent_names:
                return (
                    f"Error: unknown agent '{agent_name}'. "
                    f"Available: {sorted(agent_names)}"
                )
            context.state["_transfer_request"] = {
                "to": agent_name,
                "reason": reason,
            }
            return f"Transferring conversation to '{agent_name}': {reason}"

        return FunctionTool(transfer_to)

    # ── Public API ─────────────────────────────────────────────────────────────

    @property
    def transfer_tool(self) -> BaseTool:
        """The shared ``transfer_to`` tool injected into all agents."""
        return self._transfer_tool

    async def run(self, user_input: str) -> TransferResult:
        """Run the conversation, following transfers until none remain.

        Parameters
        ----------
        user_input:
            The initial message sent to the entry agent.

        Returns
        -------
        TransferResult
            Contains the final output, shared context, transfer log, and any
            error.
        """
        ctx = ExecutionContext()
        current = self._entry
        transfer_log: list[dict[str, str]] = []
        transfer_count = 0
        result: AgentResult | None = None

        # Entry agent handles the original request
        result = await self._agents[current].run(user_input, context=ctx)
        ctx = result.context

        # Follow transfer chain
        while True:
            transfer_req = ctx.state.pop("_transfer_request", None)
            if transfer_req is None:
                break

            if transfer_count >= self._max_transfers:
                logger.warning(
                    "Transfer limit (%d) reached; stopping. Last agent: %s",
                    self._max_transfers,
                    current,
                )
                return TransferResult(
                    output=result.output,
                    context=ctx,
                    transfer_log=transfer_log,
                    error=(
                        f"Transfer limit ({self._max_transfers}) exceeded. "
                        f"Last active agent: '{current}'"
                    ),
                )

            target: str = transfer_req["to"]
            reason: str = transfer_req["reason"]

            if target not in self._agents:
                return TransferResult(
                    output=result.output,
                    context=ctx,
                    transfer_log=transfer_log,
                    error=f"Transfer target '{target}' not found in registry",
                )

            transfer_log.append({"from": current, "to": target, "reason": reason})
            transfer_count += 1
            logger.debug(
                "Transfer %d/%d: %s → %s (%s)",
                transfer_count,
                self._max_transfers,
                current,
                target,
                reason,
            )

            current = target
            continuation = (
                f"[Transferred from '{transfer_log[-1]['from']}': {reason}]"
            )
            result = await self._agents[current].run(continuation, context=ctx)
            ctx = result.context

        return TransferResult(
            output=result.output,
            context=ctx,
            transfer_log=transfer_log,
        )
