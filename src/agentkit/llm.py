"""LLM abstraction layer — LlmRequest, LlmResponse, LlmClient (block 7).

LlmClient wraps LiteLLM's acompletion with:
- Structured request/response types that speak ContentItem
- Error handling: transient/network errors → LlmResponse(error_message=...)
  configuration errors (bad API key, unknown model) → re-raise
- _build_messages reuses transcript.items_to_messages so the conversion
  logic lives in exactly one place
"""
from __future__ import annotations

import json
import logging
from typing import Any

import litellm
from pydantic import BaseModel, ConfigDict

from agentkit.tools.base import BaseTool
from agentkit.transcript import items_to_messages
from agentkit.types import ContentItem, Message, ToolCall

logger = logging.getLogger(__name__)

# Errors that indicate wrong configuration — the model cannot fix these,
# so we re-raise instead of wrapping in LlmResponse.
_CONFIG_ERRORS = (
    litellm.AuthenticationError,
    litellm.NotFoundError,
    litellm.PermissionDeniedError,
)


class LlmRequest(BaseModel):
    """Input to a single LLM call."""

    # arbitrary_types_allowed is required because BaseTool is not a Pydantic model.
    # Without it Pydantic would refuse to store BaseTool instances in a field.
    model_config = ConfigDict(arbitrary_types_allowed=True)

    instructions: list[str] = []
    contents: list[ContentItem] = []
    tools: list[BaseTool] = []
    tool_choice: str | None = None


class LlmResponse(BaseModel):
    """Output from a single LLM call."""

    content: list[ContentItem] = []
    error_message: str | None = None
    usage_metadata: dict[str, Any] = {}


class LlmClient:
    """Thin async wrapper around LiteLLM.

    ``generate()`` converts an LlmRequest to a LiteLLM call, parses the
    response into ContentItems, and returns an LlmResponse.
    """

    def __init__(self, model: str, **config: Any) -> None:
        self._model = model
        self._config = config

    # ── Public API ────────────────────────────────────────────────────────────

    async def generate(self, request: LlmRequest) -> LlmResponse:
        """Send *request* to the model and return a structured response.

        Transient errors (network, timeout, rate-limit) are caught and
        returned as LlmResponse(error_message=...).
        Configuration errors (bad key, unknown model) are re-raised so the
        caller can surface them immediately — the model cannot self-correct
        these.
        """
        messages = self._build_messages(request)
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            **self._config,
        }
        if request.tools:
            kwargs["tools"] = [t.tool_definition for t in request.tools]
            if request.tool_choice is not None:
                kwargs["tool_choice"] = request.tool_choice

        try:
            raw = await litellm.acompletion(**kwargs)
            return self._parse_response(raw)
        except _CONFIG_ERRORS:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.warning("LlmClient.generate error: %s", exc)
            return LlmResponse(error_message=str(exc), usage_metadata={})

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_messages(self, request: LlmRequest) -> list[dict[str, Any]]:
        """Convert LlmRequest to the list[dict] format LiteLLM expects.

        instructions → system messages (prepended)
        contents     → converted via items_to_messages (shared with transcript)
        """
        system_msgs = [{"role": "system", "content": instr} for instr in request.instructions]
        return system_msgs + items_to_messages(request.contents)

    def _parse_response(self, raw: Any) -> LlmResponse:
        """Extract ContentItems and usage from a raw LiteLLM response."""
        msg = raw.choices[0].message
        content: list[ContentItem] = []

        if msg.content:
            content.append(Message(role="assistant", content=msg.content))

        if msg.tool_calls:
            for tc in msg.tool_calls:
                content.append(
                    ToolCall(
                        tool_call_id=tc.id,
                        name=tc.function.name,
                        arguments=json.loads(tc.function.arguments),
                    )
                )

        usage = raw.usage
        usage_dict: dict[str, Any] = {
            "input_tokens": getattr(usage, "prompt_tokens", 0) or 0,
            "output_tokens": getattr(usage, "completion_tokens", 0) or 0,
        }

        return LlmResponse(content=content, usage_metadata=usage_dict)
