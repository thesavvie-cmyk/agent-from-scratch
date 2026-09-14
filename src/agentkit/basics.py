from __future__ import annotations

import asyncio
from typing import Any

import litellm
from pydantic import BaseModel

from agentkit.config import FAST_MODEL


class StructuredOutputError(ValueError):
    """Raised when the model response cannot be validated against the target schema."""

    def __init__(self, message: str, *, raw: str) -> None:
        super().__init__(message)
        self.raw = raw


def ask(
    prompt: str,
    *,
    model: str = FAST_MODEL,
    system: str | None = None,
    temperature: float | None = None,
) -> str:
    """Single stateless LLM call. Returns the text reply."""
    messages: list[dict[str, str]] = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    kwargs: dict[str, Any] = {"model": model, "messages": messages}
    if temperature is not None:
        kwargs["temperature"] = temperature

    response = litellm.completion(**kwargs)
    return response.choices[0].message.content


class Conversation:
    """Stateful multi-turn conversation."""

    def __init__(
        self,
        *,
        system: str | None = None,
        model: str = FAST_MODEL,
    ) -> None:
        self._model = model
        self._system = system
        self._history: list[dict[str, str]] = []
        self._last_usage: Any = None

    def send(self, text: str) -> str:
        """Append a user turn, call the model, store and return the assistant reply."""
        self._history.append({"role": "user", "content": text})
        response = litellm.completion(
            model=self._model,
            messages=self._build_messages(),
        )
        self._last_usage = response.usage
        reply: str = response.choices[0].message.content
        self._history.append({"role": "assistant", "content": reply})
        return reply

    def _build_messages(self) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if self._system is not None:
            messages.append({"role": "system", "content": self._system})
        messages.extend(self._history)
        return messages

    @property
    def messages(self) -> list[dict[str, str]]:
        """Copy of conversation history, excluding the system prompt."""
        return list(self._history)

    @property
    def last_usage(self) -> Any:
        """Usage stats from the most recent send() call (None before first call)."""
        return self._last_usage

    def reset(self) -> None:
        """Clear history while keeping system prompt and model."""
        self._history = []
        self._last_usage = None


def extract(
    text: str,
    schema: type[BaseModel],
    *,
    model: str = FAST_MODEL,
) -> BaseModel:
    """Extract structured data from text, validated against a Pydantic schema."""
    response = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": text}],
        response_format=schema,
    )
    raw: str | None = response.choices[0].message.content
    if raw is None:
        raise StructuredOutputError("Model returned no content", raw="")
    try:
        return schema.model_validate_json(raw)
    except Exception as exc:
        raise StructuredOutputError(
            f"Could not validate model output against {schema.__name__}: {exc}",
            raw=raw,
        ) from exc


async def ask_many(
    prompts: list[str],
    *,
    concurrency: int = 5,
    model: str = FAST_MODEL,
    num_retries: int = 3,
) -> list[str | Exception]:
    """Call the model for each prompt concurrently with a Semaphore limit.

    Preserves prompt order. Exceptions from individual calls are returned
    in-place rather than raised, so one failure does not cancel the rest.
    """
    semaphore = asyncio.Semaphore(concurrency)

    async def _call(prompt: str) -> str:
        async with semaphore:
            response = await litellm.acompletion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                num_retries=num_retries,
            )
            return response.choices[0].message.content

    tasks = [asyncio.create_task(_call(p)) for p in prompts]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return list(results)
