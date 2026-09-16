"""Token counting utilities (block 10).

Uses the Anthropic API's native count_tokens endpoint — NOT tiktoken, which
is the OpenAI tokeniser and gives wrong results for Claude models.

Falls back to a character-based approximation (~4 chars/token) when the API
is unavailable, emitting a warning so the caller knows counts are estimates.
"""
from __future__ import annotations

import logging
import os
import warnings

logger = logging.getLogger(__name__)


def count_tokens(text: str, model: str | None = None) -> int:
    """Count tokens in *text* for the given Claude *model*.

    Parameters
    ----------
    text:
        Text to measure.
    model:
        Claude model ID (litellm format, e.g. ``"anthropic/claude-haiku-4-5"``
        or bare ``"claude-haiku-4-5"``).  Defaults to FAST_MODEL.

    Returns
    -------
    int
        Exact token count when the Anthropic API is reachable; an
        approximation (len(text) // 4) otherwise.
    """
    from agentkit.config import FAST_MODEL

    raw_model = model or FAST_MODEL
    # Strip "anthropic/" prefix — the Anthropic SDK uses bare model IDs
    sdk_model = raw_model.removeprefix("anthropic/")

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        warnings.warn(
            "ANTHROPIC_API_KEY not set — using approximate token count (~4 chars/token)",
            stacklevel=2,
        )
        return max(1, len(text) // 4)

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.count_tokens(
            model=sdk_model,
            messages=[{"role": "user", "content": text}],
        )
        return response.input_tokens
    except Exception as exc:  # noqa: BLE001
        warnings.warn(
            f"count_tokens API call failed ({exc}) — using approximate count",
            stacklevel=2,
        )
        return max(1, len(text) // 4)
