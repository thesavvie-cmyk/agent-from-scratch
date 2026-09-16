"""Budget guard — hourly token/request limits with file-based state (block 9).

Limits are configured via environment variables:
  MAX_REQUESTS_PER_HOUR  (default: 500)
  MAX_TOKENS_PER_HOUR    (default: 1_000_000)

State is persisted to a JSON file so it survives process restarts.
State file path: $BUDGET_STATE_FILE or ~/.agentkit/budget.json
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


class BudgetExceededError(RuntimeError):
    """Raised when an hourly budget limit is exceeded.

    This is treated as a configuration-class error (like a bad API key): the
    caller cannot self-correct it, so it propagates uncaught through the
    agent loop instead of being swallowed by the generic exception handler.
    """


def _default_state_path() -> Path:
    path = os.getenv("BUDGET_STATE_FILE")
    if path:
        return Path(path)
    return Path.home() / ".agentkit" / "budget.json"


def _load_state(path: Path) -> dict[str, Any]:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {
        "window_start": time.time(),
        "requests": 0,
        "input_tokens": 0,
        "output_tokens": 0,
    }


def _save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state), encoding="utf-8")


class BudgetGuard:
    """Hourly budget guard with file-backed state.

    Usage::

        guard = BudgetGuard()
        guard.check()                                   # raises if limit already hit
        guard.record(input_tokens=1000, output_tokens=200)  # persist after call
    """

    WINDOW_SECONDS: int = 3600

    def __init__(
        self,
        max_requests: int | None = None,
        max_tokens: int | None = None,
        state_path: Path | None = None,
    ) -> None:
        self.max_requests: int = max_requests if max_requests is not None else int(
            os.getenv("MAX_REQUESTS_PER_HOUR", "500")
        )
        self.max_tokens: int = max_tokens if max_tokens is not None else int(
            os.getenv("MAX_TOKENS_PER_HOUR", "1000000")
        )
        self._path: Path = state_path or _default_state_path()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _get_state(self) -> dict[str, Any]:
        """Load state; reset window if it has expired."""
        state = _load_state(self._path)
        now = time.time()
        if now - state.get("window_start", 0) > self.WINDOW_SECONDS:
            state = {
                "window_start": now,
                "requests": 0,
                "input_tokens": 0,
                "output_tokens": 0,
            }
            _save_state(self._path, state)
        return state

    # ── Public API ────────────────────────────────────────────────────────────

    def check(self) -> None:
        """Check if another request is allowed.  Raises BudgetExceededError if not.

        Call BEFORE the API request.  Because BudgetExceededError is not caught
        by LlmClient's generic handler, it propagates to the caller unchanged.
        """
        state = self._get_state()
        if state["requests"] >= self.max_requests:
            raise BudgetExceededError(
                f"Request budget exceeded: {state['requests']}/{self.max_requests} "
                f"requests in current hour window."
            )
        tokens_used = state["input_tokens"] + state["output_tokens"]
        if tokens_used >= self.max_tokens:
            raise BudgetExceededError(
                f"Token budget exceeded: {tokens_used:,}/{self.max_tokens:,} tokens "
                f"in current hour window."
            )

    def record(self, input_tokens: int = 0, output_tokens: int = 0) -> None:
        """Persist token and request usage after a successful API call."""
        state = self._get_state()
        state["requests"] += 1
        state["input_tokens"] += input_tokens
        state["output_tokens"] += output_tokens
        _save_state(self._path, state)

    def status(self) -> dict[str, Any]:
        """Return current usage statistics."""
        state = self._get_state()
        now = time.time()
        window_start = state.get("window_start", now)
        resets_in = max(0.0, self.WINDOW_SECONDS - (now - window_start))
        tokens_used = state["input_tokens"] + state["output_tokens"]
        return {
            "requests": state["requests"],
            "max_requests": self.max_requests,
            "tokens_used": tokens_used,
            "input_tokens": state["input_tokens"],
            "output_tokens": state["output_tokens"],
            "max_tokens": self.max_tokens,
            "resets_in_seconds": round(resets_in),
            "window_start": window_start,
        }
