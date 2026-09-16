"""Tests for budget guard (block 9)."""
from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import patch

import pytest

from agentkit.budget import BudgetExceededError, BudgetGuard

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def state_file(tmp_path: Path) -> Path:
    return tmp_path / "budget.json"


@pytest.fixture()
def guard(state_file: Path) -> BudgetGuard:
    return BudgetGuard(max_requests=5, max_tokens=1000, state_path=state_file)


# ── Basic limits ──────────────────────────────────────────────────────────────


def test_check_passes_initially(guard: BudgetGuard) -> None:
    guard.check()  # no exception


def test_record_increments_usage(guard: BudgetGuard) -> None:
    guard.record(input_tokens=100, output_tokens=50)
    s = guard.status()
    assert s["requests"] == 1
    assert s["input_tokens"] == 100
    assert s["output_tokens"] == 50
    assert s["tokens_used"] == 150


def test_request_limit_exceeded(guard: BudgetGuard) -> None:
    for _ in range(5):
        guard.record()
    with pytest.raises(BudgetExceededError, match="Request budget exceeded"):
        guard.check()


def test_token_limit_exceeded(guard: BudgetGuard) -> None:
    guard.record(input_tokens=600, output_tokens=500)  # 1100 > max_tokens=1000
    with pytest.raises(BudgetExceededError, match="Token budget exceeded"):
        guard.check()


def test_exactly_at_request_limit_raises(guard: BudgetGuard) -> None:
    # 5 requests → at limit → next check should raise
    for _ in range(5):
        guard.record()
    with pytest.raises(BudgetExceededError):
        guard.check()


# ── State survives object recreation ─────────────────────────────────────────


def test_state_persists_across_instances(state_file: Path) -> None:
    g1 = BudgetGuard(max_requests=10, max_tokens=10000, state_path=state_file)
    g1.record(input_tokens=200, output_tokens=100)
    g1.record(input_tokens=300, output_tokens=150)

    # Re-create a new object pointing to the same file
    g2 = BudgetGuard(max_requests=10, max_tokens=10000, state_path=state_file)
    s = g2.status()
    assert s["requests"] == 2
    assert s["input_tokens"] == 500
    assert s["output_tokens"] == 250


# ── Window reset ──────────────────────────────────────────────────────────────


def test_window_resets_after_hour(state_file: Path) -> None:
    guard = BudgetGuard(max_requests=2, max_tokens=100, state_path=state_file)
    guard.record(input_tokens=50)
    guard.record(input_tokens=60)  # 110 tokens — over limit

    with pytest.raises(BudgetExceededError):
        guard.check()

    # Advance time by more than one hour
    future = time.time() + 3700
    with patch("agentkit.budget.time") as mock_time:
        mock_time.time.return_value = future
        # _get_state sees expired window → resets
        guard2 = BudgetGuard(max_requests=2, max_tokens=100, state_path=state_file)
        guard2.check()  # should not raise after window reset


def test_status_resets_in_decreases(guard: BudgetGuard) -> None:
    s = guard.status()
    assert 0 <= s["resets_in_seconds"] <= 3600


# ── Integration with LlmClient ────────────────────────────────────────────────


def test_llm_client_raises_budget_exceeded(state_file: Path) -> None:
    """BudgetExceededError propagates through LlmClient (not swallowed)."""
    from agentkit.budget import BudgetGuard
    from agentkit.llm import LlmClient

    guard = BudgetGuard(max_requests=0, max_tokens=0, state_path=state_file)
    client = LlmClient("anthropic/claude-haiku-4-5", budget_guard=guard)

    import asyncio

    from agentkit.llm import LlmRequest

    with pytest.raises(BudgetExceededError):
        asyncio.run(client.generate(LlmRequest()))
