from __future__ import annotations

import json
from unittest.mock import MagicMock

import litellm
import pytest

from agentkit.tools_manual import calculator, execute_tool_call, run_with_tools

# ── helpers ───────────────────────────────────────────────────────────────────

def _make_tc(call_id: str, name: str, arguments: str) -> MagicMock:
    fn = MagicMock()
    fn.name = name
    fn.arguments = arguments
    tc = MagicMock()
    tc.id = call_id
    tc.function = fn
    return tc


def _tool_resp(call_id: str, name: str, args: dict) -> MagicMock:
    msg = MagicMock()
    msg.content = None
    msg.tool_calls = [_make_tc(call_id, name, json.dumps(args))]
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _final_resp(content: str) -> MagicMock:
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = None
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


# ── calculator ────────────────────────────────────────────────────────────────

def test_calculator_add() -> None:
    assert calculator("add", 3, 4) == 7.0


def test_calculator_subtract() -> None:
    assert calculator("subtract", 10, 3) == 7.0


def test_calculator_multiply() -> None:
    assert calculator("multiply", 6, 7) == 42.0


def test_calculator_divide() -> None:
    assert calculator("divide", 10, 2) == 5.0


def test_calculator_divide_by_zero() -> None:
    with pytest.raises(ValueError, match="zero"):
        calculator("divide", 5, 0)


def test_calculator_unknown_operator() -> None:
    with pytest.raises(ValueError, match="Unknown"):
        calculator("power", 2, 3)


# ── execute_tool_call ─────────────────────────────────────────────────────────

def test_execute_tool_call_success() -> None:
    tc = _make_tc("c1", "calculator", '{"operator":"multiply","first_number":3,"second_number":4}')
    assert float(execute_tool_call(tc)) == 12.0


def test_execute_tool_call_unknown_tool() -> None:
    tc = _make_tc("c1", "unknown_tool", "{}")
    result = execute_tool_call(tc)
    assert result.startswith("Error:")
    assert "unknown" in result.lower()


def test_execute_tool_call_invalid_json() -> None:
    tc = _make_tc("c1", "calculator", "not valid json {{")
    result = execute_tool_call(tc)
    assert result.startswith("Error:")


def test_execute_tool_call_inner_exception() -> None:
    tc = _make_tc("c1", "calculator", '{"operator":"divide","first_number":5,"second_number":0}')
    result = execute_tool_call(tc)
    assert result.startswith("Error:")
    assert "zero" in result.lower()


# ── run_with_tools ────────────────────────────────────────────────────────────

def test_run_with_tools_single_call(monkeypatch: pytest.MonkeyPatch) -> None:
    call_n = 0

    def fake_completion(**kwargs: object) -> MagicMock:
        nonlocal call_n
        call_n += 1
        if call_n == 1:
            return _tool_resp("call_001", "calculator", {"operator": "multiply", "first_number": 3, "second_number": 4})
        return _final_resp("The answer is 12.")

    monkeypatch.setattr(litellm, "completion", fake_completion)

    answer, messages = run_with_tools("3 * 4?")

    assert answer == "The answer is 12."
    # user → assistant(tool_calls) → tool → assistant(final)
    assert len(messages) == 4
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"
    assert messages[1]["tool_calls"][0]["id"] == "call_001"
    assert messages[2]["role"] == "tool"
    assert messages[2]["tool_call_id"] == "call_001"
    assert messages[3]["role"] == "assistant"
    assert messages[3]["content"] == "The answer is 12."


def test_run_with_tools_max_rounds(monkeypatch: pytest.MonkeyPatch) -> None:
    def always_tool(**kwargs: object) -> MagicMock:
        return _tool_resp("call_loop", "calculator", {"operator": "add", "first_number": 1, "second_number": 2})

    monkeypatch.setattr(litellm, "completion", always_tool)

    with pytest.raises(RuntimeError, match="max_rounds"):
        run_with_tools("loop forever", max_rounds=3)


# ── live ──────────────────────────────────────────────────────────────────────

@pytest.mark.live
def test_run_with_tools_multiply_live() -> None:
    answer, _ = run_with_tools("What is 1234 * 5678?")
    assert "7006652" in answer.replace(",", "")


@pytest.mark.live
def test_run_with_tools_no_tool_for_factual() -> None:
    answer, messages = run_with_tools("What is the capital of South Korea?")
    tool_msgs = [m for m in messages if m["role"] == "tool"]
    assert len(tool_msgs) == 0
    assert "Seoul" in answer
