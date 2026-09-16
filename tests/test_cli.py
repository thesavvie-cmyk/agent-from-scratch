"""Tests for CLI argument parsing and --json output (block 9)."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentkit.cli import _build_parser

# ── Argument parsing ──────────────────────────────────────────────────────────


def test_ask_parses_question() -> None:
    parser = _build_parser()
    args = parser.parse_args(["ask", "What is 2+2?"])
    assert args.command == "ask"
    assert args.question == "What is 2+2?"
    assert args.no_tools is False
    assert args.max_steps == 8
    assert args.json_out is False
    assert args.trace is False


def test_ask_flags() -> None:
    parser = _build_parser()
    args = parser.parse_args([
        "ask", "hello",
        "--model", "anthropic/claude-sonnet-5",
        "--max-steps", "3",
        "--no-tools",
        "--trace",
        "--json",
    ])
    assert args.model == "anthropic/claude-sonnet-5"
    assert args.max_steps == 3
    assert args.no_tools is True
    assert args.trace is True
    assert args.json_out is True


def test_chat_parses_flags() -> None:
    parser = _build_parser()
    args = parser.parse_args(["chat", "--no-tools", "--json"])
    assert args.command == "chat"
    assert args.no_tools is True
    assert args.json_out is True


def test_doctor_parses() -> None:
    parser = _build_parser()
    args = parser.parse_args(["doctor"])
    assert args.command == "doctor"
    assert args.json_out is False


def test_doctor_json_flag() -> None:
    parser = _build_parser()
    args = parser.parse_args(["doctor", "--json"])
    assert args.json_out is True


def test_budget_parses() -> None:
    parser = _build_parser()
    args = parser.parse_args(["budget"])
    assert args.command == "budget"


def test_missing_subcommand_exits() -> None:
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


# ── cmd_ask --json output ─────────────────────────────────────────────────────


@pytest.fixture()
def _mock_agent_result():
    from agentkit.agent import AgentResult
    from agentkit.context import ExecutionContext

    ctx = ExecutionContext()
    ctx.state["token_usage"] = {"input_tokens": 100, "output_tokens": 20}
    ctx._current_step = 2
    return AgentResult(output="42", context=ctx, error=None)


async def test_cmd_ask_json_output(capsys, _mock_agent_result, tmp_path) -> None:
    """--json flag produces valid JSON with expected keys."""
    from agentkit.cli import cmd_ask

    parser = _build_parser()
    args = parser.parse_args(["ask", "What is 2+2?", "--no-tools", "--json"])

    with (
        patch("agentkit.cli._make_agent") as mock_make,
    ):
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=_mock_agent_result)
        mock_make.return_value = mock_agent

        exit_code = await cmd_ask(args)

    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["output"] == "42"
    assert "steps" in data
    assert "input_tokens" in data
    assert "elapsed_s" in data
    assert data["error"] is None
    assert exit_code == 0


async def test_cmd_ask_json_error_output(capsys, tmp_path) -> None:
    """--json flag with error: exit code 1, error field set."""
    from agentkit.agent import AgentResult
    from agentkit.cli import cmd_ask
    from agentkit.context import ExecutionContext

    ctx = ExecutionContext()
    ctx.state["token_usage"] = {}
    error_result = AgentResult(output="", context=ctx, error="connection failed")

    parser = _build_parser()
    args = parser.parse_args(["ask", "hello", "--no-tools", "--json"])

    with patch("agentkit.cli._make_agent") as mock_make:
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=error_result)
        mock_make.return_value = mock_agent

        exit_code = await cmd_ask(args)

    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["error"] == "connection failed"
    assert exit_code == 1


# ── budget command ────────────────────────────────────────────────────────────


def test_cmd_budget_json(capsys, tmp_path) -> None:
    from agentkit.budget import BudgetGuard
    from agentkit.cli import cmd_budget

    parser = _build_parser()
    args = parser.parse_args(["budget", "--json"])

    state_file = tmp_path / "budget.json"
    guard = BudgetGuard(state_path=state_file)
    guard.record(input_tokens=50, output_tokens=10)

    with patch("agentkit.cli.BudgetGuard", return_value=guard):
        cmd_budget(args)

    data = json.loads(capsys.readouterr().out)
    assert "requests" in data
    assert "max_requests" in data
    assert data["requests"] == 1
