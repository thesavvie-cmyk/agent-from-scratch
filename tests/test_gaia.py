from __future__ import annotations

from unittest.mock import MagicMock

import litellm
import pytest
from pydantic import ValidationError

import agentkit.gaia as gaia_mod
from agentkit.gaia import GaiaOutput, evaluate_single, is_correct, solve_problem

# ── helpers ───────────────────────────────────────────────────────────────────

def _acomp_resp(content: str | None, finish_reason: str = "stop") -> MagicMock:
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.finish_reason = finish_reason
    choice.message = msg
    usage = MagicMock()
    usage.prompt_tokens = 10
    usage.completion_tokens = 5
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = usage
    return resp


# ── is_correct ────────────────────────────────────────────────────────────────

def test_is_correct_case_insensitive() -> None:
    assert is_correct("Paris", "paris")
    assert is_correct("PARIS", "Paris")


def test_is_correct_strips_whitespace() -> None:
    assert is_correct("  Paris  ", "Paris")
    assert is_correct("Paris", "  Paris  ")


def test_is_correct_none_prediction() -> None:
    assert not is_correct(None, "Paris")


def test_is_correct_mismatch() -> None:
    assert not is_correct("London", "Paris")


# ── GaiaOutput ────────────────────────────────────────────────────────────────

def test_gaia_output_defaults() -> None:
    out = GaiaOutput(is_solvable=True)
    assert out.unsolvable_reason == ""
    assert out.final_answer == ""


def test_gaia_output_invalid_json_raises() -> None:
    with pytest.raises(ValidationError):
        GaiaOutput.model_validate_json("not json at all")


# ── solve_problem ─────────────────────────────────────────────────────────────

async def test_solve_problem_refusal(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_acompletion(**kwargs: object) -> MagicMock:
        return _acomp_resp(content=None, finish_reason="refusal")

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    result = await solve_problem("anthropic/claude-haiku-4-5", "some question")

    assert result.is_solvable is False
    assert result.unsolvable_reason  # non-empty explanation


# ── evaluate_single ───────────────────────────────────────────────────────────

async def test_evaluate_single_catches_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def failing_raw_solve(*args: object, **kwargs: object) -> None:
        raise RuntimeError("API down")

    monkeypatch.setattr(gaia_mod, "_raw_solve", failing_raw_solve)

    problem = {"task_id": "t1", "Question": "What is 2+2?", "Final answer": "4"}
    result = await evaluate_single(problem, "anthropic/claude-haiku-4-5")

    assert isinstance(result, dict)
    assert result["task_id"] == "t1"
    assert result["error"] is not None
    assert "RuntimeError" in result["error"]


# ── live tests ────────────────────────────────────────────────────────────────

@pytest.mark.live
def test_load_level1_returns_problems() -> None:
    from agentkit.gaia import load_level1

    problems = load_level1(limit=3)
    assert len(problems) == 3
    first = problems[0]
    assert "Question" in first
    assert "Final answer" in first
    assert "task_id" in first


@pytest.mark.live
async def test_solve_problem_live() -> None:
    from agentkit.config import FAST_MODEL

    result = await solve_problem(FAST_MODEL, "What is the capital of France?")
    assert isinstance(result, GaiaOutput)
    assert isinstance(result.is_solvable, bool)
    if result.is_solvable:
        assert result.final_answer  # non-empty when solvable
