"""Unit tests for planning_critic and Agent(planning="critiqued")."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentkit.planning import Plan, Task
from agentkit.planning_critic import (
    _parse_revised_steps,
    _plan_to_numbered_list,
    run_critique_rounds,
)


# ── _parse_revised_steps ──────────────────────────────────────────────────────


class TestParseRevisedSteps:
    def test_plain_json_array(self):
        text = '["step one", "step two", "step three"]'
        assert _parse_revised_steps(text) == ["step one", "step two", "step three"]

    def test_with_markdown_fences(self):
        text = "```json\n[\"a\", \"b\"]\n```"
        assert _parse_revised_steps(text) == ["a", "b"]

    def test_with_plain_fences(self):
        text = "```\n[\"x\", \"y\"]\n```"
        assert _parse_revised_steps(text) == ["x", "y"]

    def test_embedded_in_text(self):
        text = 'Here is the revised plan: ["do this", "then that"] Done.'
        result = _parse_revised_steps(text)
        assert result == ["do this", "then that"]

    def test_empty_strings_filtered(self):
        text = '["step one", "", "step three"]'
        result = _parse_revised_steps(text)
        assert result == ["step one", "step three"]

    def test_invalid_json_returns_none(self):
        assert _parse_revised_steps("not json at all") is None

    def test_object_not_array_returns_none(self):
        # Pure object with no embedded array at top-level returns None
        assert _parse_revised_steps('{"key": "value"}') is None

    def test_mixed_types_returns_none(self):
        assert _parse_revised_steps('[1, "two", 3]') is None


# ── _plan_to_numbered_list ────────────────────────────────────────────────────


def test_plan_to_numbered_list():
    plan = Plan(tasks=[
        Task(id="t1", description="Search"),
        Task(id="t2", description="Calculate"),
    ])
    result = _plan_to_numbered_list(plan)
    assert result == "1. Search\n2. Calculate"


# ── run_critique_rounds ───────────────────────────────────────────────────────


@pytest.fixture
def simple_plan() -> Plan:
    return Plan(tasks=[
        Task(id="t1", description="Search for data"),
        Task(id="t2", description="Report result"),
    ])


def _make_mock_llm(critic_response: str, revisor_response: str):
    """Build an LlmClient mock that returns different responses for critic/revisor."""
    from agentkit.types import Message

    call_count = 0

    async def fake_generate(request):
        nonlocal call_count
        call_count += 1
        resp = MagicMock()
        resp.error_message = None
        # Odd calls = critic, even calls = revisor
        text = critic_response if call_count % 2 == 1 else revisor_response
        msg = Message(role="assistant", content=text)
        resp.content = [msg]
        resp.usage_metadata = {}
        return resp

    llm = MagicMock()
    llm.generate = fake_generate
    return llm


@pytest.mark.asyncio
async def test_run_critique_rounds_revises_plan(simple_plan):
    llm = _make_mock_llm(
        critic_response="Missing a verification step.",
        revisor_response='["Search for data", "Verify data", "Report result"]',
    )
    result = await run_critique_rounds(llm, "Solve the task", simple_plan, rounds=1)
    assert len(result.tasks) == 3
    assert result.tasks[1].description == "Verify data"
    assert result.tasks[0].id == "t1"
    assert result.tasks[2].id == "t3"


@pytest.mark.asyncio
async def test_run_critique_rounds_exits_early_on_no_issues(simple_plan):
    """If the critic says 'No issues', stop immediately without calling the revisor."""
    call_count = 0

    from agentkit.types import Message

    async def fake_generate(request):
        nonlocal call_count
        call_count += 1
        resp = MagicMock()
        resp.error_message = None
        resp.content = [Message(role="assistant", content="No issues.")]
        resp.usage_metadata = {}
        return resp

    llm = MagicMock()
    llm.generate = fake_generate

    result = await run_critique_rounds(llm, "task", simple_plan, rounds=2)
    # Only 1 call (critic said no issues → exit early, no revisor call)
    assert call_count == 1
    assert result is simple_plan  # unchanged


@pytest.mark.asyncio
async def test_run_critique_rounds_returns_original_on_parse_failure(simple_plan):
    """If revisor returns garbage, keep original plan."""
    llm = _make_mock_llm(
        critic_response="Missing step.",
        revisor_response="I cannot rewrite this plan.",
    )
    result = await run_critique_rounds(llm, "task", simple_plan, rounds=1)
    assert result is simple_plan


@pytest.mark.asyncio
async def test_run_critique_rounds_2_rounds(simple_plan):
    """Two full rounds — revisor called twice."""
    call_count = 0

    from agentkit.types import Message

    async def fake_generate(request):
        nonlocal call_count
        call_count += 1
        resp = MagicMock()
        resp.error_message = None
        if call_count % 2 == 1:  # critic
            resp.content = [Message(role="assistant", content="Add more steps.")]
        else:  # revisor
            resp.content = [Message(role="assistant", content='["a", "b", "c"]')]
        resp.usage_metadata = {}
        return resp

    llm = MagicMock()
    llm.generate = fake_generate

    result = await run_critique_rounds(llm, "task", simple_plan, rounds=2)
    assert call_count == 4  # critic + revisor × 2
    assert len(result.tasks) == 3


# ── Agent integration ─────────────────────────────────────────────────────────


class TestAgentCritiquedPlanning:
    """Verify planning="critiqued" wires up correctly without real LLM calls."""

    def _make_agent(self, planning, critique_rounds=2):
        from agentkit.agent import Agent
        from agentkit.llm import LlmClient

        model = MagicMock(spec=LlmClient)
        return Agent(
            model=model,
            planning=planning,
            critique_rounds=critique_rounds,
        )

    def test_critiqued_sets_critique_rounds(self):
        agent = self._make_agent("critiqued", critique_rounds=1)
        assert agent._critique_rounds == 1

    def test_planning_true_does_not_set_critique_rounds(self):
        agent = self._make_agent(True)
        assert agent._critique_rounds == 0

    def test_planning_false_does_not_set_critique_rounds(self):
        agent = self._make_agent(False)
        assert agent._critique_rounds == 0

    def test_critiqued_has_planning_tools(self):
        agent = self._make_agent("critiqued")
        assert "create_plan" in agent._toolbox
        assert "update_task" in agent._toolbox
        assert "get_plan" in agent._toolbox

    def test_extract_task_from_context(self):
        from agentkit.agent import Agent
        from agentkit.context import ExecutionContext
        from agentkit.llm import LlmClient

        model = MagicMock(spec=LlmClient)
        agent = Agent(model=model, planning="critiqued")
        ctx = ExecutionContext()
        ctx.add_message("user", "What is the capital of France?", author="agent")
        task = agent._extract_task_from_context(ctx)
        assert "France" in task

    def test_extract_task_fallback(self):
        from agentkit.agent import Agent
        from agentkit.context import ExecutionContext
        from agentkit.llm import LlmClient

        model = MagicMock(spec=LlmClient)
        agent = Agent(model=model, planning="critiqued")
        ctx = ExecutionContext()
        task = agent._extract_task_from_context(ctx)
        assert task == "Complete the task."

    @pytest.mark.asyncio
    async def test_critique_plan_updates_context_state(self):
        from agentkit.agent import Agent
        from agentkit.context import ExecutionContext
        from agentkit.llm import LlmClient
        from agentkit.planning import Plan, Task

        model = MagicMock(spec=LlmClient)
        agent = Agent(model=model, planning="critiqued", critique_rounds=1)

        ctx = ExecutionContext()
        original_plan = Plan(tasks=[Task(id="t1", description="Search")])
        ctx.state["plan"] = original_plan.model_dump()

        revised_plan = Plan(tasks=[
            Task(id="t1", description="Search"),
            Task(id="t2", description="Verify"),
        ])

        with patch(
            "agentkit.planning_critic.run_critique_rounds",
            new=AsyncMock(return_value=revised_plan),
        ):
            await agent._critique_plan(ctx, "test task")

        stored = Plan.model_validate(ctx.state["plan"])
        assert len(stored.tasks) == 2
        assert ctx.state.get("plan_critique_applied") == 1

    @pytest.mark.asyncio
    async def test_critique_plan_skips_when_no_plan(self):
        from agentkit.agent import Agent
        from agentkit.context import ExecutionContext
        from agentkit.llm import LlmClient

        model = MagicMock(spec=LlmClient)
        agent = Agent(model=model, planning="critiqued", critique_rounds=1)
        ctx = ExecutionContext()  # no plan in state

        with patch("agentkit.planning_critic.run_critique_rounds") as mock_rr:
            await agent._critique_plan(ctx, "task")
            mock_rr.assert_not_called()


# ── Maintenance helpers ───────────────────────────────────────────────────────


class TestMaintenanceHelpers:
    def test_tool_error_counts(self):
        from agentkit.maintenance import _tool_error_counts

        spans = [
            {"name": "tool.execute", "attributes": {"tool.name": "web_search", "tool.status": "error"}},
            {"name": "tool.execute", "attributes": {"tool.name": "web_search", "tool.status": "error"}},
            {"name": "tool.execute", "attributes": {"tool.name": "calculator", "tool.status": "success"}},
            {"name": "agent.run", "attributes": {}},
        ]
        counts = _tool_error_counts(spans)
        assert counts == {"web_search": 2}

    def test_hit_max_count(self):
        from agentkit.maintenance import _hit_max_count

        spans = [
            {"name": "agent.run", "attributes": {"agent.steps_used": 10, "agent.max_steps": 10}},
            {"name": "agent.run", "attributes": {"agent.steps_used": 5, "agent.max_steps": 10}},
            {"name": "agent.run", "attributes": {"agent.steps_used": 10, "agent.max_steps": 10}},
        ]
        assert _hit_max_count(spans) == 2

    def test_make_case_drafts_deduplicates(self):
        from agentkit.maintenance import _make_case_drafts

        records = [
            {"task_id": "t1", "question": "Q1?", "output": "A", "verdict": "FAIL", "gold": "X"},
            {"task_id": "t1", "question": "Q1?", "output": "B", "verdict": "FAIL", "gold": "X"},  # dup Q
            {"task_id": "t2", "question": "Q2?", "output": "C", "verdict": "FAIL", "gold": "Y"},
        ]
        drafts = _make_case_drafts(records)
        assert len(drafts) == 2
        questions = {d["question"] for d in drafts}
        assert questions == {"Q1?", "Q2?"}

    def test_extract_fail_cases(self):
        from agentkit.maintenance import _extract_fail_cases

        records = [
            {"verdict": "PASS"},
            {"verdict": "FAIL"},
            {"verdict": "SKIP"},
            {"verdict": "FAIL"},
        ]
        fails = _extract_fail_cases(records)
        assert len(fails) == 2
