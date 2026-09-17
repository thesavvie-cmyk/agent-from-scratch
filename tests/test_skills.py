"""Unit and live tests for block 19: agent skills."""
from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agentkit.skills import (
    SkillInfo,
    discover_skills,
    format_skills_for_prompt,
    load_skill,
    make_read_skill_tool,
    parse_frontmatter,
)

# ── parse_frontmatter ──────────────────────────────────────────────────────────


def test_parse_frontmatter_normal() -> None:
    content = "---\nname: my-skill\ndescription: A short desc\n---\n\nBody"
    meta = parse_frontmatter(content)
    assert meta["name"] == "my-skill"
    assert meta["description"] == "A short desc"


def test_parse_frontmatter_colon_in_description() -> None:
    """Quoted colons in description are parsed correctly.

    In valid YAML, a colon in a string value requires quoting.
    A naive split(':')[1] approach would produce "Strategy" for
    'description: "Strategy: search, then extract"' — PyYAML returns
    the full string.
    """
    content = '---\nname: test\ndescription: "Strategy: search, then extract"\n---\n'
    meta = parse_frontmatter(content)
    assert meta["description"] == "Strategy: search, then extract"


def test_parse_frontmatter_multiline_value() -> None:
    content = textwrap.dedent("""\
        ---
        name: test
        description: Short
        notes: |
          line one
          line two
        ---
        Body
    """)
    meta = parse_frontmatter(content)
    assert "line one" in meta["notes"]
    assert "line two" in meta["notes"]


def test_parse_frontmatter_no_frontmatter() -> None:
    content = "# Just a markdown doc\n\nNo frontmatter here."
    assert parse_frontmatter(content) == {}


def test_parse_frontmatter_missing_closing_fence() -> None:
    content = "---\nname: test\ndescription: something\n"
    assert parse_frontmatter(content) == {}


def test_parse_frontmatter_invalid_yaml() -> None:
    content = "---\n: this is invalid\n  - yaml:\n---\n"
    # Should return {} and emit a warning, not raise
    result = parse_frontmatter(content)
    assert isinstance(result, dict)


# ── load_skill ─────────────────────────────────────────────────────────────────


def test_load_skill_no_skill_md(tmp_path: Path) -> None:
    skill_dir = tmp_path / "empty-skill"
    skill_dir.mkdir()
    assert load_skill(skill_dir) is None


def test_load_skill_missing_required_fields(tmp_path: Path) -> None:
    skill_dir = tmp_path / "bad-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("---\nversion: '1.0'\n---\nNo name or description.")
    assert load_skill(skill_dir) is None


def test_load_skill_normal(tmp_path: Path) -> None:
    skill_dir = tmp_path / "web-research"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: web-research\ndescription: 'Strategy: search and extract'\nversion: '2.0'\n---\n\nDetails."
    )
    skill = load_skill(skill_dir)
    assert skill is not None
    assert skill.name == "web-research"
    assert skill.description == "Strategy: search and extract"
    assert skill.version == "2.0"
    assert skill.path == skill_dir


def test_load_skill_scripts_field(tmp_path: Path) -> None:
    skill_dir = tmp_path / "with-scripts"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: with-scripts\ndescription: Has scripts\nscripts:\n  - scripts/foo.py\n---\n"
    )
    skill = load_skill(skill_dir)
    assert skill is not None
    assert "scripts/foo.py" in skill.scripts


# ── discover_skills ────────────────────────────────────────────────────────────


def test_discover_skills_nested_structure(tmp_path: Path) -> None:
    """discover_skills only looks one level deep and sorts by name."""
    for name, has_skill in [("alpha", True), ("beta", False), ("gamma", True)]:
        d = tmp_path / name
        d.mkdir()
        if has_skill:
            (d / "SKILL.md").write_text(
                f"---\nname: {name}\ndescription: Desc for {name}\n---\n"
            )
    skills = discover_skills(tmp_path)
    assert [s.name for s in skills] == ["alpha", "gamma"]


def test_discover_skills_dir_without_skill_md_skipped(tmp_path: Path) -> None:
    no_skill_dir = tmp_path / "not-a-skill"
    no_skill_dir.mkdir()
    (no_skill_dir / "README.md").write_text("Just a readme.")
    assert discover_skills(tmp_path) == []


def test_discover_skills_nonexistent_root() -> None:
    assert discover_skills(Path("/nonexistent/path")) == []


def test_discover_skills_sorted_by_name(tmp_path: Path) -> None:
    for name in ["zzz", "aaa", "mmm"]:
        d = tmp_path / name
        d.mkdir()
        (d / "SKILL.md").write_text(f"---\nname: {name}\ndescription: D\n---\n")
    skills = discover_skills(tmp_path)
    assert [s.name for s in skills] == ["aaa", "mmm", "zzz"]


# ── format_skills_for_prompt ───────────────────────────────────────────────────


def test_format_skills_for_prompt_contains_all_names(tmp_path: Path) -> None:
    skills = [
        SkillInfo("web-research", "Search strategy", tmp_path),
        SkillInfo("data-extraction", "Extract data", tmp_path),
    ]
    text = format_skills_for_prompt(skills)
    assert "web-research" in text
    assert "data-extraction" in text
    assert "read_skill" in text


def test_format_skills_for_prompt_empty() -> None:
    assert format_skills_for_prompt([]) == ""


# ── make_read_skill_tool ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_read_skill_returns_content(tmp_path: Path) -> None:
    skill_dir = tmp_path / "test-skill"
    skill_dir.mkdir()
    expected = "---\nname: test-skill\ndescription: Test\n---\n\n# Guide\nDetails here."
    (skill_dir / "SKILL.md").write_text(expected)
    skill = SkillInfo("test-skill", "Test", skill_dir)

    tool = make_read_skill_tool([skill])
    from agentkit.context import ExecutionContext
    ctx = ExecutionContext()
    result = await tool.execute(ctx, name="test-skill")
    assert result == expected


@pytest.mark.asyncio
async def test_read_skill_unknown_returns_error(tmp_path: Path) -> None:
    skill = SkillInfo("real-skill", "Real", tmp_path)
    tool = make_read_skill_tool([skill])
    from agentkit.context import ExecutionContext
    ctx = ExecutionContext()
    result = await tool.execute(ctx, name="no-such-skill")
    assert "Unknown" in result or "not found" in result.lower()
    assert "real-skill" in result  # lists available skills


@pytest.mark.asyncio
async def test_read_skill_name_is_read_skill() -> None:
    tool = make_read_skill_tool([])
    assert tool.name == "read_skill"


# ── Agent integration ──────────────────────────────────────────────────────────


def test_agent_skills_dir_adds_instructions(tmp_path: Path) -> None:
    """skills_dir → skill list appears in _effective_instructions."""
    d = tmp_path / "my-skill"
    d.mkdir()
    (d / "SKILL.md").write_text("---\nname: my-skill\ndescription: Does stuff\n---\n")

    from agentkit.agent import Agent
    from agentkit.llm import LlmClient
    agent = Agent(model=MagicMock(spec=LlmClient), skills_dir=tmp_path)
    assert "my-skill" in agent._effective_instructions
    assert "read_skill" in agent._toolbox


def test_agent_skills_dir_none_no_read_skill_tool() -> None:
    from agentkit.agent import Agent
    from agentkit.llm import LlmClient
    agent = Agent(model=MagicMock(spec=LlmClient))
    assert "read_skill" not in agent._toolbox
    assert agent._skills == []


def test_agent_skills_loaded(tmp_path: Path) -> None:
    """Agent._skills contains SkillInfo objects."""
    for name in ("skill-a", "skill-b"):
        d = tmp_path / name
        d.mkdir()
        (d / "SKILL.md").write_text(f"---\nname: {name}\ndescription: D\n---\n")

    from agentkit.agent import Agent
    from agentkit.llm import LlmClient
    agent = Agent(model=MagicMock(spec=LlmClient), skills_dir=tmp_path)
    assert len(agent._skills) == 2
    names = {s.name for s in agent._skills}
    assert names == {"skill-a", "skill-b"}


def test_agent_skills_dir_str_accepted(tmp_path: Path) -> None:
    """skills_dir accepts str as well as Path."""
    d = tmp_path / "s"
    d.mkdir()
    (d / "SKILL.md").write_text("---\nname: s\ndescription: D\n---\n")

    from agentkit.agent import Agent
    from agentkit.llm import LlmClient
    agent = Agent(model=MagicMock(spec=LlmClient), skills_dir=str(tmp_path))
    assert len(agent._skills) == 1


@pytest.mark.asyncio
async def test_agent_read_skill_returns_content(tmp_path: Path) -> None:
    """Agent's read_skill tool returns the SKILL.md content."""
    d = tmp_path / "my-skill"
    d.mkdir()
    content = "---\nname: my-skill\ndescription: Helps\n---\n\n# Guide\nDo this."
    (d / "SKILL.md").write_text(content)

    from agentkit.agent import Agent
    from agentkit.context import ExecutionContext
    from agentkit.llm import LlmClient
    agent = Agent(model=MagicMock(spec=LlmClient), skills_dir=tmp_path)
    ctx = ExecutionContext()
    result = await agent._toolbox["read_skill"].execute(ctx, name="my-skill")
    assert result == content


# ── Live test ──────────────────────────────────────────────────────────────────


@pytest.mark.live
@pytest.mark.asyncio
async def test_live_agent_reads_web_research_skill() -> None:
    """Real agent reads web-research skill when asked to do structured research."""
    from agentkit.agent import Agent
    from agentkit.config import FAST_MODEL, find_uv
    from agentkit.llm import LlmClient
    from agentkit.mcp_client import McpToolset
    from agentkit.tools.mcp import load_mcp_tools

    skills_dir = Path(__file__).parent.parent / "skills"
    mcp_cmd = (find_uv(), ["run", "python", "-m", "agentkit.servers.tavily_server"])

    async with McpToolset(*mcp_cmd) as ts:
        search_tools = list(load_mcp_tools(ts))

    agent = Agent(
        model=LlmClient(FAST_MODEL),
        tools=search_tools,
        instructions="Answer research questions. Read relevant skills for guidance.",
        max_steps=10,
        skills_dir=skills_dir,
    )

    result = await agent.run(
        "List the birthplaces of the first 3 US presidents, numbered."
    )

    # The agent should have read web-research or answered correctly without it
    output = str(result.output)
    assert "Washington" in output or "Virginia" in output, f"Unexpected output: {output[:200]}"
    # At least 3 presidents should be listed
    numbered = [line for line in output.split("\n") if line.strip() and line.strip()[0].isdigit()]
    assert len(numbered) >= 3, f"Expected at least 3 entries, got: {output[:300]}"
