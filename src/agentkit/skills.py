"""Agent skills: hierarchical capability documents (block 19).

Three-level access pattern
--------------------------
Level 1 — always in system prompt: compact list (name + one-line description).
Level 2 — on demand via read_skill tool: full SKILL.md when the task matches.
Level 3 — execution: scripts in the skill folder are uploaded to the sandbox
           and run with run_command.

Skill layout on disk
--------------------
    skills/
        web-research/
            SKILL.md          ← frontmatter + guidance
            scripts/          ← optional Python scripts
                helper.py
        data-extraction/
            SKILL.md

SKILL.md frontmatter format
---------------------------
    ---
    name: web-research
    description: One-line summary (may contain colons — parsed by PyYAML)
    version: "1.0"
    ---

    # Full guidance follows …

PyYAML is used for parsing so colons in descriptions and multiline values
work correctly (unlike a naive split(':') approach).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from agentkit.context import ExecutionContext

if TYPE_CHECKING:
    from agentkit.tools.base import BaseTool

logger = logging.getLogger(__name__)

# ── Data types ─────────────────────────────────────────────────────────────────


@dataclass
class SkillInfo:
    """Metadata extracted from a skill's YAML frontmatter."""

    name: str
    description: str
    path: Path
    version: str = "1.0"
    scripts: list[str] = field(default_factory=list)

    def skill_file(self) -> Path:
        return self.path / "SKILL.md"

    def scripts_dir(self) -> Path:
        return self.path / "scripts"

    def script_paths(self) -> list[Path]:
        d = self.scripts_dir()
        if not d.exists():
            return []
        return sorted(d.glob("*.py"))


# ── Parsing ────────────────────────────────────────────────────────────────────


def parse_frontmatter(content: str) -> dict[str, Any]:
    """Extract and parse the YAML frontmatter block from *content*.

    Expected format::

        ---
        name: my-skill
        description: A description with: colons and other special chars
        ---

        Body text …

    Returns the parsed dict, or an empty dict when:

    - the file does not start with ``---``
    - the closing ``---`` is missing
    - the YAML is invalid (a warning is emitted)
    """
    lines = content.splitlines(keepends=True)
    if not lines or lines[0].rstrip("\r\n") != "---":
        return {}

    end = None
    for i, line in enumerate(lines[1:], start=1):
        if line.rstrip("\r\n") == "---":
            end = i
            break
    if end is None:
        return {}

    yaml_src = "".join(lines[1:end])
    try:
        parsed = yaml.safe_load(yaml_src)
    except yaml.YAMLError as exc:
        logger.warning("Invalid YAML frontmatter: %s", exc)
        return {}
    return parsed if isinstance(parsed, dict) else {}


# ── Loaders ────────────────────────────────────────────────────────────────────


def load_skill(skill_dir: Path) -> SkillInfo | None:
    """Load a skill from *skill_dir*.

    Returns ``None`` when:
    - ``SKILL.md`` does not exist in *skill_dir*
    - the frontmatter is missing the required ``name`` or ``description`` keys
    """
    skill_file = skill_dir / "SKILL.md"
    if not skill_file.exists():
        return None

    content = skill_file.read_text(encoding="utf-8")
    meta = parse_frontmatter(content)

    name = meta.get("name")
    description = meta.get("description")
    if not name or not description:
        logger.warning(
            "Skill at %s missing required frontmatter fields (name, description) — skipping",
            skill_dir,
        )
        return None

    return SkillInfo(
        name=str(name),
        description=str(description),
        path=skill_dir,
        version=str(meta.get("version", "1.0")),
        scripts=list(meta.get("scripts", [])),
    )


def discover_skills(root: Path) -> list[SkillInfo]:
    """Find all skills in immediate subdirectories of *root*.

    Each subdirectory that contains a valid ``SKILL.md`` is treated as a skill.
    Directories without a ``SKILL.md`` (or with invalid frontmatter) are
    silently skipped.  Results are sorted by skill name.
    """
    if not root.exists():
        return []
    skills: list[SkillInfo] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        skill = load_skill(child)
        if skill is not None:
            skills.append(skill)
    return sorted(skills, key=lambda s: s.name)


# ── Prompt formatting ──────────────────────────────────────────────────────────


def format_skills_for_prompt(skills: list[SkillInfo]) -> str:
    """Return a compact level-1 block for the system prompt.

    Includes one line per skill plus instructions on when and how to read
    the full details.  Designed to be minimal — the full guidance is only
    loaded on demand via ``read_skill``.
    """
    if not skills:
        return ""
    lines = [
        "## Available skills",
        "Call read_skill(name=…) to read the full guide for a skill.",
        "Only call it when the current task matches — skip for simple questions.",
        "",
    ]
    for s in skills:
        lines.append(f"• {s.name}: {s.description}")
    return "\n".join(lines)


# ── read_skill tool factory ────────────────────────────────────────────────────


def make_read_skill_tool(skills: list[SkillInfo]) -> BaseTool:
    """Return a ``read_skill`` FunctionTool closed over *skills*.

    The tool is created dynamically so the skills dict is baked in —
    no runtime lookup through ``ExecutionContext.state`` is needed.
    """
    from agentkit.tools.base import FunctionTool

    skills_map: dict[str, SkillInfo] = {s.name: s for s in skills}

    async def read_skill(context: ExecutionContext, name: str) -> str:
        """Read the full SKILL.md guide for a named skill (level 2).

        The skills list in your instructions (level 1) shows one-line summaries.
        Call this tool ONLY when the current task matches a skill and you need
        the detailed strategy — do not read skills that are not relevant.

        If the skill includes scripts, they are available in the sandbox at
        ``/home/user/skills/<name>/`` (when code_execution is enabled).

        Parameters
        ----------
        name : str
            Skill name exactly as listed in the instructions (e.g. "web-research").
        """
        skill = skills_map.get(name)
        if skill is None:
            available = sorted(skills_map.keys())
            return (
                f"Unknown skill '{name}'. "
                f"Available: {', '.join(available) if available else '(none)'}"
            )
        skill_file = skill.skill_file()
        if not skill_file.exists():
            return f"SKILL.md not found for skill '{name}' at {skill_file}"
        return skill_file.read_text(encoding="utf-8")

    return FunctionTool(read_skill)
